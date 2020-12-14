import cv2
import numpy as np
import numpy.linalg as la
import pandas as pd

import glob
import yaml

from scipy.optimize import linear_sum_assignment as lsa

from CalibrationOperations import getCamera3DfromImage
from CalibrationOperations import getGlobal3DfromCamera3D

import evaluation as evals


def getIPM(K, Pwc):
    """
    compute an IPM matrix from calibration matrices

    The projection operation by 3x4 matrix 
    [1 0 0 0]
    [0 1 0 0]
    [0 0 1 0] is not invertible. 
    Therefore the true world coordinate [x y z 1]' from the image coorindate [i j 1]' 
    can only be recovered with the depth map. 
     
    For IPM, however, the constraint z = 0 allows us to recover x and y without any loss of information.
    
    Since [i j 1]' = K_3x3 [R_3x3 | T_3x1] [x y 0 1]' is equivalent to [i j 1]' = K [R_3x2 | T_3x1] [x y 1]', 
    the IPM matrix is inv(K [R[:3, :2] | T[:3, 0]]).      
    """

    R = Pwc[:3,:3]
    T = Pwc[:3,3].reshape((3,1))
    H = np.linalg.inv(K @ np.concatenate((R[:, :2], T), axis=1))

    return H


def runIPMonSingleImage(img_in, ipm_matrix, multiplier=20):
    h = img_in.shape[0]
    w = img_in.shape[1]

    img_ipm = cv2.warpPerspective(img_in, ipm_matrix, (multiplier*w, multiplier*h))

    # convert to the original size for the visualization
    img_ipm = cv2.resize(img_ipm, (w, h))

    return img_ipm


def getBCfromBbox(bbox):
    # return bottom center coordinate of the given bounding box (xmin, ymin, width, height)
    x_bc = int(bbox[0] + bbox[2]/2)
    y_bc = int(bbox[1] + bbox[3])

    # homogeneous coordinate
    pc = np.array([[x_bc, y_bc, 1]]).T

    return pc


def getCentroidfromBbox(bbox, mode):
    if mode == 'centroid':
        # return centroid coordinate of the given bounding box (xmin, ymin, width, height)
        x_bc = int(bbox[0] + bbox[2] / 2)
        y_bc = int(bbox[1] + bbox[3] / 2)
    elif mode == 'face':
        # return estimation of face location (1/8 from the top - center)
        x_bc = int(bbox[0] + bbox[2] / 2)
        y_bc = int(bbox[1] + bbox[3] / (8/3))
    else:
        raise Exception("Invalid centroid mode! (getCentroidfromBbox) Should be either of centroid / face")

    # homogeneous coordinate
    pc = np.array([[x_bc, y_bc, 1]]).T

    return pc


def matchBbox(bbox0, bbox1):
    # Hungarian matching of the bipartite graph between two sets of bounding boxes
    # weight (cost) is the distance between bounding boxes
    # if #row > #col, it will return #row of indices, and vice versa

    cost_matrix = np.zeros((len(bbox0), len(bbox1)))
    for row, bbox_gt_ in enumerate(bbox0):
        center_x_gt = bbox_gt_[0]+bbox_gt_[2]/2
        center_y_gt = bbox_gt_[1]+bbox_gt_[3]/2

        for col, bbox_eff_ in enumerate(bbox1):
            center_x_eff = bbox_eff_[0]+bbox_eff_[2]/2
            center_y_eff = bbox_eff_[1]+bbox_eff_[3]/2

            cost_matrix[row, col] = ((center_x_gt-center_x_eff)**2+(center_y_gt-center_y_eff)**2)**0.5

    row_ind, col_ind = lsa(cost_matrix)

    return row_ind, col_ind


def retrieveGT(gt, num_img=950):
    bbox_gt = []        # gt bbox
    pw_gt = []          # gt world coordinate
    pc_gt = []          # gt bottom center of bbox

    for idx_frame in range(num_img):
        bbox_gt_ = []
        pw_gt_ = []
        pc_gt_ = []

        if bool(gt[idx_frame]) is True:
            for key in gt[idx_frame]:
                bbox_gt_.append(gt[idx_frame][key]['bbox'])
                pw_gt_.append(gt[idx_frame][key]['pw'])

            for bbox_ in bbox_gt_:
                pc_gt_.append(getBCfromBbox(bbox_))

        else:
            bbox_gt_.append([])
            pw_gt_.append([])
            pc_gt_.append([])

        # sort with the x-center of bbox
        if len(bbox_gt_[0]) > 0:
            bb = np.array(bbox_gt_)
            pw = np.array(pw_gt_)
            pc = np.array(pc_gt_)

            bb_sorted = bb[np.argsort(bb[:,0] + bb[:,2]/2), :]
            pw_sorted = pw[np.argsort(bb[:,0] + bb[:,2]/2), :]
            pc_sorted = pc[np.argsort(bb[:,0] + bb[:,2]/2), :]

        else:
            bb_sorted = np.array([])
            pw_sorted = np.array([])
            pc_sorted = np.array([])

        bbox_gt.append(bb_sorted)
        pw_gt.append(pw_sorted)
        pc_gt.append(pc_sorted)

    return bbox_gt, pw_gt, pc_gt


def getPwfromPc_IPM(pc_in, H_ipm, num_img=950):
    # perform IPM on Pc to compute its world coordinate on z=0 plane
    pw_ipm = []

    for idx_frame in range(num_img):
        pw_ipm_ = []

        for pc in pc_in[idx_frame]:
            if len(pc) > 0:
                pw_ipm_inhomo = np.insert(H_ipm @ pc, 2, 0)
                pw_ipm_.append(pw_ipm_inhomo / pw_ipm_inhomo[-1])
            else:
                pw_ipm_.append([])

        pw_ipm.append(pw_ipm_)

    return pw_ipm


def computeDistErr_IPM(bbox_gt, bbox_in, pw_gt, pw_in1, num_img=950):
    distance_pw0, distance_pw1 = [], []
    error = []
    avg_error = []

    for idx_frame in range(num_img):
        error_ = []
        distance_pw0_ = []
        distance_pw1_ = []

        if len(bbox_gt[idx_frame]) > 0:
            if np.array_equal(bbox_gt[idx_frame], bbox_in[idx_frame]) is True:
                pw0 = np.array(pw_gt[idx_frame])
                pw1 = np.array(pw_in1[idx_frame])

            else:
                # align points in world coordinate from gt and efficientdet by matching their bounding boxes
                row_ind, col_ind = matchBbox(bbox_gt[idx_frame], bbox_in[idx_frame])

                # use only matched boxes
                pw0 = np.array(pw_gt[idx_frame])[row_ind]
                pw1 = np.array(pw_in1[idx_frame])[col_ind]

        else:
                pw0 = np.array([])
                pw1 = np.array([])

        if len(pw0) > 0:
            for i in range(len(pw0)):
                for j in range(i+1, len(pw1)):
                    distance0 = np.linalg.norm(pw0[i] - pw0[j])
                    distance1 = np.linalg.norm(pw1[i] - pw1[j])
                    distance_pw0_.append(distance0)
                    distance_pw1_.append(distance1)

                    error_.append((distance0 - distance1) / distance0 * 100)

            if len(error_) > 0:
                avg_error.append(np.mean(np.array(error_)))
            else:
                avg_error.append(np.nan)

        else:
            # won't add to the error
            avg_error.append(np.nan)

        distance_pw0.append(distance_pw0_)
        distance_pw1.append(distance_pw1_)

        error.append(error_)

    return distance_pw0, distance_pw1, error, avg_error


def computeDistErr_Depth(bbox_gt, bbox_in, pw_gt, K, Pcw, frames_depth, mode, num_img=950):
    distance_gt, distance_in = [], []
    error = []
    avg_error = []

    for idx_frame in range(num_img):
        error_ = []
        distance_gt_ = []
        distance_in_ = []

        if len(bbox_gt[idx_frame]) > 0:
            if np.array_equal(bbox_gt[idx_frame], bbox_in[idx_frame]) is True:
                # this is only for comparing ground truth pw to the world coordinate that is computed from the bounding box
                pw_gt_aligned = np.array(pw_gt[idx_frame])
                bbox_gt_matched = np.array(bbox_gt[idx_frame])
                bbox_in_matched = np.array(bbox_in[idx_frame])

            else:
                # align points in world coordinate from gt and efficientdet by matching their bounding boxes
                row_ind, col_ind = matchBbox(bbox_gt[idx_frame], bbox_in[idx_frame])

                # use only matched boxes
                pw_gt_matched = np.array(pw_gt[idx_frame])[row_ind]
                #bbox_gt_matched = np.array(bbox_gt[idx_frame])[row_ind]
                bbox_in_matched = np.array(bbox_in[idx_frame])[col_ind]
        else:
            pw_gt_matched = np.array([])
            bbox_gt_matched = np.array([])
            bbox_in_matched = np.array([])

        if len(pw_gt_matched) > 0:
            # if there is any object in the frame
            for i in range(len(pw_gt_matched)):
                for j in range(i+1, len(bbox_in_matched)):
                    pc_in_obj0 = getCentroidfromBbox(bbox_in_matched[i], mode)
                    pc_in_obj1 = getCentroidfromBbox(bbox_in_matched[j], mode)

                    # convert (x,y,1) -> (y,x,1)
                    pc_in_obj0 = np.array([pc_in_obj0[1], pc_in_obj0[0]]).astype('int')
                    pc_in_obj1 = np.array([pc_in_obj1[1], pc_in_obj1[0]]).astype('int')

                    pcw_in_obj0 = getCamera3DfromImage(pc_in_obj0, frames_depth[idx_frame][int(pc_in_obj0[0]), int(pc_in_obj0[1])], K).astype('float')
                    pcw_in_obj1 = getCamera3DfromImage(pc_in_obj1, frames_depth[idx_frame][int(pc_in_obj1[0]), int(pc_in_obj1[1])], K).astype('float')

                    # (optional) Use extrinsic matrix to map into the world coordinate. The distance is preserved, so we do not need to perform this to get the distancing.
                    #pw_in_obj0 = getGlobal3DfromCamera3D(pcw_in_obj0, Pcw).astype('float')
                    #pw_in_obj1 = getGlobal3DfromCamera3D(pcw_in_obj1, Pcw).astype('float')

                    distance_gt__ = np.linalg.norm(pw_gt_matched[i] - pw_gt_matched[j])
                    distance_in__ = np.linalg.norm(pcw_in_obj0 - pcw_in_obj1)

                    distance_gt_.append(distance_gt__)
                    distance_in_.append(distance_in__)

                    error_.append((distance_gt__ - distance_in__) / distance_gt__ * 100)

            if len(error_) > 0:
                avg_error.append(np.mean(np.array(error_)))
            else:
                avg_error.append(np.nan)

        else:
            # won't add to the error
            avg_error.append(np.nan)

        distance_gt.append(distance_gt_)
        distance_in.append(distance_in_)

        error.append(error_)

    return distance_gt, distance_in, error, avg_error


def computeDistErr_Depth_Projected(bbox_gt, bbox_in, pw_gt, K, Pcw, frames_depth, mode, num_img=950):
    distance_gt, distance_in = [], []
    error = []
    avg_error = []

    for idx_frame in range(num_img):
        error_ = []
        distance_gt_ = []
        distance_in_ = []

        if len(bbox_gt[idx_frame]) > 0:
            if np.array_equal(bbox_gt[idx_frame], bbox_in[idx_frame]) is True:
                # this is only for comparing ground truth pw to the world coordinate that is computed from the bounding box
                pw_gt_aligned = np.array(pw_gt[idx_frame])
                bbox_gt_matched = np.array(bbox_gt[idx_frame])
                bbox_in_matched = np.array(bbox_in[idx_frame])

            else:
                # align points in world coordinate from gt and efficientdet by matching their bounding boxes
                row_ind, col_ind = matchBbox(bbox_gt[idx_frame], bbox_in[idx_frame])

                # use only matched boxes
                pw_gt_matched = np.array(pw_gt[idx_frame])[row_ind]
                #bbox_gt_matched = np.array(bbox_gt[idx_frame])[row_ind]
                bbox_in_matched = np.array(bbox_in[idx_frame])[col_ind]
        else:
            pw_gt_matched = np.array([])
            bbox_gt_matched = np.array([])
            bbox_in_matched = np.array([])

        if len(pw_gt_matched) > 0:
            # if there is any object in the frame
            for i in range(len(pw_gt_matched)):
                for j in range(i+1, len(bbox_in_matched)):
                    pc_in_obj0 = getCentroidfromBbox(bbox_in_matched[i], mode)
                    pc_in_obj1 = getCentroidfromBbox(bbox_in_matched[j], mode)

                    # convert (x,y,1) -> (y,x,1)
                    pc_in_obj0 = np.array([pc_in_obj0[1], pc_in_obj0[0]]).astype('int')
                    pc_in_obj1 = np.array([pc_in_obj1[1], pc_in_obj1[0]]).astype('int')

                    pcw_in_obj0 = getCamera3DfromImage(pc_in_obj0, frames_depth[idx_frame][int(pc_in_obj0[0]), int(pc_in_obj0[1])], K).astype('float')
                    pcw_in_obj1 = getCamera3DfromImage(pc_in_obj1, frames_depth[idx_frame][int(pc_in_obj1[0]), int(pc_in_obj1[1])], K).astype('float')

                    # (optional) Use extrinsic matrix to map into the world coordinate. The distance is preserved, so we do not need to perform this to get the distancing.
                    pw_in_obj0 = getGlobal3DfromCamera3D(pcw_in_obj0, Pcw).astype('float')
                    pw_in_obj1 = getGlobal3DfromCamera3D(pcw_in_obj1, Pcw).astype('float')

                    if idx_frame == 786:
                        print(pw_in_obj0)
                        print(pw_in_obj1)



                    # project pw_in to z = 0, to eliminate the height difference between centroids.
                    pw_in_obj0 = pw_in_obj0[:2]
                    pw_in_obj1 = pw_in_obj1[:2]

                    distance_gt__ = np.linalg.norm(pw_gt_matched[i] - pw_gt_matched[j])
                    distance_in__ = np.linalg.norm(pw_in_obj0 - pw_in_obj1)

                    distance_gt_.append(distance_gt__)
                    distance_in_.append(distance_in__)

                    error_.append((distance_gt__ - distance_in__) / distance_gt__ * 100)

            if len(error_) > 0:
                avg_error.append(np.mean(np.array(error_)))
            else:
                avg_error.append(np.nan)

        else:
            # won't add to the error
            avg_error.append(np.nan)

        distance_gt.append(distance_gt_)
        distance_in.append(distance_in_)

        error.append(error_)

    return distance_gt, distance_in, error, avg_error


def loadBboxfromEffDet(eff_det):
    bbox_effdet = []

    # parse the result from 'efficientdet_detections.txt' and unpack into list
    idx = 0
    idx_max = 3829

    for i in range(num_img):

        bbox_effdet_frame = []
        if idx+1 > idx_max:
            pass
        elif eff_det[0][idx+1][0] == 'I':
            bbox_effdet_frame.append([])
            idx += 1
        else:
            bbox_effdet_frame.append([float(num) for num in eff_det[0][idx+1].split(',')])

            if idx+2 > idx_max:
                pass
            elif eff_det[0][idx+2][0] == 'I':
                idx += 2
            else:
                bbox_effdet_frame.append([float(num) for num in eff_det[0][idx+2].split(',')])
                if idx+3 > idx_max:
                    pass
                elif eff_det[0][idx+3][0] == 'I':
                    idx += 3
                else:
                    bbox_effdet_frame.append([float(num) for num in eff_det[0][idx+3].split(',')])

                    if idx+4 > idx_max:
                        pass
                    elif eff_det[0][idx+4][0] == 'I':
                        idx += 4
                    else:
                        bbox_effdet_frame.append([float(num) for num in eff_det[0][idx+4].split(',')])

                        if idx+5 > idx_max:
                            pass
                        elif eff_det[0][idx+5][0] == 'I':
                            idx += 5
                        else:
                            bbox_effdet_frame.append([float(num) for num in eff_det[0][idx+5].split(',')])
                            idx += 6
                            # will have to think how to rule out wrong bounding box... or how to compare it with ground truth

        # remove scores
        bbox_effdet_frame = np.array(bbox_effdet_frame)[:, 1:]

        # rearrange bounding boxes from: (bbox_ymin, bbox_xmin, bbox_ymax, bbox_xmax) -> (xmin, ymin, width, height)
        if len(bbox_effdet_frame[0]) > 0:
            bb = bbox_effdet_frame[:, [1,0,3,2]]

            width = (bb[:, 2] - bb[:, 0]).reshape((bb.shape[0],1))
            height = (bb[:, 3] - bb[:, 1]).reshape((bb.shape[0],1))

            bb = np.insert(bb, [2], width, axis=1)
            bb = (np.insert(bb, [3], height, axis=1)[:, :4])

            # sort with the x-center of bbox
            bb = bb[np.argsort(bb[:,0] + bb[:,2]/2), :].astype('int')

        else:
            bb = np.array([])

        bbox_effdet.append(bb.tolist())

    return bbox_effdet


def loadBboxfromRCNN(path_rcnn):
    bbox_rcnn = np.load(path_rcnn, allow_pickle=True)
    bbox_rcnn_out = []

    num_img = bbox_rcnn.shape[0]

    for i in range(num_img):
        bbox_rcnn_frame = bbox_rcnn[i]

        # if bbox is not empty
        if bbox_rcnn_frame.size > 0:
            # remove scores
            bbox_rcnn_frame = bbox_rcnn_frame[:, :4]

            # rearrange bounding boxes from: (bbox_ymin, bbox_xmin, bbox_ymax, bbox_xmax) -> (xmin, ymin, width, height)
            bb = bbox_rcnn_frame[:, [1,0,3,2]]

            width = (bb[:, 2] - bb[:, 0]).reshape((bb.shape[0],1))
            height = (bb[:, 3] - bb[:, 1]).reshape((bb.shape[0],1))

            bb = np.insert(bb, [2], width, axis=1)
            bb = (np.insert(bb, [3], height, axis=1)[:, :4])

            # sort with the x-center of bbox
            bb = bb[np.argsort(bb[:,0] + bb[:,2]/2), :].astype('int')

        else:
            bb = np.array([])

        bbox_rcnn_out.append(bb.tolist())

    return bbox_rcnn_out

if __name__ == '__main__':

    """ 
    define path for input (rgb & depth) image, calibration matrices, and ground truth 
    """
    # epfl_lab rgbd dataset
    dataset_path_lab = r'./epfl_cvlab_stereo/epfl_lab'          # this is a local path; please change it according to your actual path. (we don't upload the dataset to the repo)
    tag = '20140804_160621_00'
    num_img = 950   # epfl_lab

    # epfl_lab calibration matrices
    path_calib = './calibration/calibration.yaml'

    # epfl_lab ground truth
    gt_path_lab = r'./calibration/lab_20140804_160621_00_gt3d.pickle'

    # EfficientDet detection results
    eff_detections = pd.read_csv('./efficientdet_detections.txt', sep='\t', header=None)

    # load calibration matrices
    yaml_ = yaml.safe_load(open(path_calib, 'r'))
    K = np.array(yaml_['K'])
    Pcw = np.array(yaml_['Pcw'])
    Pwc = np.array(yaml_['Pwc'])
    dist = np.array(yaml_['dist'])

    # generate ipm matrix and get intrinsic & distortion
    H_ipm = getIPM(K, Pwc)

    """
    Load all depth image frames and retrieve ground truth information
    """
    # load all depth images and undistort it
    frames_depth = [cv2.undistort(cv2.imread(f, cv2.IMREAD_UNCHANGED), K, dist, None) for f in sorted(glob.glob("%s/%s/depth*.png"%(dataset_path_lab, tag)))]

    # load ground truth
    gt_dict = evals.load_ground_truth(gt_path_lab, frames_depth, K, Pcw)

    # retrieve bounding box and world coordinate
    bbox_gt, pw_gt, pc_gt = retrieveGT(gt_dict)

    '''
    This part was only for testing IPM using ground truth Bbox and ground truth world coordinates
    """
    Use IPM to retrieve p_w from gt bbox (this will be compared with p_w from efficientdet bbox)
    """

    # Now, transform the bottom center in image into the world coordinate using IPM matrix
    pw_ipm_gt = getPwfromPc_IPM(pc_gt, H_ipm)

    # Computer the error between:
    # The distance from the IPM using the ground truth bbox, and the distance from the ground truth world coordinate
    distance_gt, distance_ipm_gt, _, avg_err_ipm_gt = computeDistErr_IPM(bbox_gt, bbox_gt, pw_gt, pw_ipm_gt)
    print("Average distance error between the ground truth and IPM from gt bbox: %1.3f " % np.nanmean(np.absolute(np.array(avg_err_ipm_gt)))+"%")    # avg error < 5%
    '''

    """
    Task 1: Use EfficientDet
    """
    # retrieve bbox from the textfile and get the world coordinate
    bbox_effdet = loadBboxfromEffDet(eff_detections)

    """
    Task 1-1: Get the world coordinate from EfficientDet Bbox using IPM and compute the error against the ground truth
    """
    pc_effdet = []
    for bbox_frame in bbox_effdet:
        pc_effdet_ = []
        if len(bbox_frame) > 0:
            for bbox in bbox_frame:
                pc_effdet_.append(getBCfromBbox(bbox))
        else:
            pc_effdet_.append([])

        pc_effdet.append(pc_effdet_)

    # get the pw from ipm
    pw_ipm_effdet = getPwfromPc_IPM(pc_effdet, H_ipm)

    # compute the distance error
    _, _, error_effdet, avg_err_effdet = computeDistErr_IPM(bbox_gt, bbox_effdet, pw_gt, pw_ipm_effdet)
    print("Average distance error between the ground truth and IPM from efficientdet bbox: %1.3f" % np.nanmean(np.absolute(avg_err_effdet))+"%")    # avg error < 5%

    """
    Task 1-2: Get the camera coordinate from EfficientDet Bbox & depth map and compute the error against the ground truth
    """

    # compute the distance error
    _, _, err_effdet_rgbd, avg_err_effdet_rgbd = computeDistErr_Depth(bbox_gt, bbox_effdet, pw_gt, K, Pcw, frames_depth, 'centroid', num_img=950)
    print("Average distance error between the ground truth and true world coordinate from efficientdet bbox: %1.3f" % np.nanmean(np.absolute(avg_err_effdet_rgbd))+"%")    # avg error < 5%

    # compute the distance error from the projected 3d points
    _, _, _, avg_err_effdet_rgbd_prj = computeDistErr_Depth_Projected(bbox_gt, bbox_effdet, pw_gt, K, Pcw, frames_depth, 'centroid', num_img=950)
    print("Average distance error between the ground truth and projected world coordinate from efficientdet bbox: %1.3f" % np.nanmean(np.absolute(avg_err_effdet_rgbd_prj))+"%")    # avg error < 5%

    """
    Task 2: Use Faster R-CNN
    """
    # retrieve bbox from the textfile and get the world coordinate
    bbox_rcnn = loadBboxfromRCNN('rcnn_detections.npy')

    """
    Task 2-1: Get the world coordinate from R-CNN Bbox using IPM and compute the error against the ground truth
    """
    pc_rcnn = []
    for bbox_frame in bbox_rcnn:
        pc_rcnn_ = []
        if len(bbox_frame) > 0:
            for bbox in bbox_frame:
                pc_rcnn_.append(getBCfromBbox(bbox))
        else:
            pc_rcnn_.append([])

        pc_rcnn.append(pc_rcnn_)

    # get the pw from ipm
    pw_ipm_rcnn = getPwfromPc_IPM(pc_rcnn, H_ipm)

    # compute the distance error
    _, _, error_rcnn, avg_err_rcnn = computeDistErr_IPM(bbox_gt, bbox_rcnn, pw_gt, pw_ipm_rcnn)
    print("Average distance error between the ground truth and IPM from R-CNN bbox: %1.3f" % np.nanmean(np.absolute(avg_err_rcnn))+"%")    # avg error < 5%


    """
    Task 2-2: Get the camera coordinate from R-CNN Bbox & depth map and compute the error against the ground truth
    """

    # compute the distance error
    _, _, err_rcnn_rgbd, avg_err_rcnn_rgbd = computeDistErr_Depth(bbox_gt, bbox_rcnn, pw_gt, K, Pcw, frames_depth, 'centroid', num_img=950)
    print("Average distance error between the ground truth and true world coordinate from R-CNN bbox: %1.3f" % np.nanmean(np.absolute(avg_err_rcnn_rgbd))+"%")    # avg error < 5%

    # compute the distance error from projected 3d points
    _, _, _, avg_err_rcnn_rgbd_prj = computeDistErr_Depth_Projected(bbox_gt, bbox_rcnn, pw_gt, K, Pcw, frames_depth, 'centroid', num_img=950)
    print("Average distance error between the ground truth and projected world coordinate from R-CNN bbox: %1.3f" % np.nanmean(np.absolute(avg_err_rcnn_rgbd_prj))+"%")    # avg error < 5%



    print("End of Program")