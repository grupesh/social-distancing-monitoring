import cv2
import numpy as np
import numpy.linalg as la
import pandas as pd

import glob

import yaml
import pickle

import pomcommon as pom

"""
IPM and distance calculation (w/ IPM)
"""


def generate_ipm_matrix(K, Pwc):
    """
    compute IPM matrix from calibration matrices

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


def perform_ipm(img_in, ipm_matrix, multiplier=20):
    h = img_in.shape[0]
    w = img_in.shape[1]

    img_ipm = cv2.warpPerspective(img_in, ipm_matrix, (multiplier*w, multiplier*h))

    # convert to the original size for the visualization
    img_ipm = cv2.resize(img_ipm, (w, h))

    return img_ipm


def get_btm_center_from_bbox(bbox):

    x_bc = int(bbox[0] + bbox[2]/2)
    y_bc = int(bbox[1] + bbox[3])

    pc = np.array([[x_bc, y_bc, 1]]).T

    return pc


"""
Retrieve EPFL RGBD dataset ground truth from the pickle file
"""
# methods below are from the EPFL RGBD dataset ground truth file. Required to generate bounding box from their ground truth pickle file.
def bb_pw(pw, K, Pwc):
    '''make a bounding box given world coordinate on the ground plane'''
    p = np.zeros([4,4])
    p[0] = pw - np.array([0,250,100,0])
    p[1] = pw + np.array([0,250,-100,0])
    p[2] = pw - np.array([0,250,-1750,0])
    p[3] = pw + np.array([0,250,1750,0])

    pts_i = []
    for i in range(0,4):
        pts_i.append(pom.world_to_image(p[i], K, Pwc))
    return np.vstack(pts_i)


def bb_pts(pts_i):
    '''construct a bounding box given a list of points'''
    xmin = np.min(pts_i[:,0])
    xmax = np.max(pts_i[:,0])
    ymin = np.min(pts_i[:,1])
    ymax = np.max(pts_i[:,1])
    return [xmin, ymin, xmax-xmin+1, ymax-ymin+1]


def load_ground_truth_single_frame(gt_path, dataset_path, tag, fid, K, Pcw):

    # slightly modified version to access ground truth bbox of only one frame
    gt_raw = pickle.load(open(gt_path, 'rb'))
    gt = {}

    Pwc = la.pinv(Pcw)

    for oid in gt_raw[fid]:
        pts_i = np.vstack(gt_raw[fid][oid]['points_image'])

        frame = [cv2.imread(f, cv2.IMREAD_UNCHANGED) for f in sorted(glob.glob("%s/%s/depth%06d.png" % (dataset_path, tag, fid)))][0]

        #z = frames[fid][pts_i[:,1],pts_i[:,0]]
        z = frame[pts_i[:,1],pts_i[:,0]]

        pts_c = pom.image_to_camera(pts_i[:,0], pts_i[:,1], z, K)
        pts_c = np.hstack([pts_c, np.ones((pts_c.shape[0],1))])
        pts_w = pts_c.dot(Pcw.T)
        pts_w[:,2] = 0

        pw = np.mean(pts_w, 0)
        pc = np.mean(pts_c, 0)

        pi = pom.world_to_image(pw, K, Pwc)
        bbox = bb_pts(np.vstack([pts_i, pi, bb_pw(pw, K, Pwc)]))

        gt[oid] = {'pw': pw,
                    'pc': pc,
                    'bbox': bbox,
                    'num_pts': len(pts_i),
                    'pts_i': pts_i}
    return gt


def load_ground_truth(gt_path, frames, K, Pcw):
    '''loading ground truth'''
    gt_raw = pickle.load(open(gt_path, 'rb'))
    gt = {}

    Pwc = la.pinv(Pcw)
    for fid in gt_raw:
        gt[fid] = {}
        for oid in gt_raw[fid]:
            pts_i = np.vstack(gt_raw[fid][oid]['points_image'])
            z = frames[fid][pts_i[:,1],pts_i[:,0]]
            pts_c = pom.image_to_camera(pts_i[:,0], pts_i[:,1], z, K)
            pts_c = np.hstack([pts_c, np.ones((pts_c.shape[0],1))])
            pts_w = pts_c.dot(Pcw.T)
            pts_w[:,2] = 0

            pw = np.mean(pts_w, 0)
            pc = np.mean(pts_c, 0)

            pi = pom.world_to_image(pw, K, Pwc)
            bbox = bb_pts(np.vstack([pts_i, pi, bb_pw(pw, K, Pwc)]))

            gt[fid][oid] = {'pw': pw,
                            'pc': pc,
                            'bbox': bbox,
                            'num_pts': len(pts_i),
                            'pts_i': pts_i}
    return gt


def get_bbox_pw_pc_from_ground_truth(gt, num_img=950):
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
                pc_gt_.append(get_btm_center_from_bbox(bbox_))

        else:
            bbox_gt_.append([])
            pw_gt_.append([])
            pc_gt_.append([])

        bbox_gt.append(bbox_gt_)
        pw_gt.append(pw_gt_)
        pc_gt.append(pc_gt_)

    return bbox_gt, pw_gt, pc_gt


def get_pw_from_pc(pc_in, H_ipm, num_img=950):
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


def compute_distance_and_error(pw_in0, pw_in1, num_img=950):
    distance_pw0, distance_pw1 = [], []
    error = []

    for idx_frame in range(num_img):
        for i in range(len(pw_in0[idx_frame])):
            for j in range(i+1, len(pw_in0[idx_frame])):

                # have to account for exclusive cases where efficientdet cannot detect all the people in the frame

                distance_pw0_ = np.linalg.norm(pw_in0[idx_frame][i] - pw_in0[idx_frame][j])
                distance_pw1_ = np.linalg.norm(pw_in1[idx_frame][i] - pw_in1[idx_frame][j])
                distance_pw0.append(distance_pw0)
                distance_pw1.append(distance_pw1)

                error.append((distance_pw0_ - distance_pw1_) / distance_pw0_ * 100)

    average_error = np.average(np.array(error))

    return distance_pw0, distance_pw1, error, average_error


def load_bbox_effdet_from_txt(eff_det):
    bbox_effdet = []

    # parse the result and unpack into list
    idx = 0
    for i in range(num_img):

        bbox_effdet_frame = []
        if eff_det[0][idx+1][0] == 'I':
            bbox_effdet_frame.append([])
            idx += 1
        else:
            bbox_effdet_frame.append([int(float(num)) for num in eff_det[0][idx+1].split(',')[1:]])
            if eff_det[0][idx+2][0] == 'I':
                idx += 2
            else:
                bbox_effdet_frame.append([int(float(num)) for num in eff_det[0][idx+2].split(',')[1:]])
                if eff_det[0][idx+3][0] == 'I':
                    idx += 3
                else:
                    bbox_effdet_frame.append([int(float(num)) for num in eff_det[0][idx+3].split(',')[1:]])
                    if eff_det[0][idx+4][0] == 'I':
                        idx += 4
                    else:
                        bbox_effdet_frame.append([int(float(num)) for num in eff_det[0][idx+4].split(',')[1:]])
                        idx += 5
        bbox_effdet.append(bbox_effdet_frame)

    return bbox_effdet


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

    """
    main code
    """
    # load a single frame rgb image for test
    frame_id = 718
    img = [cv2.imread(f, cv2.IMREAD_UNCHANGED) for f in sorted(glob.glob("%s/%s/rgb%06d.png" % (dataset_path_lab, tag, frame_id)))][0]

    # load calibration matrices
    yaml_ = yaml.safe_load(open(path_calib, 'r'))
    K = np.array(yaml_['K'])
    Pcw = np.array(yaml_['Pcw'])
    Pwc = np.array(yaml_['Pwc'])
    dist = np.array(yaml_['dist'])

    # generate ipm matrix and get intrinsic & distortion
    H_ipm = generate_ipm_matrix(K, Pwc)

    # Undistort the input image using the distortion coefficients
    # Note: I think this undistorted image makes straight lines looking more straight... needs double check.
    img = cv2.undistort(img, K, dist, None)

    # get a bird's eye view
    img_ipm = perform_ipm(img, H_ipm, 16)
    img_ipm = cv2.rotate(img_ipm, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)

    """
    Load all depth image frames to retrieve ground truth 
    """
    # load all depth images
    frames = [cv2.imread(f, cv2.IMREAD_UNCHANGED) for f in sorted(glob.glob("%s/%s/depth*.png"%(dataset_path_lab, tag)))]

    # load ground truth
    #gt_dict = load_ground_truth_single_frame(gt_path_lab, dataset_path_lab, tag, frame_id, K, Pcw)     # this is loading gt of a single frame for testing purpose.
    gt_dict = load_ground_truth(gt_path_lab, frames, K, Pcw)

    # retrieve bounding box and world coordinate
    bbox_gt, pw_gt, pc_gt = get_bbox_pw_pc_from_ground_truth(gt_dict)

    """
    Use IPM to retrieve p_w from gt bbox (this will be compared with p_w from efficientdet bbox)
    """

    # Now, transform the bottom center in image into the world coordinate using IPM matrix
    pw_ipm_gt = get_pw_from_pc(pc_gt, H_ipm)

    # Computer the error between:
    # The distance from the IPM using the ground truth bbox, and the distance from the ground truth world coordinate
    distance_gt, distance_ipm_gt, _, avg_err_ipm_gt = compute_distance_and_error(pw_gt, pw_ipm_gt)
    print("Average distance error between the ground truth and IPM from gt bbox: %1.3f %" % avg_err_ipm_gt+"%")    # avg error < 5%

    """
    Now, load bounding boxes from EfficientDet and compute the error between distances from IPM and ground truth world coordinate for all frames
    """
    # retrieve bbox from the textfile and get the world coordinate
    bbox_effdet = load_bbox_effdet_from_txt(eff_detections)

    pc_effdet = []
    for bbox_frame in bbox_effdet:
        pc_effdet_ = []
        if len(bbox_frame[0]) > 0:
            for bbox in bbox_frame:
                pc_effdet_.append(get_btm_center_from_bbox(bbox))
        else:
            pc_effdet_.append([])

        pc_effdet.append(pc_effdet_)

    # get the pw from ipm
    pw_ipm_effdet = get_pw_from_pc(pc_effdet, H_ipm)

    # compute the error
    # TODO: Revise 'compute_distance_and_error' for the case that EfficientDet couldn't detect all people in the frmae

    #_, distance_ipm_effdet, _, avg_err_ipm_effdet = compute_distance_and_error(pw_gt, pw_ipm_effdet)
    #print("Average distance error between the ground truth and IPM from gt bbox: %1.3f %" % avg_err_ipm_effdet+"%")    # avg error < 5%

    """
    TODO: Compute IOU btw bbox_gt and bbox_effdet
    """

    # TODO: compare the distance from RGB+depth map to the ground truth (or to the IPM)
    # TODO: Integrate with the visualization in the social distancing detector

    # display
    while 1:
        cv2.imshow('image', img_ipm)

        key = cv2.waitKey(20) & 0xFF
        if key == ord("q"):
            break
