import torch
import torchvision
import os
import numpy as np
import time
import pickle
import matplotlib.pyplot as plt
from datetime import datetime
from plotting import plot_frame, plot_frame_one_row, get_roi_pts
from calibration.CalibrationOperations import getCamera3DfromImage, getGlobal3DfromCamera3D
from visualization import imagePlaneToWorldCoordStereo, imagePlaneToWorldCoordIPM, run3DVisualizationIPM, run3DVisualizationStereo

import yaml
import cv2

# Not sure if this is still needed for our categories
from utils import COCO_INSTANCE_CATEGORY_NAMES as LABELS
import cv2
np.set_printoptions(precision=4)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")


# Our main program is expected to do a few things:
# Person Location Detection
# 1) Import the video and its depth frame
# 2) Use either EfficientDet OR Yolo to find pedestrian bounding boxes
# 3) Get the bottom bounding box center and convert to 3d via IPM OR get centroid of bounding box and convert with depth image to 3D
# 4) Record all of these points in a vector

# We will end up having a list of 3D detections for each frame

# Anomaly Detection
# 1) For each record of points (per frame list of person points) find the distance between all points and all other points
# 2) If any are below our 'social distancing' threshold, record these on a seperate list of anomalys
# 3) each entry will contain the two points in question and the distance

# Visualization
# 1) We are looking to overaly the rgb00000 IPM image with points from the IPM or Stereo person detect
# 2) We take the anomaly list per frame and use that to draw lines of bad social distancing
def find_violation(pts, dist=2.0):
    """

    :param pts: positions of all pedestrians in a single frame
    :param dist: social distance
    :return: a list of index pairs indicating two pedestrians who are violating social distancing
    """
    n = len(pts)  # number of pedestrians
    pairs = []
    for i in np.arange(0, n, 1):
        for j in np.arange(i+1, n, 1):
            if np.linalg.norm(pts[i] - pts[j]) < dist:
                pairs.append((i, j))
    return pairs

def initPyTorchDetector(device):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.to(device=device)
    model.eval()
    return model

def runPyTorch(raw_img,device,model):
    # convert image from OpenCV format to PyTorch tensor format
    img_t = np.moveaxis(raw_img, -1, 0) / 255
    img_t = torch.tensor(img_t, device=device).float()

    # pedestrian detection
    predictions = model([img_t])
    boxes = predictions[0]['boxes'].cpu().data.numpy()
    classIDs = predictions[0]['labels'].cpu().data.numpy()
    scores = predictions[0]['scores'].cpu().data.numpy()

    return boxes, classIDs, scores

def find3DPeople_IPM(boxes, ipm_mat, pts_world):
    for i in range(len(boxes)):
        # extract the bounding box coordinates
        (x1, y1) = (boxes[i][0], boxes[i][1])
        (x2, y2) = (boxes[i][2], boxes[i][3])

        # find the bottom center position and convert it to world coordinate
        p_c = np.array([[(x1 + x2) / 2], [y2], [1]])
        p_w = ipm_mat @ p_c
        p_w = p_w / p_w[2] # Here they project the point onto the IPM plane: shouldn't it already be their?
        pts_world.append([p_w[0][0], p_w[1][0],p_w[2][0]])

def find3DPeopleStereo(boxes, depth_img, in_mat, ex_mat, pts_world):
    for i in range(len(boxes)):
        # extract the bounding box coordinates
        (x1, y1) = (boxes[i][0], boxes[i][1])
        (x2, y2) = (boxes[i][2], boxes[i][3])

        # Find center of the box
        x_cen = int((x2-x1)/2.0)
        y_cen = int((y2-y1)/2.0)
        point = np.array([x_cen, y_cen]).T
        depth = depth_img[y_cen, x_cen]

        # Lookup the center
        point3dc = getCamera3DfromImage(point, depth, in_mat)
        point3d = getGlobal3DfromCamera3D(point3dc,ex_mat)
        pts_world.append([point3d[0],point3d[1],point3d[2]])

def plotPeopleAndPairs(pts_world, pairs,ax):
    print(len(pts_world))
    if len(pts_world) == 1:
        ax.plot(pts_world[0][0], pts_world[0][1],pts_world[0][2], 'og', alpha=0.5)
    else:
        print(pts_world[:,0])
        ax.plot(pts_world[:, 0], pts_world[:, 1],pts_world[:,2], 'og', alpha=0.5)
    for pair in pairs:
        data = np.array([pts_world[pair[0]], pts_world[pair[1]]])
        ax.plot(data[:, 0], data[:, 1],data[:,2], '-r')

def getBoxesFromGT(frame_num, gt,frame):
    # GT is in min_x, min_y, width, height
    n = len(gt[frame_num])  # number of pedestrians
    raw_boxes = gt[frame_num]
    boxes = []
    for i in np.arange(0, n, 1):
        #Find centroid
        x_min = min(max(    raw_boxes[i][0]     ,0),frame.shape[1]-1)
        y_min = min(max(    raw_boxes[i][1]     ,0),frame.shape[0]-1)

        x_max = min(max(    raw_boxes[i][0] + raw_boxes[i][2]   ,0),frame.shape[1]-1)
        y_max = min(max(    raw_boxes[i][1] + raw_boxes[i][3]   ,0),frame.shape[0]-1)

        box =[x_min,y_min,x_max,y_max]


        boxes.append(box)

    return boxes

def main(yaml_path,gt_path=""):
    print("ECE Social Distancing Detection")

    if gt_path != "":
        with open(gt_path,'rb') as file:
            gt = pickle.load(file)
    test = gt[238]
    test1 = test[0]
    # Make a results folder to contain output
    #path_result = os.path.join('results', data_time + '_' + detector, dataset)
    #os.makedirs(path_result, exist_ok=True)

    # initialize detector
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #model = initPyTorchDetector(device)

    # load background
    #img_bkgd_bev = cv2.imread('calibration/' + dataset + '_background_calibrated.png')

    # load transformation matrices -------------------------------------------------------------------------
    print(yaml_path)

    with open(yaml_path) as f:
        yaml_list = yaml.load(f)

    # Intrinsic
    in_mat = np.array(yaml_list['K'])
    # Camera to World Extrinsic
    ex_mat = np.array(yaml_list['Pcw'])
    Pwc = np.array(yaml_list['Pwc'])


    # open video of dataset
    rgb_cap = cv2.VideoCapture(os.path.join(r'C:\Users\rohan\Documents\repos\epfl_lab\20140804_160621_00', 'rgb%06d.png'))
    #EXAMPLE READ
    #rgb_cap = cv2.VideoCapture(os.path.join('datasets', 'TownCentreXVID.avi'))

    depth_cap = cv2.VideoCapture(os.path.join(r'C:\Users\rohan\Documents\repos\epfl_lab\20140804_160621_00', 'depth%06d.png'))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    i_frame = 0
    # while cap.isOpened() and i_frame < 5000:
    while rgb_cap.isOpened():
        while depth_cap.isOpened():
            ret,rgb_img = rgb_cap.read()
            if ret is False:
                rgb_cap.release()
                depth_cap.release()
                break
            if(rgb_img.shape[2] > 3):
                # Remove alpha channel
                #test = np.zeros((428,512,3), dtype = "uint8")
                rgb_img = cv2.cvtColor(rgb_img,cv2.COLOR_BGRA2BGR)
            ret,depth_img = depth_cap.read()


            if (i_frame < 412):
                i_frame += 1
                continue

            #cv2.imshow("RGB",rgb_img)
            #cv2.imshow("Depth",depth_img)
            #cv2.waitKey(1)

            # Find 3D Person Points -----------------------------------------------------------------------------------
            # For PYTORCH YOLO Implementation
            #boxes, classIDs, scores = runPyTorch(rgb_img, device, model)
            # Here is also where we'd add our version
            # our_boxes = efficientDet(input)
            boxes = getBoxesFromGT(i_frame, gt,rgb_img)

            #Debug: Draw on frame
            #debug_img = np.copy(rgb_img)
            #for box in boxes:
            #    cv2.rectangle(debug_img,(box[0],box[1]),(box[2],box[3]),(255,0,0),3)
            #cv2.imshow("Debug Boxes", debug_img)
            #cv2.waitKey(1)
            #Use IPM
            #ipm_points = []
            ipm_mat = [] # NEEDS TO BE REPLACED WITH OUR IPM MATRIX
            # Here is an example one
            #ipm_mat = np.loadtxt('calibration/' + 'oxford_town' + '_matrix_cam2world.txt')
            #thr_score = 0.9 #Not sure what is appropriate here
            #find3DPeople_IPM(boxes, classIDs, scores, thr_score, ipm_mat, ipm_points)
            #ipm_points = np.array(ipm_points)

            #Use Stereo
            #Here is where we'd get stereo points
            stereo_points = []
            find3DPeopleStereo(boxes, depth_img, in_mat, ex_mat, stereo_points)
            stereo_points = np.array(stereo_points)
            # Find Anomaly Pairs --------------------------------------------------------------------------------------

            from visualization import convertPixelLocTo3DWorldStereo
            world3DCentroids = []
            for box in boxes:
                xmin, ymin, xmax, ymax = box
                xcenter = int((xmin+xmax)/2)
                ycenter = int((ymin+ymax)/2)

                point = convertPixelLocTo3DWorldStereo(ycenter, xcenter, depth_img, in_mat, ex_mat)
                x,y,z = point
                world3DCentroids.append([x,-y,z])
                
            violation_pairs = find_violation(np.array(world3DCentroids), dist=1500)

            person_centroids = []
            for i in range(stereo_points.shape[0]):
                # Get the 3D world x,y,z coordinate
                x = stereo_points[i][0]
                y = -stereo_points[i][1]
                z = stereo_points[i][2]
                person_centroids.append([x,y,z])

            # Create depth map's point cloud
            points3DWorldStereo = imagePlaneToWorldCoordStereo(rgb_img, depth_img, in_mat, ex_mat)
            run3DVisualizationStereo(points3DWorldStereo,world3DCentroids,violation_pairs, i_frame, render=True)


            pts_world = []
            from ipm import generate_ipm_matrix
            H = generate_ipm_matrix(in_mat,Pwc)
            find3DPeople_IPM(boxes, H, pts_world)
            pts_world = np.array(pts_world)


            violation_pairs = find_violation(pts_world, dist=1500)

            # Here is an example of placing dots and anomaly lines
            person_centroids = []
            for i in range(pts_world.shape[0]):
                # Get the 3D world x,y,z coordinate
                x = pts_world[i][0]
                y = pts_world[i][1]
                z = pts_world[i][2]
                person_centroids.append([x,y,z])     

            points3DWorldIPM = imagePlaneToWorldCoordIPM(rgb_img, in_mat, Pwc)
            

            run3DVisualizationIPM(points3DWorldIPM,person_centroids,violation_pairs, i_frame, render=True)
            '''
            plt.cla()
            ax = fig.add_subplot(111, projection='3d')
            if(len(stereo_points) > 0 ):
                plotPeopleAndPairs(stereo_points, violation_pairs,ax)
            fig.savefig(os.path.join('test_results', 'frame%04d.png' % i_frame))
            #plt.draw()
            #plt.show(block=False)
            '''
            print("Frame: ", i_frame)
            i_frame += 1
            if i_frame == 236:
                print("stop")

            # Maintain Anomaly Statistics --------------------------------------------------------------------------




if __name__ == '__main__':
    # We care about
    # a) the yaml file path with the calibration information
    # b) the detection coming from bounding boxes
    #main(yaml_path='calibration/calibration.yaml')
    main(yaml_path='calibration/calibration.yaml',gt_path='bbox_effdet.pkl')