import numpy as np
import yaml
import cv2
import numpy.linalg as la
import matplotlib.pyplot as plt
import math
import pandas as pd
from ipm import generate_ipm_matrix, perform_ipm, load_bbox_effdet_from_txt
from calibration.CalibrationOperations import getGlobal3DfromCamera3D, getCamera3DfromImage, getUndistortImage
import vispy.scene
from vispy.scene import visuals
import vispy.io as io
import sys


def convertPixelLocTo3DWorldStereo(row, col, depthImage, K, Pcw):
    '''
    Returns the (x,y,z) world coordinates of a pixel in an RGB image, using the stereo/depth image.

            Parameters:
                    row (int): A decimal integer; the row of the pixel
                    
                    col (int): A decimal integer; the column of the pixel
                    
                    depthImage (numpy array): The depth image, loaded by cv2
                    
                    K (3x3 numpy array): The intrinsic matrix
                    
                    Pcw (numpy array): The extrinsic matrix
            Returns:
                    point3dWorld (1x3 numpy array): World coordinate (x,y,z)
    '''
    point = np.array([row,col]).T
    depth = depthImage[row,col]
    point3dCam = getCamera3DfromImage(point, depth, K)
    point3dWorld = getGlobal3DfromCamera3D(point3dCam,Pcw)
    x,y,z = point3dWorld[0],point3dWorld[1],point3dWorld[2]
    point3dWorld = np.array([x,y,z])
    return point3dWorld

def convertPixelLocTo3DWorldIPM(row, col, H):
    '''
    Returns the (x,y,z) world coordinates of a pixel in an RGB image, using the ipm approach. The z coordinate will always be 0.

            Parameters:
                    row (int): A decimal integer; the row of the pixel
                    
                    col (int): A decimal integer; the column of the pixel
                    
                    H (numpy matrix): The homographic transform matrix
            Returns:
                    point3dWorld (1x3 numpy array): World coordinate (x,y,z)
    '''
    point3dCam = np.array([col,row,1]).T
    point3dWorld = H @ point3dCam
    point3dWorld /= point3dWorld[-1]
    return [point3dWorld[0],point3dWorld[1],0]

def imagePlaneToWorldCoordStereo(rgbImage, depthImage, K, Pcw):
    '''
    Returns the set of (x,y,z,c) world coordinates (with color) of each pixel in a 2D image, using the depth map.

            Parameters:
                    rgbImage (numpy array): An RGB image loaded by cv2
                    
                    depthImage (numpy array): The RGB image's matching depth map, loaded by cv2
                    
                    K (3x3 numpy array): The intrinsic matrix
                    
                    Pcw (numpy array): The extrinsic matrix
            Returns:
                    points (list): World coordinate with color (x,y,z,[r,g,b])
    '''
    height, width = rgbImage.shape[0:2]

    points = []

    for r in range(150,height):
        for c in range(width):
            point3dWorld = convertPixelLocTo3DWorldStereo(r, c, depthImage, 
                                                            K, Pcw)
            
            avg_clr = rgbImage[r,c]/255
            tmp = avg_clr[0]
            avg_clr[0] = avg_clr[2]
            avg_clr[2] = tmp
            
            if (point3dWorld[0] > 0 and point3dWorld[2] > 0):
                points.append([point3dWorld[0], -point3dWorld[1], point3dWorld[2], avg_clr])
    return points

def imagePlaneToWorldCoordIPM(rgbImage, depthImage, K, Pcw):
    '''
    Returns the set of (x,y,z,c) world coordinates (with color) of each pixel in a 2D image, using the ipm approach.

            Parameters:
                    rgbImage (numpy array): An RGB image loaded by cv2
                    
                    depthImage (numpy array): The RGB image's matching depth map, loaded by cv2
                    
                    K (3x3 numpy array): The intrinsic matrix
                    
                    Pcw (numpy array): The extrinsic matrix
            Returns:
                    points (list): World coordinate with color (x,y,z,[r,g,b])
    '''
    height, width = rgbImage.shape[0:2]
    points = []
    
    H = generate_ipm_matrix(K, Pwc)

    for r in range(150, height):
        for c in range(width):
            point3dWorld = convertPixelLocTo3DWorldIPM(r, c, H)
            avg_clr = rgbImage[r,c]/255
            tmp = avg_clr[0]
            avg_clr[0] = avg_clr[2]
            avg_clr[2] = tmp

            if (point3dWorld[0] > -1000 and math.sqrt(point3dWorld[0]**2 + point3dWorld[1]**2) < 8000):
                points.append([point3dWorld[0], -point3dWorld[1], 0, avg_clr])

    return points

def run3DVisualization(depthPoints, ipmPoints, centroids, violations):
    '''
    Takes all the world coordinates and their color values and plots them in 3D space, using vispy. Also draws cylinders around 3D points corresponding to person detections' bounding boxes. Also draws 3D lines between the cylinders that represent a pair of people which are violating the 6' restriction.

            Parameters:
                    depthPoints (list): World coordinates with color, generated by the depth/stereo image (x,y,z,[r,g,b])

                    ipmPoints (list): World coordinates with color, generated by the ipm method (x,y,z,[r,g,b])

                    centroids (list): A list of lists, where each inner list object is a 3D world point, representing the centroid of a person (found using the efficientdet bounding boxes), in format [x,y,z]
                    
                    violations (list): A list of lists, where each inner list is a pair of integers, which are two indices in the bBoxes list representing a pair of people violating the 6' restriction. Formatted as [[pi1,pi2],[pi3,pi4]]
                    
            Returns:
    '''
    
    # Create canvas to draw everything
    canvas = vispy.scene.SceneCanvas(keys='interactive', show=True)
    view = canvas.central_widget.add_view()

    # Unpack depth and ipm points
    depthPos = [] 
    depthColor = []
    for point in depthPoints:
        depthPos.append([point[0], point[1], point[2]])
        r = point[3][0]
        g = point[3][1]
        b = point[3][2]
        depthColor.append([r,g,b])

    # Unpack ipm points
    ipmPos = []
    ipmColor = []
    for point in ipmPoints:
        ipmPos.append([point[0], point[1], point[2]])
        r = point[3][0]
        g = point[3][1]
        b = point[3][2]
        ipmColor.append([r,g,b])

    # Concatenate the two lists of points, for a joint 3D plot
    pos = depthPos + ipmPos
    pos = np.array(pos)
    colors = depthColor + ipmColor
    colors = np.array(colors)

    # 3D scatter plot to show depth map pointcloud
    scatter = visuals.Markers()
    scatter.set_data(pos, edge_color=None, face_color=colors, size=5)
    view.add(scatter)

    
    # Draw cylinders around centroids
    for point in centroids:
        x,y,z = point

        cyl_mesh = vispy.geometry.create_cylinder(10,10,radius=[500,500],length=50)

        # Move cylinder to correct location in 3D space
        # Make sure to negate the y value, otherwise everything will be mirrored
        vertices = cyl_mesh.get_vertices()
        center=np.array([x,-y,z],dtype=np.float32)
        vtcs = np.add(vertices,center)
        cyl_mesh.set_vertices(vtcs)

        cyl = visuals.Mesh(meshdata=cyl_mesh, color='g')
        view.add(cyl)


    # Draw lines between violating people
    for pair in violations:
        x1,y1,z1 = centroids[pair[0]]
        x2,y2,z2 = centroids[pair[1]]
        #lin = visuals.Line(pos=np.array([[x1,-y1,z1+1],[x2,-y2,z2+1]]), color='r', method='gl')
        #view.add(lin)
        tube = visuals.Tube(points=np.array([[x1,-y1,z1],[x2,-y2,z2]]), radius=50, color='red')
        view.add(tube)
    
    view.camera = 'turntable'  # or try 'arcball'


    view.camera.elevation = 11.0#21.5
    view.camera.azimuth = -103.5#-92.5
    view.camera.distance = 7500
    if (len(ipmPoints) > 0):
        view.camera.elevation = 39.0#21.5
        view.camera.azimuth = -83.0#-92.5
        view.camera.distance = 10980.75


    # Add a colored 3D axis for orientation
    axis = visuals.XYZAxis(parent=view.scene)
    #img = canvas.render()
    #io.write_png("test.png",img)
    vispy.app.run()
    print(view.camera.elevation, view.camera.azimuth, view.camera.distance)
    

if __name__ == "__main__":
    # Get the calibration matrices
    with open('calibration/calibration.yaml') as f:
        yaml_list = yaml.load(f)

    K = np.array(yaml_list['K'])
    Pcw = np.array(yaml_list['Pcw'])
    Pwc = np.array(yaml_list['Pwc'])
    dist_coeffs = np.array(yaml_list['dist'])

    # Read in an image to perform the visualization on
    raw_img = cv2.imread(r'C:\Users\rohan\Documents\repos\epfl_lab\20140804_160621_00\rgb000397.png',cv2.IMREAD_COLOR)
    depth_img = cv2.imread(r'C:\Users\rohan\Documents\repos\epfl_lab\20140804_160621_00\depth000397.png',cv2.IMREAD_UNCHANGED)

    # The known bounding boxes efficientdet found for frame 397
    bboxes = []
    bboxes.append([0.9391376972198486,183.533203125,154.81825256347656,399.5244140625,250.7292938232422])
    bboxes.append([0.9221212267875671,213.22979736328125,314.9386901855469,398.277099609375,378.55963134765625])
    bboxes.append([0.7441033124923706,192.76376342773438,219.3519287109375,349.45172119140625,265.079833984375])
    bboxes.append([0.6069592833518982,199.52059936523438,331.6083984375,256.5639343261719,366.62689208984375])

    points3DWorldStereo = imagePlaneToWorldCoordStereo(raw_img, depth_img, K, Pcw)

    points3DWorldIPM = imagePlaneToWorldCoordIPM(raw_img, depth_img, K, Pcw)

    # Get the 3D world centroids for the bounding boxes
    world3DCentroids = []
    for box in bboxes:
        score, ymin, xmin, ymax, xmax = box
        xcenter = int((xmin+xmax)/2)
        ycenter = int((ymin+ymax)/2)

        point = convertPixelLocTo3DWorldStereo(ycenter, xcenter, depth_img, K, Pcw)

        x,y,z = point
        world3DCentroids.append([x,y,z])

    violations = [[0,1],[2,3]]
    run3DVisualization(points3DWorldStereo, [], world3DCentroids, violations)

    # Get the 3D world centroids for the bounding boxes
    H = generate_ipm_matrix(K,Pwc)
    world3DCentroids = []
    for box in bboxes:
        score, ymin, xmin, ymax, xmax = box
        xcenter = int((xmin+xmax)/2)
        ycenter = ymax

        point = convertPixelLocTo3DWorldIPM(ycenter, xcenter, H)

        x,y,z = point
        world3DCentroids.append([x,y,z])

    violations = [[0,1],[2,3]]
    run3DVisualization([], points3DWorldIPM, world3DCentroids, violations)




