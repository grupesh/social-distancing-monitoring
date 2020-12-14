# The purpose here is to take a point in the image and return a 3D point associated with it
# This can be 3D in camera coordinate system or in world
import numpy as np
import yaml
import cv2
import numpy.linalg as la

# Calibration Operations
def getGlobal3DfromCamera3D(p,ex_mat):

    # Multiply current point by inverse
    # Note we add a homogenous coordinate to point3d in order to use the extrinsic matrix
    point3d = np.hstack([p, 1])
    point3d = ex_mat.dot(point3d)
    return point3d

def getCamera3DfromImage(p, depth, in_mat):
    #So the first thing that needs to be done is to get the Camera 3D from the image
    #This requires knowing the distance to the object in the camera frame (which can be obtained by stereo)
    focal_x = in_mat[0,0]
    focal_y = in_mat[1,1]
    offset_x = in_mat[0,2]
    offset_y = in_mat[1,2]

    # Get camera based values (without depth adjustment)
    #Note that in our point we have y as the 1st dimension and x as the second
    xc = (p[0] - offset_x)/focal_x
    yc = (p[1] - offset_y)/focal_y

    # Adjust for depth
    xc = xc * depth
    yc = yc * depth

    # Create point3dc for depth adjusted 3d point in camera frame
    #Note that at this point we have x as the 1st dimension and y as the second
    point3dc = np.array([xc,yc,depth])
    return point3dc.T

def getUndistortImage(raw_image, in_mat, dist_coeffs):
    undist_image = np.copy(raw_image)
    cv2.undistort(raw_image, in_mat, dist_coeffs, undist_image)
    return undist_image

if __name__ == '__main__':
    # visualize_camera_cal_oxford_town()
    print("Hello World")
    # 0) Load Matrices
    with open('calibration.yaml') as f:
        yaml_list = yaml.load(f)

    # Intrinsic
    in_mat = np.array(yaml_list['K'])
    print(in_mat.shape)
    print(in_mat)
    # Camera to World Extrinsic
    c2w_mat = np.array(yaml_list['Pcw'])
    print(c2w_mat.shape)
    print(c2w_mat)
    #Distortion Coefficients
    dist_coeffs = np.array(yaml_list['dist'])
    print(dist_coeffs.shape)
    print(dist_coeffs)


    # 1) Load Raw Image
    raw_img = cv2.imread('rgb000000.png',cv2.IMREAD_COLOR)
    cv2.imshow("Image",raw_img)

    depth_img = cv2.imread('depth000000.png',cv2.IMREAD_UNCHANGED)
    print(depth_img.shape)
    print(depth_img[0:2,0:2])
    cv2.imshow("Depth Image", depth_img)
    cv2.waitKey(1)

    # 2) Transform to Undistorted
    # This data already looks like it was ran through some anti distortion function so skip
    # REMOVE THIS IN THE ACTUAL IMPLEMENTATION
    undist_img = getUndistortImage(raw_img, in_mat, dist_coeffs)
    undist_depth_img = getUndistortImage(depth_img, in_mat, dist_coeffs)
    cv2.imshow("Image", undist_img)
    cv2.imshow("Depth Image", undist_depth_img)
    cv2.imwrite("rgbundist000000.png",undist_img)
    cv2.imwrite("depthundist000000.png",undist_depth_img)
    cv2.waitKey(1)

    # 3) Select a Pixel
    #142x,170y
    # point = np.array([170,142]).T
    point = np.array([341,263]).T

    select_img = cv2.circle(undist_img,(263,341),2,(255,0,0),-1)
    cv2.imshow("Image", select_img)
    cv2.waitKey(0)

    # 4) Get Distance at that Pixel from Depth Map
    depth = undist_depth_img[341,263]

    #Note for these matrices we assume a RDF coordinate frame since the data
    #comes from Kinect and that is the Kinect frame
    # 5) Transform from Pixel to Camera 3D
    # (y,x)

    #(x,y,z)
    point3dc = getCamera3DfromImage(point, depth, in_mat)
    print("Camera Frame 3D Point")
    print(point3dc)
    print(point3dc.shape)
    # 6) Transform from Camera Frame to Global Frame
    point3d = getGlobal3DfromCamera3D(point3dc,c2w_mat)
    print("Global Frame 3D Point")
    print(point3d)

