
import cv2
import numpy as np
import numpy.linalg as la
from transformations import translation_matrix

# def camera_to_image(K, dist, pc):
#   uv, rvecs = cv2.projectPoints(np.array([pc]), np.zeros((3,1)), np.zeros((3,1)), K, dist)
#   return np.round(uv[0][0])
# def lid_to_world(lid):
#     # convert location id to world coordinates of the cell
#     row = int(lid / cols)
#     col = int(lid % cols)
#     return np.array([col * cell_width, row * cell_height, 0, 1])

# def world_to_camera(pw):
#     pc = P.dot(pw + shift)
#     return pc[:3]
# def world_to_image(pw):
#     pc = world_to_camera(pw)
#     return camera_to_image(K, dist, pc[:3]).astype(np.int)

BLUE_COLOR = [1,0,0]
GREEN_COLOR = [0,1,0]
RED_COLOR = [0,0,1]
YELLOW_COLOR = [0,1,1]

def to_rgb_frame(frame):
    return (255 * cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR).astype(np.double) / 10000.0).astype(np.uint8)


def draw_grid(img, K, Pwc,  length=500, rows=5, cols=5, h = 0, color=[0,255,0], rad=2, thickness=2):
    for x in range(0, length*rows, length):
        for y in range(0, length*cols, length):
            p0 = world_to_image([x,y,h,1], K, Pwc)
            p1 = world_to_image([x+length,y,h,1], K, Pwc)
            p2 = world_to_image([x,y+length,h,1], K, Pwc)
            cv2.circle(img, p0, rad, color, thickness)

def image_to_camera(u, v, Z, K):
    """converts image -> camera, no distortion"""
    X = (u - K[0,2]) * Z / K[0,0]
    Y = (v - K[1,2]) * Z / K[1,1]
    return np.array([X, Y, Z]).T


def estimate_plane(pts, prop_points=0.3):
    """infer a plane given a set of 3D points
    pts - Nx3 array of points
    prop_points - the proportion of points sampled from the dense region
    """

    # now, random subsample
    rind = np.random.choice(len(pts), len(pts) * prop_points)

    centroid = np.mean(pts[rind], axis=0)
    PTS = (pts - centroid).T

    u,s,vh = la.svd(PTS.dot(PTS.T))
    n = - vh.conj().transpose()[:,-1]
    bias = -np.inner(n, centroid)

    return np.hstack((n, bias)), centroid


def estimate_camera_to_plane(plane, pts, p0, p1):
    """estimate an affine transform from camera to plane
    plane - 4x1 3D plane equation
    pts - Nx3 points on the plane in camera coordinates
    p0 - the new origin in world coordinates
    """

    # TODO: maybe, select this point in a smarter way
    a = (p1 - p0)
    a /= la.norm(a)

    b = np.cross(a, plane[:3])
    b /= la.norm(b)

    # projecting the points to the new basis
    proj_pts = np.inner(pts - p0, [a, b, plane[:3]])

    ret, T, val = cv2.estimateAffine3D(pts, proj_pts, confidence=1.0)
    return np.vstack((T, [0, 0, 0, 1]))


def calibrate_extrinsics_depth(frame,K,rect,pi0,pi1,shift,z_direction=-1):
    """calibrating depth camera w.r.t. ground plane
    frame - depth frame
    K - camera matrix
    rect - [u,v,w,h] image rectangle that contains plane points
    pi0 - (u,v) image point corresponding to the 0,0,0
    pi1 - (u,v) image point corresponding to the direction 0,1,0
    z_directon - -1 or 1, for the direction of the z
    returns Pwc, Pcw - transformation matrices world->camera, camera->world
    """
    u0,v0,h,w = rect
    uv = np.array(zip(*(x.flat for x in np.mgrid[u0:u0+h,v0:v0+w])))

    z = frame[uv[:,0], uv[:,1]].reshape((-1,1))

    indices = np.where(z != 0)[0]

    uvz = np.hstack([uv[indices], z[indices]])
    pts_c = image_to_camera(uvz[:,1], uvz[:,0], uvz[:,2], K)
    plane, ctr = estimate_plane(pts_c, 1)
    plane *= z_direction

    # the direction for X (rows)
    u,v = pi0
    p0 = image_to_camera(v, u, frame[u,v], K)
    u,v = pi1
    p1 = image_to_camera(v, u, frame[u,v], K)

    Pcw = estimate_camera_to_plane(plane, pts_c, p0, p1)
    # tune this to move coordinate system
    T = translation_matrix(np.array(shift))
    Pwc = la.pinv(Pcw).dot(T)
    Pcw = la.pinv(Pwc)

    return Pwc, Pcw

def world_to_image(p_w, K, Pwc):
    """converts world -> image, no distortion"""
    p = K.dot(Pwc[:-1,:].dot(p_w))
    p = (p / p[2]).astype(np.int)
    return (p[0], p[1])

def camera_to_world(p_c, Pcw):
    return Pcw.dot(np.hstack([p_c,1]))

# TODO: depend on (P, K, dist, shift)
def lid_to_world(lid, cols, rows, cell_width, cell_height):
    # convert location id to world coordinates of the cell
    row = int(lid / cols)
    col = int(lid % cols)
    return np.array([col * cell_width, row * cell_height, 0, 1])

def lid_to_cell_center(lid):
    # convert location id to cell center in world coordinates
    row, col = int(lid / cols), int(lid % cols)
    return np.array([(col+0.5) * cell_width, (row+0.5) * cell_height, 0, 1])

def print_matrix(P, name='P'):
    pstr = name + " = ["
    for i in range(0,P.shape[0]):
        for j in range(0,P.shape[1]):
            pstr += "%04f," % P[i,j]
        pstr += "\n"
    pstr = pstr[:-2] + "];"
    print(pstr)
