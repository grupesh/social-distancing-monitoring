from scipy.optimize import linear_sum_assignment as linear_assignment

import numpy as np
import numpy.linalg as la
import scipy.io as sio
import pomcommon as pom
import pickle
import cv2
import pylibconfig2 as libcfg
import os
import copy


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


def load_ground_truth(path, frames, K, Pcw):
    '''loading ground truth'''
    gt_raw = pickle.load(open(path, 'rb')) 
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

def filter_gt(gt, filter_func):
    updated_gt = copy.deepcopy(gt)
    for fid in gt:
        for tid, h in gt[fid].items():
            if filter_func(h):
                updated_gt[fid][tid]['allow_miss'] = True
    return updated_gt

def load_detections_pom(seq_path, start_fid, end_fid, p_threshold = 1e-4, scale=1.0):
    '''
    seq_path - path with the pom output for the sequence
    start_fid, end_fid - frame range
    p_threshold - do not load detections below this probability
    scale - resize the bounding box'''

    seq_tag = os.path.basename(os.path.normpath(seq_path))
    room = libcfg.Config(open(seq_path + '%s.cfg' % seq_tag).read()).room
    bboxes = load_bboxes_pom(seq_path + 'bboxes.txt', scale)
    dets = {}
    for fid in range(start_fid, end_fid):
        dets[fid] = {}
        for l in open(seq_path + 'pom/%06d.txt' % fid).readlines():
            lid, prob = map(float,l.split('\t'))
            if prob > p_threshold:
                tid = len(dets[fid])
                pw = pom.lid_to_world(lid,
                                      room.plane.cols, room.plane.rows,
                                      room.plane.cell_width, room.plane.cell_height)
                dets[fid][tid] = {'location': lid,
                                  'pw': pw,
                                  'bbox': bboxes[lid],
                                  'confidence': prob }
    return dets


def load_detections_pom_3d(seq_path, start_fid, end_fid, p_threshold = 1e-4, scale=1.0):
    '''
    seq_path - path with the pom output for the sequence
    start_fid, end_fid - frame range
    p_threshold - do not load detections below this probability
    scale - resize the bounding box'''

    seq_tag = os.path.basename(os.path.normpath(seq_path))
    room = libcfg.Config(open(seq_path + '%s.cfg' % seq_tag).read()).room

    rows = room.cube.rows
    cols = room.cube.cols
    size = room.cube.size

    def lid_to_world_3d(lid):
        z = int(lid) / (rows * cols);
        y = (int(lid) - z * (rows * cols)) / rows
        x = (int(lid) - z * (rows * cols)) % rows
        return np.array([(x + 0.5) * size, (y + 0.5) * size, (z + 0.5) * size, 1.0])


    bboxes = load_bboxes_pom(seq_path + 'bboxes.txt', scale)
    dets = {}
    for fid in range(start_fid, end_fid):
        dets[fid] = {}
        for l in open(seq_path + 'pom/%06d.txt' % fid).readlines():
            lid, prob = map(float,l.split('\t'))
            if prob > p_threshold:
                tid = len(dets[fid])
                pw = lid_to_world_3d(lid)
                dets[fid][tid] = {'location': lid,
                                  'pw': pw,
                                  'bbox': bboxes[lid],
                                  'confidence': prob }
    return dets

def load_detections_kinect2(dets_format, start_fid, end_fid, Pcw):

    RELEVANT_JOINTS = set(['SpineBase','SpineMid','Neck','Head','ShoulderRight','ShoulderLeft','HipLeft','HipRight'])

    def parse_joint(l):
        vals = l.split('\t')
        joint_type, result_type = vals[:2]
        coords = np.array(map(float, vals[2:]))
        pi = np.round(coords[3:5]).astype(np.int)
        pc = coords[:3] * 1000
        pw = pom.camera_to_world(pc, Pcw)
        return joint_type, {'pi':pi,
                            'pc':pc,
                            'pw':pw,
                            'result_type':result_type}

    hypotheses = {}
    for fid in range(start_fid, end_fid):
        hypotheses[fid] = {}
        lines = open(dets_format % fid).readlines()
        tid = 0
        for line in range(0, 6*26, 26):
            if int(lines[line]) == 0:
                continue
            joints = dict(map(parse_joint, lines[line+1:line+26]))

            # averaging over relevant joints (limbs are not very reliable)
            pw = np.array([0,0,0,1.0])
            for jtype in RELEVANT_JOINTS:
                joint = joints[jtype]
                pw[:2] += joint['pw'][:2]
            pw[:2] /= len(RELEVANT_JOINTS)

            # getting bbox around all the joints
            pts_i = np.vstack([j['pi'] for j in joints.values()])

            hypotheses[fid][tid] = {'joints': joints,
                                    'bbox': bb_pts(pts_i),
                                    'pw': pw}
            tid += 1
    return hypotheses


# reading the output of munaro detector
def load_detections_pcl(dets_path, K, Pcw, min_confidence = -100):

    def points_to_bbox(pts_c):
        pts_i = [K.dot(p) for p in pts_c]
        pts_i = np.array([pi[:2] / pi[2] for pi in pts_i])
        xmin, xmax = np.min(pts_i[:,0]), np.max(pts_i[:,0])
        ymin, ymax = np.min(pts_i[:,1]), np.max(pts_i[:,1])
        return map(int,(xmin,ymin,xmax-xmin+1,ymax-ymin+1))

    dets = {}
    for l in open(dets_path).readlines():
        values = map(float, l.split(','))
        fid = int(values[0])
        dets[fid] = {}

        num_detections = int(values[1])

        values = values[2:]
        step = (4*3+1)

        for d in range(0,step*num_detections,step):
            confidence = values[d]
            points = np.array([values[d+1:d+step]]).reshape((-1,3))

            bb_points = np.array([points[0,:] - [0.2,0,0],
                                  points[0,:] + [0.2,0,0],
                                  points[1,:] + [0.2,0,0],
                                  points[1,:] - [0.2,0,0]])

            points = np.vstack([points, bb_points])

            tid = len(dets[fid])
            dets[fid][tid] = {'pw': Pcw.dot(np.hstack((points[1]*1000,1))),
                              'bbox': points_to_bbox(points),
                              'confidence': confidence}
    return dets

def load_bboxes_pom(path, scale=2):
    """scale - multiply bbox by this"""
    bboxes = {}
    for l in open(path):
        lid,x,y,w,h = map(int, l.split('\t'))
        bboxes[lid] = map(lambda x: scale*x, (x,y,w,h))
    return bboxes


def load_bboxes_dollar(bbs_format, start_fid, end_fid, max_confidence=80.0):
    '''load detections from .mat bounding boxes'''
    dets = {}
    for fid in range(start_fid, end_fid+1):
        bbs = sio.loadmat(bbs_format % fid)['bbs']
        dets[fid] = {}
        for did, bb in enumerate(bbs):
            dets[fid][did] = {'bbox': bb[:4].astype(np.int),
                              'confidence': min(bb[4],max_confidence)}
    return dets


def draw_bbox(frame, bbox, color=[255,0,0], thickness=2):
    x,y,w,h = bbox
    cv2.rectangle(frame, (x,y), (x+w,y+h), color, thickness)


def threshold_dets(dets, threshold):
    '''threshold detections at a given confidence threshold
    e.g. leave only those det['condfidence'] > threshold '''
    new_dets = {}
    for fid, fdets in dets.items():
        new_dets[fid] = {}
        for did, det in fdets.items():
            if det['confidence'] > threshold:
                new_dets[fid][did] = det
    return new_dets
