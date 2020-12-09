import os
import sys
import tensorflow.compat.v1 as tf
import numpy as np
path = os.path.join(os.getcwd(),'automl','efficientdet')
os.chdir(path)
sys.path.insert(0,path)

from automl.efficientdet.model_inspect import ModelInspector

MODEL = 'efficientdet-d7'  #@param
ckpt_path = os.path.join(os.getcwd(), MODEL)
saved_model_dir = 'savedmodel'

def efficientdet_detector(image_path_list = ['testdata/img1.jpg'],
                          out_image_folder = 'serve_image_out',
                          min_score_thresh = 0.35,
                          max_boxes_to_draw = 200,
                          line_thickness = 2):

  """
  input :
  image_path_list : list of input image paths
  out_image_folder : path of folder that contains output images
  min_score_thresh : threshold for selecting detections
  max_boxes_to_draw : maximum number of boxes
  line_thickness

  output :
  detections - list of detections for each image
  each list item has shape (num_detections,5)
  with entries [ymin, xmin, ymax, xmax, score],
  for class 'person'

  """
  tf.enable_v2_tensorshape()
  tf.disable_eager_execution()

  all_detections = []

  # create an instance of ModelInspector class
  inspector = ModelInspector(
    model_name=MODEL,
    logdir='/tmp/deff/',
    tensorrt=None,
    use_xla=False,
    ckpt_path=ckpt_path,
    export_ckpt=None,
    saved_model_dir=saved_model_dir,
    tflite_path=None,
    batch_size=1,
    score_thresh=min_score_thresh,
    max_output_size=max_boxes_to_draw,
    nms_method='hard')

  dets = inspector.run_model(
    'saved_model_infer',
    input_image=image_path_list,
    output_image_dir=out_image_folder,
    input_video=None,
    output_video=None,
    line_thickness=line_thickness,
    max_boxes_to_draw=max_boxes_to_draw,
    min_score_thresh=min_score_thresh,
    nms_method='hard',
    bm_runs=10,
    threads=0,
    trace_filename=None)

  for idx, img in enumerate(dets):
    #print("Image {}".format(idx))
    count = 0
    img_detections = []
    for det in img[0]:
      score = det[5]
      class_id = det[6]
      if score >= min_score_thresh and class_id == 1.0:
        img_detections.append([det[1],det[2],det[3],det[4],det[5]])
    all_detections.append(np.array(img_detections))

  return all_detections

if __name__ == '__main__':
  detections = efficientdet_detector()
  for idx, img in enumerate(detections):
    print("Image {}".format(idx))
    for det in img:
      avg_r = int((det[0] + det[2]) / 2)
      avg_c = int((det[1] + det[3]) / 2)
      print("Score: {}, \tClass: Person, \t@ ({}, {})".format(det[4], avg_r, avg_c))