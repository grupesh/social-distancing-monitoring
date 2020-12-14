# social-distancing-monitoring
Six Feet Apart : Stereo Vision-based Social Distancing Monitoring

In this repo, we develop a prototype system that can detect and analyze social distancing from the scene
using an RGB-D pedestrian image dataset from EPFL lab dataset (https://www.epfl.ch/labs/cvlab/data/data-rgbd-pedestrian/).
We used a novel CNN detector, EfficientDet-d7, to find pedestrians’ locations in the RGB image,
and then leveraged the camera calibration matrices and the depth map to accurately find their 3D locations in the world coordinate.

To run our code, run ece_detect.py with the corresponding path to the dataset location.
The detector branch implements the efficientdet-d7 detector module. To run efficientdet detections, run efficientdet_detector function
from run_efficientdet_detector.py with list of image paths to frames.

The visualization branch implements the 3D visualization for stereo and IPM detections.
![](demo.gif)