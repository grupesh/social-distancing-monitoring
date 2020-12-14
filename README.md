# Six Feet Apart
## Stereo Vision-based Social Distancing Monitoring

In this repo, we present a prototype system that can detect and analyze social distancing from an image scene, using an RGB-D pedestrian image dataset from EPFL lab dataset (https://www.epfl.ch/labs/cvlab/data/data-rgbd-pedestrian/).
We used a novel CNN detector, EfficientDet-d7, to find pedestrians’ locations in the RGB image, and then leveraged the provided camera calibration matrices and depth map to accurately find their 3D locations in the world coordinate system.

To run our code, run ece_detect.py with the corresponding path to the dataset location.
The `detector` branch implements the efficientdet-d7 detector module. To run efficientdet detections, run the `efficientdet_detector` function
from `run_efficientdet_detector.py` with list of image paths to frames.

The `visualization.py` file has utility functions, used in `ece_detect.py` to visualize the scene, detections, and distance violations, for both stereo and IPM detection modes. 

#### Demo of IPM people and violation detector (left) and stereo vision people and violation detector (right)
![](demo.gif)