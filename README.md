# social-distancing-monitoring
Six Feet Apart : Stereo Vision-based Social Distancing Monitoring

In this repo, we develop a prototype system that can detect and analyze social distancing from the scene
using an RGB-D pedestrian image dataset from EPFL lab dataset (https://www.epfl.ch/labs/cvlab/data/data-rgbd-pedestrian/).
We used a novel CNN detector, EfficientDet-d7, to find pedestrians’ locations in the RGB image,
and then leveraged the camera calibration matrices and the depth map to accurately find their 3D locations in the world coordinate.
