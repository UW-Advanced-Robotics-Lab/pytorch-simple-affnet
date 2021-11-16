# PyTorch-Simple-AffordanceNet

The code is developed to simply and implement [AffordanceNet](https://github.com/nqanh/affordance-net) in PyTorch (AffordanceNet was developed in Caffe).
Here is the [original paper for AffordanceNet](https://arxiv.org/pdf/1709.07326.pdf).

I based this work on [TorchVision](https://github.com/pytorch/vision) and [PyTorch-Simple-MaskRCNN](https://github.com/Okery/PyTorch-Simple-MaskRCNN). 

I used pytorch-simple-affnet with the following repos:

1. [LabelFusion](https://github.com/RobotLocomotion/LabelFusion) for generating real images.
2. [NDDS](https://github.com/NVIDIA/Dataset_Synthesizer) for generating synthetic images.
3. [arl-affpose-dataset-utils](https://github.com/UW-Advanced-Robotics-Lab/arl-affpose-dataset-utils) a custom dataset that I generated.
4. [densefusion](https://github.com/UW-Advanced-Robotics-Lab) for predicting an object 6-DoF pose.
5. [arl-affpose-ros-node](https://github.com/UW-Advanced-Robotics-Lab/arl-affpose-ros-node): for deploying our network for 6-DoF pose estimation with our ZED camera.
6. [barrett_tf_publisher](https://github.com/UW-Advanced-Robotics-Lab/barrett-wam-arm) for robotic grasping experiments. Specifically barrett_tf_publisher and barrett_trac_ik. 

In the sample below we see the differences between traditional Object Instance Segmentation (left) and Object-based Affordance Detection (right).
![Alt text](samples/AffPose.png?raw=true "Title")
