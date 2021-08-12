# PyTorch-Simple-AffNet

The code is developed to simply and implement [AffordanceNet](https://github.com/nqanh/affordance-net) in PyTorch (AffordanceNet was developed in Caffe).
Here is the [original paper for AffordanceNet](https://arxiv.org/pdf/1709.07326.pdf).

I based this work on [TorchVision](https://github.com/pytorch/vision) and [PyTorch-Simple-MaskRCNN](https://github.com/Okery/PyTorch-Simple-MaskRCNN). 

I use AffNet with the following repos:

1. [Labelusion](https://github.com/akeaveny/LabelFusion) for generating Real Images
2. [NDDS](https://github.com/NVIDIA/Dataset_Synthesizer) for generating Synthetic Images   
3. [DenseFusion](https://github.com/akeaveny/DenseFusion) for predicting 6-DoF Object Pose.
4. AffDenseFusionROSNode: coming soon.
5. [Barrett_WAM_Arm_tf_publisher](https://github.com/akeaveny/Barrett_WAM_Arm_tf_publisher) for transforming object pose reference frame to the base link of our 7-DoF manipulator.

In the sample below we see the differences between traditional Object Instance Segmentation (left) and Object-based Affordance Detection (right).
![Alt text](samples/AffPose.png?raw=true "Title")