# PyTorch-Simple-AffNet

The code is developed to simply and implement [AffordanceNet](https://github.com/nqanh/affordance-net) in PyTorch (AffordanceNet was developed in Caffe).
Here is the [original paper for AffordanceNet](https://arxiv.org/pdf/1709.07326.pdf).

I based this work on [TorchVision](https://github.com/pytorch/vision) and [PyTorch-Simple-MaskRCNN](https://github.com/Okery/PyTorch-Simple-MaskRCNN). I use AffNet with the following repos:

1. [Labelusion](https://github.com/akeaveny/LabelFusion) for generating Real Images
2. [NDDS](https://github.com/NVIDIA/Dataset_Synthesizer) for generating Synthetic Images   
3. [DenseFusion](https://github.com/akeaveny/DenseFusion) for predicting 6-DoF Object Pose.

In the sample below we see the differences between traditional Object Instance Segmentation (left) and Object-based Affordance Detection (right).
![Alt text](samples/AffPose.png?raw=true "Title")

## Requirements
   ```
   conda env create -f environment.yml --name MaskRCNNTorch
   ```

## AffNet
1. To inspect dataset statistics run (first look at relative paths for root folder of dataset in cfg.py):
   ```
   python scripts/inspect_dataset_stats.py
   ```
2. To test a forward pass of my model:
   ```
   python scripts/test_model_forward_pass.py
   ```
3. To run training:
   ```
   python scripts/train.py
   ```
4. To get predicted Object-Based Affordance-Detection Masks run (preformance is evaluated with the weighted F-b measure in MATLAB):
   ```
   python scripts/eval.py
   ```
