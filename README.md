# RockNet
Image classification research
## 👋 Introduction
This project is about rock classification. We construct a CNN-based deep progressive learning model named RockNet to identify lithology from multi-view heterogeneous rock microscopic images.  
RockNet is fused to enhance the perception of local details and global information of rocks in the image. In order to make up for the feature loss caused by the down-sampling of the model in the process of training, RockNet uses global-local feature fusion to optimize the model from the loss. In order to make better use of the data information, we proposes K-fold verification substitution based on the idea of k-fold cross-validation, which alternately replaces the training set and the verification set during the training process. In order to reduce the negative impact of category imbalance, the training process combines category equilibrium sampling and multiple data augmentation, which not only solves the problem of category imbalance, but also generates more different training data and improves the generalization ability of the model.
## 👀 Notice！
👀The training script in this article is **train.py**, and the training time is multi-GPU training.   
👀The prediction script is **predict.py**.  
👀The command to be executed during training is: **CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --use_env train.py** CUDA_VISIBLE_DEVICES is the GPU device you need to work with, nproc_per_node indicates the number of GPUs to which it applies.  
👀The command to be executed during the test is: **python3 predict.py**
