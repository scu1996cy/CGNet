# CGNet: A Correlation-Guided Registration Network for Unsupervised Deformable Image Registration

This is the official pytorch implementation of the paper 'CGNet: A Correlation-Guided Registration Network for Unsupervised Deformable Image Registration'.

By Yuan Chang, Zheng Li and Wenzheng Xu.

The paper is currently under review, and more details will be disclosed once it is accepted.

# Environment

We implement the code on python 3.6, pytorch 1.7.1, torchvision 0.8.2, and simpleitk 2.1.1.

# Train and Infer

First, before executing training or inference using commands, please ensure that the dataset address has been modified. Then, run 'python train_CGNet.py' or 'python infer_CGNet.py'.

# Dataset

Experiments are conducted on four publicly brain MRI datasets: Mindboggle [Mindboggle dataset official link](https://osf.io/nhtur/), LONI Probabilistic Brain Atlas [LPBA dataset official link](https://resource.loni.usc.edu/resources/atlases-downloads/), Open Access Series of Imaging Studies (OASIS) [OASIS dataset official link](https://sites.wustl.edu/oasisbrains/]), and IXI [IXI dataset official link](https://brain-development.org/ixi-dataset/).

# Baseline Methods

[VoxelMorph](https://github.com/voxelmorph/voxelmorph)

[CycleMorph](https://github.com/boahK/MEDIA_CycleMorph)

[Dual-PRNet](https://github.com/anonymous2024slnet/SLNet/blob/main/models/PRNet.py)

[Dual-PRNet++](https://github.com/anonymous2024slnet/SLNet/blob/main/models/PRNet.py)

[Swin-VoxelMorph](https://github.com/YongpeiZhu/Swin-VoxelMorph/tree/master)

[ViT-V-Net](https://github.com/junyuchen245/ViT-V-Net_for_3D_Image_Registration_Pytorch)

[TransMorph](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration)

[XMorpher](https://github.com/Solemoon/XMorpher)

[Deformer](https://github.com/CJSOrange/DMR-Deformer)

[Im2grid](https://github.com/anonymous2024slnet/SLNet/blob/main/models/Im2grid.py)

[ModeT](https://github.com/anonymous2024slnet/SLNet/blob/main/models/ModeT.py)

[TransMatch](https://github.com/tzayuan/TransMatch_TMI)
