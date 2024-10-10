# CGNet: A Correlation-Guided Registration Network for Unsupervised Deformable Image Registration

This is the official pytorch implementation of the paper 'CGNet: A Correlation-Guided Registration Network for Unsupervised Deformable Image Registration'.

By Yuan Chang, Zheng Li and Wenzheng Xu.

The paper is currently under review, and more details will be disclosed once it is accepted.

# Environment

We implement the code on ```python 3.6```, ```pytorch 1.7.1```, ```torchvision 0.8.2```, and ```simpleitk 2.1.1```.

# Train and Infer

Step 1: Before executing training or inference using commands, please ensure that the dataset address has been modified: ``` train_dir = 'Path_to_OASIS_data/Train/' ```, ``` val_dir = 'Path_to_OASIS_data/Val/' ```, and ``` test_dir = 'Path_to_OASIS_data/Test/' ```.
Step 2: ```python train_CGNet.py``` to train the CGNet. 
Step 3: ```python infer_CGNet.py``` to test the CGNet.
Modifying "batch_size," "lr," and "weights" in the training code can adjust the batch size, learning rate, and loss weights used for image similarity measure loss and deformation field regularization loss.

# Dataset

Experiments are conducted on four publicly brain MRI datasets: Mindboggle [link](https://osf.io/nhtur/), LONI Probabilistic Brain Atlas (LPBA) [link](https://resource.loni.usc.edu/resources/atlases-downloads/), Open Access Series of Imaging Studies (OASIS) [link](https://sites.wustl.edu/oasisbrains/]), and IXI [link](https://brain-development.org/ixi-dataset/).

# Baseline Method

We compare our CGNet with twelve baseline methods: [VoxelMorph](https://github.com/voxelmorph/voxelmorph), [CycleMorph](https://github.com/boahK/MEDIA_CycleMorph), [Dual-PRNet](https://github.com/anonymous2024slnet/SLNet/blob/main/models/PRNet.py), [Dual-PRNet++](https://github.com/anonymous2024slnet/SLNet/blob/main/models/PRNet.py), [Swin-VoxelMorph](https://github.com/YongpeiZhu/Swin-VoxelMorph/tree/master), [ViT-V-Net](https://github.com/junyuchen245/ViT-V-Net_for_3D_Image_Registration_Pytorch), [TransMorph](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration), [XMorpher](https://github.com/Solemoon/XMorpher), [Deformer](https://github.com/CJSOrange/DMR-Deformer), [Im2grid](https://github.com/anonymous2024slnet/SLNet/blob/main/models/Im2grid.py), [ModeT](https://github.com/anonymous2024slnet/SLNet/blob/main/models/ModeT.py), and [TransMatch](https://github.com/tzayuan/TransMatch_TMI).

# Question

For any questions, please open an issue or email changyuan@stu.scu.edu.cn.
