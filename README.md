# CGNet

# Environment

cuda 11.4

python 3.6

pytorch 1.7.1

torchvision 0.8.2

simpleitk 2.1.1

# Dataset

[OASIS](https://sites.wustl.edu/oasisbrains/]) dataset.

[IXI](https://brain-development.org/ixi-dataset/) dataset.

[LPBA](https://resource.loni.usc.edu/resources/atlases-downloads/) dataset.

[Mindboggle](https://osf.io/nhtur/) dataset.

# Training

First, you need to download the original dataset, and then perform all the processing operations mentioned in the paper using [FreeSurfer](https://surfer.nmr.mgh.harvard.edu/). Finally, run 'python train_CGNet.py'.

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
