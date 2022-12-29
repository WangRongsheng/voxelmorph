# 1、voxelmorph: 基于学习的图像配准

**voxelmorph**是一个通用的库，基于学习的用于对齐/配准的工具，以及更普遍的变形建模。

# 2、教程

访问 [VoxelMorph tutorial](http://tutorial.voxelmorph.net/) 来了解VoxelMorph和基于学习的配准。 这是一个 [additional small tutorial](https://colab.research.google.com/drive/1V0CutSIfmtgDJg1XIkEnGteJuw0u7qT-#scrollTo=h1KXYz-Nauwn) 关于将注释与图像一起扭曲的问题, 并且另外一个是 [template (atlas) construction](https://colab.research.google.com/drive/1SkQbrWTQHpQFrG4J2WoBgGZC9yAzUas2?usp=sharing) with VoxelMorph.


# 3、说明

要使用VoxelMorph库，可以克隆这个仓库并安装`setup.py`中列出的要求，或者直接用pip安装：

```
pip install voxelmorph
```

## 3.1、预训练模型

查看可用的预训练模型列表 [here](data/readme.md#models).

## 3.2、训练

如果你想训练你自己的模型，你可能需要为你自己的数据集和数据格式定制`voxelmorph/generators.py`中的一些数据加载代码。然而，假设你在训练数据集中提供了一个文件名列表，就有可能运行许多开箱即用的示例脚本。训练数据可以是NIfTI、MGZ或npz（numpy）格式，假定你的数据列表中的每个npz文件都有一个`vol`参数，指向要注册的图像数据，还有一个可选的`seg`变量，指向相应的离散分割（用于半监督式学习）。它还假定所有训练图像数据的形状是一致的，当然，如果需要的话，可以在定制的生成器中处理。

对于一个给定的图像列表文件`/images/list.txt`和输出目录`/models/output`，下面的脚本将训练一个图像到图像的注册网络（默认在MICCAI 2018中描述），具有无监督的损失。模型权重将被保存到由`--model-dir`标志指定的路径。

```
./scripts/tf/train.py --img-list /images/list.txt --model-dir /models/output --gpu 0
```

`--img-prefix`和`--img-suffix`标志可以用来为图像列表中指定的每个路径提供一个一致的前缀或后缀。通过提供一个图集文件，如`--atlas atlas.npz`，可以实现图像到图集的注册。如果你想使用原始的密集CVPR网络进行训练（没有衍射），使用`--int-steps 0`标志来指定没有流量集成步骤。使用`--help`标志来检查所有的命令行选项，这些选项可以用来微调网络结构和训练。


## 3.3、配准

如果你只是想注册两个图像，你可以使用`register.py`脚本和所需的模型文件。例如，如果我们有一个模型`model.h5`被训练来注册一个主体（移动）和一个图集（固定），我们可以运行：

```
./scripts/tf/register.py --moving moving.nii.gz --fixed atlas.nii.gz --moved warped.nii.gz --model model.h5 --gpu 0
```

这将把移动后的图像保存为`warped.nii.gz`。要同时保存预测的变形场，请使用`--save-warp`标志。npz或nifty文件都可以作为这个脚本的输入/输出。


## 3.4、测试（测量Dice分数）。

为了通过计算图集分割和扭曲的测试扫描分割之间的骰子重叠来测试模型的质量，运行。

```
./scripts/tf/test.py --model model.h5 --atlas atlas.npz --scans scan01.npz scan02.npz scan03.npz --labels labels.npz
```

就像训练数据一样，图集和测试npz文件包括`vol`和`seg`参数，`labels.npz`文件包含一个相应的解剖标签列表，以包括在计算的骰子分数中。


## 3.5、参数选择


### 3.5.1、CVPR版本

对于CC损失函数，我们发现reg参数为1的效果最好。对于MSE损失函数，我们发现0.01是最好的。


### 3.5.2、MICCAI版本

对于我们的数据，我们发现`image_sigma=0.01`和`prior_lambda=25`效果最好。

在最初的MICCAI代码中，这些参数是在速度场的缩放之后应用的。在最新的代码中，这一点已经被 "修复"，不同的默认参数反映了这种变化。我们建议运行更新后的代码。然而，如果你想运行非常原始的MICCAI2018模式，请使用`xy`索引和`use_miccai_int`网络选项，使用MICCAI2018参数。


## 3.6、空间变换和整合

- The spatial transform code, found at `voxelmorph.layers.SpatialTransformer`, accepts N-dimensional affine and dense transforms, including linear and nearest neighbor interpolation options. Note that original development of VoxelMorph used `xy` indexing, whereas we are now emphasizing `ij` indexing.

- For the MICCAI2018 version, we integrate the velocity field using `voxelmorph.layers.VecInt`. By default we integrate using scaling and squaring, which we found efficient.


# VoxelMorph Papers

If you use voxelmorph or some part of the code, please cite (see [bibtex](citations.bib)):

  * HyperMorph, avoiding the need to tune registration hyperparameters:   

    **HyperMorph: Amortized Hyperparameter Learning for Image Registration.**  
    Andrew Hoopes, [Malte Hoffmann](https://nmr.mgh.harvard.edu/malte), Bruce Fischl, [John Guttag](https://people.csail.mit.edu/guttag/), [Adrian V. Dalca](http://adalca.mit.edu)  
    IPMI: Information Processing in Medical Imaging. 2021. [eprint arxiv:2101.01035](https://arxiv.org/abs/2101.01035)


  * [SynthMorph](https://synthmorph.voxelmorph.net), avoiding the need to have data at training (!):  

    **SynthMorph: learning contrast-invariant registration without acquired images.**  
    [Malte Hoffmann](https://nmr.mgh.harvard.edu/malte), Benjamin Billot, [Juan Eugenio Iglesias](https://scholar.harvard.edu/iglesias), Bruce Fischl, [Adrian V. Dalca](http://adalca.mit.edu)  
    IEEE TMI: Transactions on Medical Imaging. 2022. [eprint arXiv:2004.10282](https://arxiv.org/abs/2004.10282)

  * For the atlas formation model:  
  
    **Learning Conditional Deformable Templates with Convolutional Networks**  
  [Adrian V. Dalca](http://adalca.mit.edu), [Marianne Rakic](https://mariannerakic.github.io/), [John Guttag](https://people.csail.mit.edu/guttag/), [Mert R. Sabuncu](http://sabuncu.engineering.cornell.edu/)  
  NeurIPS 2019. [eprint arXiv:1908.02738](https://arxiv.org/abs/1908.02738)

  * For the diffeomorphic or probabilistic model:

    **Unsupervised Learning of Probabilistic Diffeomorphic Registration for Images and Surfaces**  
[Adrian V. Dalca](http://adalca.mit.edu), [Guha Balakrishnan](http://people.csail.mit.edu/balakg/), [John Guttag](https://people.csail.mit.edu/guttag/), [Mert R. Sabuncu](http://sabuncu.engineering.cornell.edu/)  
MedIA: Medial Image Analysis. 2019. [eprint arXiv:1903.03545](https://arxiv.org/abs/1903.03545) 

    **Unsupervised Learning for Fast Probabilistic Diffeomorphic Registration**  
[Adrian V. Dalca](http://adalca.mit.edu), [Guha Balakrishnan](http://people.csail.mit.edu/balakg/), [John Guttag](https://people.csail.mit.edu/guttag/), [Mert R. Sabuncu](http://sabuncu.engineering.cornell.edu/)  
MICCAI 2018. [eprint arXiv:1805.04605](https://arxiv.org/abs/1805.04605)

  * For the original CNN model, MSE, CC, or segmentation-based losses:

    **VoxelMorph: A Learning Framework for Deformable Medical Image Registration**  
[Guha Balakrishnan](http://people.csail.mit.edu/balakg/), [Amy Zhao](http://people.csail.mit.edu/xamyzhao/), [Mert R. Sabuncu](http://sabuncu.engineering.cornell.edu/), [John Guttag](https://people.csail.mit.edu/guttag/), [Adrian V. Dalca](http://adalca.mit.edu)  
IEEE TMI: Transactions on Medical Imaging. 2019. 
[eprint arXiv:1809.05231](https://arxiv.org/abs/1809.05231)

    **An Unsupervised Learning Model for Deformable Medical Image Registration**  
[Guha Balakrishnan](http://people.csail.mit.edu/balakg/), [Amy Zhao](http://people.csail.mit.edu/xamyzhao/), [Mert R. Sabuncu](http://sabuncu.engineering.cornell.edu/), [John Guttag](https://people.csail.mit.edu/guttag/), [Adrian V. Dalca](http://adalca.mit.edu)  
CVPR 2018. [eprint arXiv:1802.02604](https://arxiv.org/abs/1802.02604)


# Notes:
- **keywords**: machine learning, convolutional neural networks, alignment, mapping, registration  
- The `master` branch is still in testing as we roll out a major refactoring of the library.     
- If you'd like to run code from VoxelMorph publications, please use the `legacy` branch.  
- **data in papers**: 
In our initial papers, we used publicly available data, but unfortunately we cannot redistribute it (due to the constraints of those datasets). We do a certain amount of pre-processing for the brain images we work with, to eliminate sources of variation and be able to compare algorithms on a level playing field. In particular, we perform FreeSurfer `recon-all` steps up to skull stripping and affine normalization to Talairach space, and crop the images via `((48, 48), (31, 33), (3, 29))`. 

We encourage users to download and process their own data. See [a list of medical imaging datasets here](https://github.com/adalca/medical-datasets). Note that you likely do not need to perform all of the preprocessing steps, and indeed VoxelMorph has been used in other work with other data.


# Creation of Deformable Templates

To experiment with this method, please use `train_template.py` for unconditional templates and `train_cond_template.py` for conditional templates, which use the same conventions as voxelmorph (please note that these files are less polished than the rest of the voxelmorph library).

We've also provided an unconditional atlas in `data/generated_uncond_atlas.npz.npy`. 

Models in h5 format weights are provided for [unconditional atlas here](http://people.csail.mit.edu/adalca/voxelmorph/atlas_creation_uncond_NCC_1500.h5), and [conditional atlas here](http://people.csail.mit.edu/adalca/voxelmorph/atlas_creation_cond_NCC_1022.h5).

**Explore the atlases [interactively here](http://voxelmorph.mit.edu/atlas_creation/)** with tipiX!


# SynthMorph

SynthMorph is a strategy for learning registration without acquired imaging data, producing powerful networks agnostic to contrast induced by MRI ([eprint arXiv:2004.10282](https://arxiv.org/abs/2004.10282)). For a video and a demo showcasing the steps of generating random label maps from noise distributions and using these to train a network, visit [synthmorph.voxelmorph.net](https://synthmorph.voxelmorph.net).

We provide model files for a ["shapes" variant](https://surfer.nmr.mgh.harvard.edu/ftp/data/voxelmorph/synthmorph/shapes-dice-vel-3-res-8-16-32-256f.h5) of SynthMorph, that we train using images synthesized from random shapes only, and a ["brains" variant](https://surfer.nmr.mgh.harvard.edu/ftp/data/voxelmorph/synthmorph/brains-dice-vel-0.5-res-16-256f.h5), that we train using images synthesized from brain label maps. We train the brains variant by optimizing a loss term that measures volume overlap of a [selection of brain labels](https://surfer.nmr.mgh.harvard.edu/ftp/data/voxelmorph/synthmorph/fs-labels.npy). For registration with either model, please use the `register.py` script with the respective model weights.

Accurate registration requires the input images to be min-max normalized, such that voxel intensities range from 0 to 1, and to be resampled in the affine space of a [reference image](https://surfer.nmr.mgh.harvard.edu/ftp/data/voxelmorph/synthmorph/ref.nii.gz). The affine registration can be performed with a variety of packages, and we choose FreeSurfer. First, we skull-strip the images with [SAMSEG](https://surfer.nmr.mgh.harvard.edu/fswiki/Samseg), keeping brain labels only. Second, we run [mri_robust_register](https://surfer.nmr.mgh.harvard.edu/fswiki/mri_robust_register):

```
mri_robust_register --mov in.nii.gz --dst out.nii.gz --lta transform.lta --satit --iscale
mri_robust_register --mov in.nii.gz --dst out.nii.gz --lta transform.lta --satit --iscale --ixform transform.lta --affine
```

where we replace `--satit --iscale` with `--cost NMI` for registration across MRI contrasts.


# Contact:
For any problems or questions please [open an issue](https://github.com/voxelmorph/voxelmorph/issues/new?labels=voxelmorph) for code problems/questions or [start a discussion](https://github.com/voxelmorph/voxelmorph/discussions) for general registration/voxelmorph question/discussion.  
