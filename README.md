# AGMM-SASS
Code for CVPR2023 paper, [**"Sparsely Annotated Semantic Segmentation with Adaptive Gaussian Mixtures"**](https://openaccess.thecvf.com/content/CVPR2023/papers/Wu_Sparsely_Annotated_Semantic_Segmentation_With_Adaptive_Gaussian_Mixtures_CVPR_2023_paper.pdf)

Authors: Linshan Wu, <a href="https://scholar.google.com/citations?user=nZizkQ0AAAAJ&hl">Zhun Zhong</a>, <a href="https://scholar.google.com/citations?hl=en&user=Gfa4nasAAAAJ">Leyuan Fang</a>, <a href="https://scholar.google.com/citations?hl=zh-CN&user=bHSKDuYAAAAJ">Xingxin He</a>, Qiang Liu, <a href="https://scholar.google.com/citations?hl=zh-CN&user=73trMQkAAAAJ">Jiayi Ma</a>, and <a href="https://scholar.google.com/citations?hl=en&user=Z_t5DjwAAAAJ">Hao Chen</a>

## Abstract
Sparsely annotated semantic segmentation (SASS) aims to learn a segmentation model by images with sparse labels (i.e., points or scribbles). Existing methods mainly focus on introducing low-level affinity or generating pseudo labels to strengthen supervision, while largely ignoring the inherent relation between labeled and unlabeled pixels. In this paper, we observe that pixels that are close to each other in the feature space are more likely to share the same
class. Inspired by this, we propose a novel SASS framework, which is equipped with an Adaptive Gaussian Mixture Model (AGMM). Our AGMM can effectively endow reliable supervision for unlabeled pixels based on the distributions of labeled and unlabeled pixels. Specifically, we first build Gaussian mixtures using labeled pixels and their relatively similar unlabeled pixels, where the labeled pixels act as centroids, for modeling the feature distribution of each class. Then, we leverage the reliable information from labeled pixels and adaptively generated GMM predictions to supervise the training of unlabeled pixels, achieving online, dynamic, and robust self-supervision. In addition, by capturing category-wise Gaussian mixtures, AGMM encourages the model to learn discriminative class decision boundaries in an end-to-end contrastive learning manner. Experimental results conducted on the PASCAL VOC 2012 and Cityscapes datasets demonstrate that our AGMM can establish new state-of-the-art SASS performance.

## Getting Started
### Prepare Dataset
- Pascal: [JPEGImages](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar) | [SegmentationClass](https://drive.google.com/file/d/1ikrDlsai5QSf2GiSUR3f8PZUzyTubcuF/view?usp=sharing)
- Cityscapes: [leftImg8bit](https://www.cityscapes-dataset.com/file-handling/?packageID=3) | [gtFine](https://drive.google.com/file/d/1E_27g9tuHm6baBqcA7jct_jqcGA89QPm/view?usp=sharing)

### Download weak labels
- Pascal: [points](https://pan.baidu.com/s/1CqyrS1XGcZh42jfF8FqUJg?pwd=1111) | [scribbles](https://pan.baidu.com/s/18lASrYxf4kHEtZ_Rn4FuAA?pwd=1111)
- Cityscapes: [points](https://pan.baidu.com/s/1527AVSxxXDw0jdNdi_vKag?pwd=1111)
- Code:1111
- Note: points and scribbles for pascal are downloaded from [TEL](https://github.com/megvii-research/TreeEnergyLoss)
```
├── [Your Pascal Path]
    ├── JPEGImages
    ├── point
    ├── scribble
    └── SegmentationClass
    
├── [Your Cityscapes Path]
    ├── leftImg8bit
    ├── 20clicks
    ├── 50clicks
    ├── 100clicks
    └── gtFine
```
### Pretrained Backbone:
[ResNet-50](https://drive.google.com/file/d/1mqUrqFvTQ0k5QEotk4oiOFyP6B9dVZXS/view?usp=sharing) | [ResNet-101](https://drive.google.com/file/d/1Rx0legsMolCWENpfvE2jUScT3ogalMO8/view?usp=sharing)
```
├── ./pretrained
    ├── resnet50.pth
    └── resnet101.pth
```

### Usage
train
```bash 
sh scripts/train_voc.sh <num_gpu> <port>
sh scripts/train_city.sh <num_gpu> <port>
```
eval
```bash 
python eval.py
```

## Citation ✏️ 📄
If you find this repo useful for your research, please consider citing the paper as follows:
```
@article{wu2025modeling,
  title={Modeling the label distributions for weakly-supervised semantic segmentation},
  author={Wu, Linshan and Zhong, Zhun and Ma, Jiayi and Wei, Yunchao and Chen, Hao and Fang, Leyuan and Li, Shutao},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2025},
  publisher={IEEE}
}
@inproceedings{AGMM,
  title={Sparsely Annotated Semantic Segmentation with Adaptive Gaussian Mixtures},
  author={Wu, Linshan and Zhong, Zhun and Fang, Leyuan and He, Xingxin and Liu, Qiang and Ma, Jiayi and Chen, Hao},
  booktitle={IEEE Conf. Comput. Vis. Pattern Recog.},
  month={June},
  year={2023},
  pages={15454-15464}
  }
@ARTICLE{Wu_DBFNet,
  author={Wu, Linshan and Fang, Leyuan and Yue, Jun and Zhang, Bob and Ghamisi, Pedram and He, Min},
  journal={IEEE Transactions on Image Processing}, 
  title={Deep Bilateral Filtering Network for Point-Supervised Semantic Segmentation in Remote Sensing Images}, 
  year={2022},
  volume={31},
  number={},
  pages={7419-7434},
  doi={10.1109/TIP.2022.3222904}}
```

