# Rethinking on Multi-Stage Networks for Human Pose Estimation

This repo is also linked to [github][9].

## Introduction
This is a pytorch realization of MSPN proposed in [ Rethinking on Multi-Stage Networks for Human Pose Estimation ][1]. In this work, we design an effective network MSPN to fulfill human pose estimation task.

Existing pose estimation approaches fall into two categories: single-stage and multi-stage methods. While multistage methods are seemingly more suited for the task, their performance in current practice is not as good as singlestage methods. This work studies this issue. We argue that the current multi-stage methods’ unsatisfactory performance comes from the insufficiency in various design choices. We propose several improvements, including the single-stage module design, cross stage feature aggregation, and coarse-tofine supervision. 

![Overview of MSPN.](/figures/MSPN.png)

The resulting method establishes the new state-of-the-art on both MS COCO and MPII Human Pose dataset, justifying the effectiveness of a multi-stage architecture.

## Results

### Results on COCO val dataset
| Model | Input Size | AP | AP<sup>50</sup> | AP<sup>75</sup> | AP<sup>M</sup> | AP<sup>L</sup> | AR | AR<sup>50</sup> | AR<sup>75</sup> | AR<sup>M</sup> | AR<sup>L</sup> |
| :-----------------: | :-----------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: |
| 1-stg MSPN | 256x192 | 71.5 | 90.1 | 78.4 | 67.4 | 77.5 | 77.0 | 93.2 | 83.1 | 72.6 | 83.1 |
| 2-stg MSPN | 256x192 | 74.5 | 91.2 | 81.2 | 70.5 | 80.4 | 79.7 | 94.2 | 85.6 | 75.4 | 85.7 |
| 3-stg MSPN | 256x192 | 75.2 | 91.5 | 82.2 | 71.1 | 81.1 | 80.3 | 94.3 | 86.4 | 76.0 | 86.4 |
| 4-stg MSPN | 256x192 | 75.9 | 91.8 | 82.9 | 72.0 | 81.6 | 81.1 | 94.9 | 87.1 | 76.9 | 87.0 |
| 4-stg MSPN<sup>\*</sup> | 384x288 | 78.8 | 93.1 | 85.6 | 74.9 | 84.7 | 83.8 | 95.9 | 89.5 | 79.7 | 89.6 |
| 4-stg MSPN<sup>\+\*</sup> | 384x288 | 79.8 | 93.4 | 86.1 | 75.9 | 85.6 | 84.3 | 96.0 | 89.7 | 80.2 | 90.0 |

### Results on COCO test-dev dataset
| Model | Input Size | AP | AP<sup>50</sup> | AP<sup>75</sup> | AP<sup>M</sup> | AP<sup>L</sup> | AR | AR<sup>50</sup> | AR<sup>75</sup> | AR<sup>M</sup> | AR<sup>L</sup> |
| :-----------------: | :-----------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: |
| 4-stg MSPN | 384x288 | 76.1 | 93.4 | 83.8 | 72.3 | 81.5 | 81.6 | 96.3 | 88.1 | 77.5 | 87.1 |
| 4-stg MSPN<sup>\*</sup> | 384x288 | 77.1 | 93.8 | 84.6 | 73.4 | 82.3 | 82.3 | 96.5 | 88.9 | 78.4 | 87.7 |
| 4-stg MSPN<sup>\+\*</sup> | 384x288 | 78.1 | 94.1 | 85.9 | 74.5 | 83.3 | 83.1 | 96.7 | 89.8 | 79.3 | 88.2 |

### Results on MPII dataset
| Model | Split | Input Size | Head | Shoulder | Elbow | Wrist | Hip | Knee | Ankle | Mean |
| :-----------------: | :------------------: | :-----------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: |
| 4-stg MSPN | val | 256x256 | 96.8 | 96.5 | 92.0 | 87.0 | 89.9 | 88.0 | 84.0 | 91.1 |
| 4-stg MSPN<sup>\#</sup> | test | 256x256 | 98.4 | 97.1 | 93.2 | 89.2 | 92.0 | 90.1 | 85.5 | 92.6 |

#### Note
* \* means using external data.
* \+ means using model ensemble.
* \# means using multi-shift test.

## Repo Structure
This repo is organized as following:
```
$MSPN_HOME
|-- cvpack
|
|-- dataset
|   |-- COCO
|   |   |-- det_json
|   |   |-- gt_json
|   |   |-- images
|   |       |-- train2014
|   |       |-- val2014
|   |
|   |-- MPII
|       |-- det_json
|       |-- gt_json
|       |-- images
|   
|-- lib
|   |-- models
|   |-- utils
|
|-- exps
|   |-- exp1
|   |-- exp2
|   |-- ...
|
|-- model_logs
|
|-- README.md
|-- requirements.txt
```

## Quick Start

### Installation

1. Install Pytorch referring to [Pytorch website][2].

2. Clone this repo, and config **MSPN_HOME** in **/etc/profile** or **~/.bashrc**, e.g.
 ```
 export MSPN_HOME='/path/of/your/cloned/repo'
 export PYTHONPATH=$PYTHONPATH:$MSPN_HOME
 ```

3. Install requirements:
 ```
 pip3 install -r requirements.txt
 ```

4. Install COCOAPI referring to [cocoapi website][3], or:
 ```
 git clone https://github.com/cocodataset/cocoapi.git $MSPN_HOME/lib/COCOAPI
 cd $MSPN_HOME/lib/COCOAPI/PythonAPI
 make install
 ```
 
### Dataset

#### COCO

1. Download images from [COCO website][4], and put train2014/val2014 splits into **$MSPN_HOME/dataset/COCO/images/** respectively.

2. Download ground truth from [Google Drive][6], and put it into **$MSPN_HOME/dataset/COCO/gt_json/**.

3. Download detection result from [Google Drive][6], and put it into **$MSPN_HOME/dataset/COCO/det_json/**.

#### MPII

1. Download images from [MPII website][5], and put images into **$MSPN_HOME/dataset/MPII/images/**.

2. Download ground truth from [Google Drive][6], and put it into **$MSPN_HOME/dataset/MPII/gt_json/**.

3. Download detection result from [Google Drive][6], and put it into **$MSPN_HOME/dataset/MPII/det_json/**.

### Model
Download ImageNet pretained ResNet-50 model from [Google Drive][6], and put it into **$MSPN_HOME/lib/models/**. For your convenience, We also provide a well-trained 2-stage MSPN model for COCO.

### Log
Create a directory to save logs and models:
```
mkdir $MSPN_HOME/model_logs
```

### Train
Go to specified experiment repository, e.g.
```
cd $MSPN_HOME/exps/mspn.2xstg.coco
```
and run:
```
python config.py -log
python -m torch.distributed.launch --nproc_per_node=gpu_num train.py
```
the ***gpu_num*** is the number of gpus.

### Test
```
python -m torch.distributed.launch --nproc_per_node=gpu_num test.py -i iter_num
```
the ***gpu_num*** is the number of gpus, and ***iter_num*** is the iteration number you want to test.

## Citation
Please considering citing our projects in your publications if they help your research.
```
@article{li2019rethinking,
  title={Rethinking on Multi-Stage Networks for Human Pose Estimation},
  author={Li, Wenbo and Wang, Zhicheng and Yin, Binyi and Peng, Qixiang and Du, Yuming and Xiao, Tianzi and Yu, Gang and Lu, Hongtao and Wei, Yichen and Sun, Jian},
  journal={arXiv preprint arXiv:1901.00148},
  year={2019}
}

@inproceedings{chen2018cascaded,
  title={Cascaded pyramid network for multi-person pose estimation},
  author={Chen, Yilun and Wang, Zhicheng and Peng, Yuxiang and Zhang, Zhiqiang and Yu, Gang and Sun, Jian},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={7103--7112},
  year={2018}
}
```
And the [code][7] of [Cascaded Pyramid Network][8] is also available. 

## Contact
You can contact us by email published in our [paper][1] or fenglinglwb@gmail.com.

[1]: https://arxiv.org/abs/1901.00148
[2]: https://pytorch.org/
[3]: https://github.com/cocodataset/cocoapi
[4]: http://cocodataset.org/#download
[5]: http://human-pose.mpi-inf.mpg.de/
[6]: https://drive.google.com/open?id=1MW27OY_4YetEZ4JiD4PltFGL_1-caECy
[7]: https://github.com/chenyilun95/tf-cpn
[8]: https://arxiv.org/abs/1711.07319
[9]: https://github.com/fenglinglwb/MSPN

