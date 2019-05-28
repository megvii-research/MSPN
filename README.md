# Rethinking on Multi-Stage Networks for Human Pose Estimation
----

## Introduction
This is a pytorch realization of MSPN proposed in [ Rethinking on Multi-Stage Networks for Human Pose Estimation ][1]. In this work, we design an effective network MSPN to fulfill human pose estimation task.

Existing pose estimation approaches fall into two categories: single-stage and multi-stage methods. While multistage methods are seemingly more suited for the task, their performance in current practice is not as good as singlestage methods. This work studies this issue. We argue that the current multi-stage methods’ unsatisfactory performance comes from the insufficiency in various design choices. We propose several improvements, including the single-stage module design, cross stage feature aggregation, and coarse-tofine supervision. 

![Overview of MSPN.](/figures/MSPN.png)

The resulting method establishes the new state-of-the-art on both MS COCO and MPII Human Pose dataset, justifying the effectiveness of a multi-stage architecture.

## Results

### COCO

#### COCO val
| Model | Dataset | Input Size | mAP |
| :-----------------: | :------------------: | :-----------: | :------: |
| 1-stg MSPN | COCO val | 256x192 | 71.5 |
| 2-stg MSPN | COCO val | 256x192 | 74.5 |
| 3-stg MSPN | COCO val | 256x192 | 75.2 |
| 4-stg MSPN | COCO val | 256x192 | 75.9 |
| 4-stg MSPN<sup>\*</sup> | COCO val | 384x288 | 79.0 |
| 4-stg MSPN<sup>\+\*</sup> | COCO val | 384x288 | 80.0 |

#### COCO test-dev
| Model | Dataset | Input Size | mAP |
| :-----------------: | :------------------: | :-----------: | :------: |
| 4-stg MSPN | COCO test-dev | 384x288 | 76.1 |
| 4-stg MSPN<sup>\*</sup> | COCO test-dev | 384x288 | 77.1 |
| 4-stg MSPN<sup>\+\*</sup> | COCO test-dev | 384x288 | 78.1 |

### MPII
| Model | Dataset | Input Size | PCKh@0.5 |
| :-----------------: | :------------------: | :-----------: | :------: |
| 4-stg MSPN | MPII val | 256x256 | 91.1 |
| 4-stg MSPN<sup>\#</sup> | MPII test | 256x256 | 92.6 |

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
Please considering citing this project in your publications if it helps your research.
```
@article{li2019rethinking,
  title={Rethinking on Multi-Stage Networks for Human Pose Estimation},
  author={Li, Wenbo and Wang, Zhicheng and Yin, Binyi and Peng, Qixiang and Du, Yuming and Xiao, Tianzi and Yu, Gang and Lu, Hongtao and Wei, Yichen and Sun, Jian},
  journal={arXiv preprint arXiv:1901.00148},
  year={2019}
}
```

[1]: https://arxiv.org/abs/1901.00148
[2]: https://pytorch.org/
[3]: https://github.com/cocodataset/cocoapi
[4]: http://cocodataset.org/#download
[5]: http://human-pose.mpi-inf.mpg.de/
[6]: https://drive.google.com/open?id=1MW27OY_4YetEZ4JiD4PltFGL_1-caECy


