
<div align="center">

# ShadowLane:Physically-Based Shadow Synthesis and Natural-to-Adversarial Attacks for Robust Lane Detection

</div>



Pytorch implementation of the paper "[ShadowLane:Physically-Based Shadow Synthesis and Natural-to-Adversarial Attacks for Robust Lane Detection]".

## Introduction
- This paper explores a new question: whether shadows cast by environmental factors in the real physical world affect lane detectors. From the perspective of adversarial attacks, we propose and define a new task: adversarial lane shadow attacks. 
- This work not only reveals the potential threat posed by natural shadows to lane detection systems, but also provides new insights and methods for evaluating the robustness of lane detectors in real-world environments.

## Installation

### Prerequisites
Only test on Ubuntu16.04:
- Python >= 3.8
- PyTorch >= 1.9.0
- CUDA (tested with cuda11.1)
- Other dependencies described in `requirements.txt`

### Install dependencies

```Shell
# Install pytorch firstly, the cudatoolkit version should be same in your system.
conda install pytorch torchvision cudatoolkit -c pytorch

# Install python packages
python setup.py build develop
```

### Data preparation

#### Tusimple
Download [Tusimple](https://github.com/TuSimple/tusimple-benchmark/issues/3). Then extract them to `data` directory.

#### CULane
Download [CULane](https://xingangpan.github.io/projects/CULane.html). Then extract them to `data` directory.

## Getting Started

### Attack
For Attack, run
```Shell
python main.py [configs/path_to_your_config] --attack --load_from [path_to_your_model] --gpus [gpu_num]
```
For example, run
```Shell
python main.py configs/clrnet/clr_resnet18_tusimple.py --attack --load_from work_dirs/clr/r18_tusimple-Original/20241215_114052_lr_1e-03_b_40/ckpt/69.pth --gpus 0
```

We will get the attack result in `/attack_img`.


## Results
![F1 vs. Latency for SOTA methods on the lane detection](.github/latency_f1score.png)

[assets]: https://github.com/turoad/CLRNet/releases

### CULane

|   Backbone  |  mF1 | F1@50  | F1@75 |
| :---  |  :---:   |   :---:    | :---:|
| [ResNet-18][assets]     | 55.23  |  79.58   | 62.21 |
| [ResNet-34][assets]     | 55.14  |  79.73   | 62.11 |
| [ResNet-101][assets]     | 55.55| 80.13   | 62.96 |
| [DLA-34][assets]     | 55.64|  80.47   | 62.78 |



### TuSimple
|   Backbone   |      F1   | Acc |      FDR     |      FNR   |
|    :---       |          ---:          |       ---:       |       ---:       |      ---:       |
| [ResNet-18][assets]     |    97.89    |   96.84  |    2.28  |  1.92      | 
| [ResNet-34][assets]       |   97.82              |    96.87          |   2.27          |    2.08      | 
| [ResNet-101][assets]      |   97.62|   96.83  |   2.37   |  2.38  |



### LLAMAS
|   Backbone    |  <center>  valid <br><center> &nbsp; mF1 &nbsp; &nbsp;  &nbsp;F1@50 &nbsp; F1@75     | <center>  test <br> F1@50 |
|  :---:  |    :---:    |        :---:|
| [ResNet-18][assets] |  <center> 70.83  &nbsp; &nbsp; 96.93 &nbsp; &nbsp; 85.23 | 96.00 |
| [DLA-34][assets]     |  <center> 71.57 &nbsp; &nbsp;  97.06  &nbsp; &nbsp; 85.43  |   96.12 |

“F1@50” refers to the official metric, i.e., F1 score when IoU threshold is 0.5 between the gt and prediction. "F1@75" is the F1 score when IoU threshold is 0.75.

## Citation

If our paper and code are beneficial to your work, please consider citing:
```
@InProceedings{Zheng_2022_CVPR,
    author    = {Zheng, Tu and Huang, Yifei and Liu, Yang and Tang, Wenjian and Yang, Zheng and Cai, Deng and He, Xiaofei},
    title     = {CLRNet: Cross Layer Refinement Network for Lane Detection},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {898-907}
}
```

## Acknowledgement
<!--ts-->
* [open-mmlab/mmdetection](https://github.com/open-mmlab/mmdetection)
* [pytorch/vision](https://github.com/pytorch/vision)
* [Turoad/lanedet](https://github.com/Turoad/lanedet)
* [ZJULearning/resa](https://github.com/ZJULearning/resa)
* [cfzd/Ultra-Fast-Lane-Detection](https://github.com/cfzd/Ultra-Fast-Lane-Detection)
* [lucastabelini/LaneATT](https://github.com/lucastabelini/LaneATT)
* [aliyun/conditional-lane-detection](https://github.com/aliyun/conditional-lane-detection)
<!--te-->
