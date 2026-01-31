
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
python main.py configs/clrnet/clr_resnet18_tusimple.py --attack --load_from work_dirs/clr/XXX/XXXX/ckpt/XX.pth --gpus 0
```

We will get the attack result in `/attack_img`.


## Results

### TuSimple
| TuSimple | Acc.(I) | F1(I) | Acc.(II) | F1(II) | Acc.(III) | F1(III) | MIR  |
|:--------|--------:|------:|---------:|-------:|----------:|--------:|-----:|
| Original | 96.84 | 97.89 | 95.16 | 88.44 | 88.69 | 48.91 | 60.36 |
| Ours (Line Shadow attack) | 89.89 | 91.03 | 85.38 | 71.87 | 72.62 | 22.83 | 68.65 |
| Ours (Tree Shadow attack) | 94.95 | 95.84 | 92.82 | 84.89 | 85.70 | 42.56 | 62.63 |


