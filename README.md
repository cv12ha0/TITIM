# TITIM

This repository is the source code for "Revisiting Training-Inference Trigger Intensity in Backdoor Attacks".



## Environment

See `requirements.txt` or `scripts/env.sh`.



## Usage

#### Generate datasets

First download clean datasets from [here]() and put it in `data/<dataset>/clean/`, then generate poisoned datasets by BadNets:

```shell
sh scripts/inject.sh
```

The poisoned datasets will be saved to `data/<dataset>/<subset>/`.



#### Train models

We provide shell scripts for training models with different trigger intensities: 

```shell
# sh scripts/train/train_<attack>.sh <device> <dataset> <model> <poison>
sh scripts/train/train_badnets.sh 0 cifar10 resnet18 0.01
```

Or train a single model:

```shell
python train.py --gpu 0 --model resnet18 --dataset cifar10 \
--subset clean --epochs 100 --bs 128 --lr 1e-2 --optimizer adam --disable_prog
```



#### Inference

Test backdoored models on different poisoned datasets:

```shell
# sh scripts/cross_tests.sh <device> <dataset> <model>
sh scripts/cross_tests.sh 0 cifar10 resnet18
```

The results are saved to `./logs/cross_<attack>.tsv`



#### Visualization

Draw heatmaps:

```shell
python utils/scripts/tsv_reader.py
```











## Cite thie work

To be continued $\rightarrow$