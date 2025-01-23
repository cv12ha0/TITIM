# TITIM

This repository is the source code for USENIX Security '25 paper "Revisiting Training-Inference Trigger Intensity in Backdoor Attacks".



## Environment

```shell
pip install -r requirements.txt
```

or run commands in `scripts/env.sh`.



## Usage

#### Generate datasets

Obtain clean datasets: 

 ```shell
 sh scripts/get_clean_datasets.sh
 ```

Then generate poisoned datasets by BadNets:

```shell
sh scripts/inject.sh
```

The poisoned datasets are saved to `data/<dataset>/<subset>/`.



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

This scripts may need to be modified for other attacks / datasets.


#### Defenses

Test backdoor defenses on poisoned datasets / models:

```shell
# sh scripts/defenses/defense_<defense>.sh <device> <dataset> <model> <...>
sh scripts/defense_abl.sh 0 cifar10 resnet18
```

The arguments may vary between different defenses, please refer to the corresponding scripts. 


#### Intensity Mixing

To train backdoored models with two intensities by BadNets(Square)

```shell
sh scripts/mixtest/inject_mix.sh  # generate poisoned datasets
sh scripts/mixtest/train_mix.sh  # train backdoored models
sh scripts/mixtest/crosstest_mix.sh  # inference with varying intensities
```

The logs are saved to `logs/cross_badnets_mixmr0.1_0.05_resnet18.tsv` by default.


## Cite thie work

To be continued $\rightarrow$