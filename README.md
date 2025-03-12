# TITIM

This repository is the source code for USENIX Security '25 paper "Revisiting Training-Inference Trigger Intensity in Backdoor Attacks".

If you encounter any errors, please check the Troubleshooting section of this doc, or feel free to open an issue.



## Dependency

#### Software Environment

```shell
# dependency of wand
sudo apt-get install libmagickwand-dev
# conda/venv environment
pip install -r requirements.txt
```

or run commands in `scripts/env.sh`. Please refer to the [Artifact Appendix](./artifact_appendix.pdf) for more details. 

#### Hardware Requirement

Experiment with ResNet18 on CIFAR-10 require at least **8GB of RAM** and a **2GB NVIDIA GPU**. For other models and datasets, we recommend a machine with Linux operating system with 128GB of RAM and a 24GB NVIDIA GPU. 

In our experiment setup, we use a server running Ubuntu 22.04.4 with 2× Intel Xeon Gold 6226R CPUs, 256GB of RAM, and 4× NVIDIA GeForce RTX 3090 GPUs. 



## File Structure

```
TITIM/
├── data/               The clean/poisoned datasets & info
│   ├── cifar10/
│   └── ...
├── heatmaps/           Plotted heatmaps
├── logs/               Inference logs
├── scripts/            Scripts for poisoning/training/inference/analysis
├── res/                Results of training
└── utils/
    ├── assets/         Fixed image patterns of some attacks/defenses
    ├── backdoor/       Implementations of attacks/defenses
    │   ├── attack/
    │   └── defense/
    ├── datasets/       Dataset I/O
    ├── misc/           Auxiliary methods (training, evaluation, etc.)
    ├── models/         Commonly used model architectures
    └── scripts/        Auxiliary python scripts
```

This repository consists of implemented backdoor attack and defense algorithms, along with corresponding scripts:

1. **Backdoor Attack and Defense Algorithms** (`utils/backdoor/`): This module contains the implementations of the algorithms discussed in the paper. 
2. **Poisoning/Training/Inference Scripts** (`scripts/`): This folder includes shell scripts that facilitate batch poisoning, training, inference, and dataset downloading.
3. **Supporting Code** (`utils/...`): In addition to the backdoor attack and defense algorithms, the `utils` module also includes implementations for dataset I/O (`utils/datasets`), commonly used models architectures (`utils/models`), and various auxiliary methods (`utils/misc`). 

Please refer to the [Zenodo](https://zenodo.org/records/14729436) repository for more details.



## Usage

#### Generate datasets

Obtain clean datasets (or download from [Zenodo](https://zenodo.org/records/14729436)): 

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



## Troubleshooting

Here are some workarounds for potential issues. Please feel free to open an issue if you still encounter any problems.

1. **ImageMagick**

   If you get errors about ImageMagick, please try rebooting, or run the following  commands:

   ```shell
   # install both ImageMagick and libmagickwand 
   sudo apt install imagemagick libmagickwand-dev
   
   # install imagemagick from Conda-forge channel
   conda install -c conda-forge imagemagick
   ```

2. **Shell and Bash**

   If the scripts don't function properly in the CLI, please try using `bash` instead of `sh` (i.e., `bash <scripit>.sh`)

3. **Downloading CelebA**

   If you get errors with Gdown while downloading CelebA dataset using `get_clean_datasets.sh`, you can manually download the datasets from the following links and place them in the `data/offic/celeba/` directory:

   ```apache
   https://drive.google.com/uc?id=0B7EVK8r0v71pZjFTYXZWM3FlRnM 		img_align_celeba.zip
   https://drive.google.com/uc?id=0B7EVK8r0v71pblRyaVFSWGxPY0U 		list_attr_celeba.txt
   https://drive.google.com/uc?id=1_ee_0u7vcNLOfNLegJRHmolfH5ICW-XS 	identity_CelebA.txt
   https://drive.google.com/uc?id=0B7EVK8r0v71pbThiMVRxWXZ4dU0 		list_bbox_celeba.txt
   https://drive.google.com/uc?id=0B7EVK8r0v71pd0FJY3Blby1HUTQ 		list_landmarks_align_celeba.txt
   https://drive.google.com/uc?id=0B7EVK8r0v71pY0NSMzRuSXJEVkk 		list_eval_partition.txt
   ```

   Then run `python utils/scripts/dataset_get_clean.py --dataset celeba8` again, and the clean dataset will be processed automatically.

   You can also download the datasets from our [Zenodo](https://zenodo.org/records/14729436) repository and extract them in the base directory of this repoitory.

4. **File End Issues**

   If you inspect the code on Windows and sync the project via an IDE (e.g., PyCharm) to a Linux server, you may encounter errors due to line endings. You can either adjust the IDE settings or run `sed -i 's/\r$//' <script>.sh` to manually correct the line endings.




## Cite the work

Please cite our paper if you find this repository useful, thanks! 

(The BibTeX entry may be updated after publication.)

```
@inproceedings{lin2025titim,
    title = {Revisiting Training-Inference Trigger Intensity in Backdoor Attacks},
    author = {Lin, Chenhao and Zhao, Chenyang and Wang, Shiwei and Wang, Longtian and Shen, Chao and Zhao, Zhengyu},
    booktitle = {34th USENIX Security Symposium (USENIX Security 25)},
    year = {2025}
}
```
