# The commands I used to create my environment

# dependency of Wand
sudo apt-get install libmagickwand-dev

# the conda environment
conda create -n titim python=3.11
conda activate titim

conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
conda install tqdm
pip install wand
conda install scikit-learn
conda install matplotlib
pip install seaborn
pip install h5py
pip install scikit-image
pip install tabulate
pip install opencv-python
pip install lpips
pip install umap-learn
pip install opentsne
pip install grad-cam
pip install gdown

