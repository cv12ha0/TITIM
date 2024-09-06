gpu=${1:-0}
trigger=wanet
dataset=${2:-cifar10}  # cifar10 gtsrb
model=${3:-resnet18}
epochs=100
ratio=${4:-0.05}
k=${5:-8}
cr=${6:-0.0}
gs=1.0
echo $(date +%Y-%m-%d_%H:%M:%S)


for s in 0.2 0.4 0.6 0.8 1.0 1.2 1.4 1.6 1.8 2.0
do
    echo training:  ${model}_${dataset}_wanet_cr${cr}_s${s}_k${k}_gs${gs}_${ratio}_e${epochs}
    python train_wanet.py --gpu ${gpu} --model ${model} --dataset ${dataset} --subset clean --target=0 --inject_ratio=${ratio} --cross_ratio=${cr} --s=${s} --k=${k} --grid_rescale=${gs} --fixed --epochs ${epochs} --bs 128 --lr 1e-2 --optimizer adam --log logs/wanet.tsv --mode cover --split_val test --disable_prog 
    printf '\n'
done