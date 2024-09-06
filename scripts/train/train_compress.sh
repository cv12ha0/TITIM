gpu=${1:-0}
trigger=compress
dataset=${2:-cifar10}  # cifar10 gtsrb
model=${3:-resnet18}
epochs=100
ratio=${4:-0.1}
alg=${5:-jpgpil}  # jpgcv
echo $(date +%Y-%m-%d_%H:%M:%S)

# train
for quality in 0 10 20 30 40 50 60 70 80 90 100
do
    echo training:  ${model}_${dataset}_compress_${alg}_q${quality}_${ratio}_e${epochs}
    python train.py --gpu ${gpu} --model ${model} --dataset ${dataset} --subset compress_${alg}_q${quality}_${ratio} --epochs ${epochs} --bs 128 --lr 1e-2 --optimizer adam --log logs/compress.tsv --mode cover --split_val test --disable_prog
    printf '\n'
done
