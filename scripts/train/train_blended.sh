gpu=${1:-0}
trigger=blended
dataset=${2:-cifar10}  # cifar10
model=${3:-resnet18}  # preactresnet18
epochs=100
pattern=${4:-noise}
log=${5:-logs/blend.tsv}
echo $(date +%Y-%m-%d_%H:%M:%S)


for mr in 0.01 0.03 0.05 0.07 0.1 0.15 0.2 0.25 0.3
do
    for ratio in 0.05 # 0.1 0.2
    do
        echo training:  ${model}_${dataset}_blended_${pattern}_mr${mr}_${ratio}_e${epochs}
        python train.py --gpu ${gpu} --model ${model} --dataset ${dataset} --subset blended_${pattern}_mr${mr}_${ratio} --epochs ${epochs} --bs 128 --lr 1e-2 --optimizer adam --mode cover --log ${log} --disable_prog
        echo -e '\n'
    done
done
