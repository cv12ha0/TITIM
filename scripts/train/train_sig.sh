gpu=${1:-0}
trigger=sig
dataset=${2:-cifar10}  # cifar10 gtsrb
model=${3:-resnet18}
epochs=100
ratio=${4:-0.1}
delta=40
f=6
echo $(date +%Y-%m-%d_%H:%M:%S)


for delta in 2 4 6 8 10 12 14 16 18 20  # 10 20 30 40 50 60 70 80
do
    echo training:  ${model}_${dataset}_sig_d${delta}_f${f}_${ratio}_e${epochs}
    python train.py --gpu ${gpu} --model ${model} --dataset ${dataset} --subset sig_d${delta}_f${f}_${ratio} --epochs ${epochs} --bs 128 --lr 1e-2 --optimizer adam --log logs/sig.tsv --mode cover --split_val test --disable_prog
    printf '\n'
done

