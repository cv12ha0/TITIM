gpu=${1:-0}
dataset=${2:-cifar10} # cifar10 gtsrb
model=${3:-resnet18}
target=0
ratio=0.1
epochs=${4:-120}  # 30
lr=${5:-1e-1}
bs=128
log=${6:-defense_nc.csv}
echo $(date +%Y-%m-%d_%H:%M:%S)



# badnets
for ratio in 0.01  # 0.05 0.1 0.2
do
for i in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
do
    python defense.py --gpu=${gpu} --dataset=${dataset} --subset=clean --split=test --model_dir=${model}_${dataset}_badnets_b4_bn3_ppt1_mr${i}_${ratio}_e100 --defense=nc --lr=${lr} --bs=${bs} --epochs=${epochs} --log ${log}
done
done
printf '\n\n\n\n' >> logs/${log}

