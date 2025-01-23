gpu=${1:-0}
dataset=${2:-cifar10} # cifar10 gtsrb
model=${3:-resnet18}
target=0
ratio=0.01
bs=128
log=defense_featurere.csv
echo $(date +%Y-%m-%d_%H:%M:%S)



# badnets
for ratio in 0.01  # 0.05 0.1 0.2
do
for i in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
do
    python defense.py --gpu=${gpu} --dataset=${dataset} --subset=clean --split=test_sample_10 --model_dir=${model}_${dataset}_badnets_b4_bn3_ppt1_mr${i}_${ratio}_e100 --defense=featurere --log ${log}
done
done
printf '\n\n\n\n' >> logs/${log}


