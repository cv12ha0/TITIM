gpu=${1:-0}
dataset=${2:-cifar10}  # cifar10 gtsrb
model=${3:-resnet18}
target=0
ratio=0.01
layer=${4:-avgpool}  # avgpool fc linear
log=${5:-defense_ss.csv}
echo $(date +%Y-%m-%d_%H:%M:%S)



# BadNets (Square)
for ratio in 0.01  # 0.05 0.1 0.2 
do
for i in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
do
    python defense.py --gpu=${gpu}  --target=${target} --dataset=${dataset} --subset=badnets_b4_bn3_ppt1_mr${i}_${ratio} --split=train --defense=ss --model_dir=${model}_${dataset}_badnets_b4_bn3_ppt1_mr${i}_${ratio}_e100 --layer_name ${layer} --poison_ratio=${ratio} --log ${log}
done
done

