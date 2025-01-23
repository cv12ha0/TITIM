gpu=${1:-0}
dataset=${2:-cifar10} # cifar10 gtsrb
model=${3:-resnet18}
target=0
ratio=0.01
thres=${4:-0.5}
log=${5:-defense_scaleup.csv}
echo $(date +%Y-%m-%d_%H:%M:%S)


# BadNets (Square)
for ratio in 0.01  # 0.05 0.1 0.2 
do
for i in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
do
    python defense.py --gpu=${gpu} --dataset=${dataset} --subset=clean --split=train_sample_100 --subset_val=badnets_b4_bn3_ppt1_mr${i}_${ratio} --model_dir=${model}_${dataset}_badnets_b4_bn3_ppt1_mr${i}_${ratio}_e100 --split_val=test --defense=scaleup --scaleup_thres=${thres} --poison_ratio=0.5 --log ${log}
done
done




# # BadNets (Square) cross 
# for ratio in 0.01  # 0.05 0.1 0.2
# do
# for mr in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
# do
#     for tmr in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
#     do
#         python defense.py --gpu=${gpu} --dataset=${dataset} --subset=clean --split=train_sample_100 --subset_val=badnets_b4_bn3_ppt1_mr${tmr}_${ratio} --model_dir=${model}_${dataset}_badnets_b4_bn3_ppt1_mr${mr}_${ratio}_e100 --split_val=test --defense=scaleup --scaleup_thres=${thres} --poison_ratio=0.5 --log ${log} --name ${model}_${dataset}_badnets_b4_bn3_ppt1_mr${mr}_${ratio}_e100_tmr${tmr}
#     done
# done
# done
# printf '\n\n\n\n' >> logs/${log}

