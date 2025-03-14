trigger=badnets
gpu=${1:-0}
dataset=${2:-cifar10}  # cifar10 gtsrb timgnet
model=${3:-resnet18}  # resnet18
epochs=100
ratio=${4:-0.05}
# verbose=${3:-wopre}
b=4
bn=3
ppt=1
echo $(date +%Y-%m-%d_%H:%M:%S)

mixmr=${5:-0.1}
printf "name\tasr" > logs/cross_badnets_mixmr${mixmr}_${ratio}_${model}.tsv
for mr in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
do
    for tmr in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
    do
        python evaluate.py --gpu ${gpu} --model_dir ${model}_${dataset}_badnets_mixtest_${ratio}mr${mixmr}_${ratio}mr${mr}_e100 --dataset ${dataset} --subset badnets_b4_bn3_ppt1_mr${tmr}_0.05 --split test --log cross_badnets_mixmr${mixmr}_${ratio}_${model}.tsv --name ${model}_badnets_mixtest_${ratio}mr${mixmr}_${ratio}mr${mr}_tmr${tmr}
    done
done
