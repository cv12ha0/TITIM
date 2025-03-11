gpu=${1:-0}
dataset=${2:-cifar10}
model=${3:-resnet18}
echo $(date +%Y-%m-%d_%H:%M:%S)

# BppAttack
trigger=bpp
epochs=100
ratio=${4:-0.05}
log=cross_bpp.tsv
echo $(date +%Y-%m-%d_%H:%M:%S)

# delete if exist, add header
if [ -f "logs/${log}" ]; then  
  rm logs/${log}
fi
printf "name\tasr" > logs/${log}

# cross_test
for depth in 8 7 6 5 4 3 2 1 0
do
    for tdepth in 8 7 6 5 4 3 2 1 0
    do  
        python evaluate.py --gpu ${gpu} --model_dir ${model}_${dataset}_bpp_d${depth}_ditF_${ratio}_e${epochs} --dataset ${dataset} --subset bpp_d${tdepth}_ditF_${ratio} --split test --log ${log} --name ${model}_${dataset}_bpp_d${depth}_ditF_${ratio}_e${epochs}_td${tdepth}
    done
done
printf '\n\n\n\n' >> logs/${log}
