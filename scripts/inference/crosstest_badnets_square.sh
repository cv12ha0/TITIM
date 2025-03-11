gpu=${1:-0}
dataset=${2:-cifar10}
model=${3:-resnet18}
echo $(date +%Y-%m-%d_%H:%M:%S)


# BadNets(square trigger)
epochs=100
ratio=0.05
log=cross_badnets_square.tsv

# delete if exist, add header
if [ -f "logs/${log}" ]; then  
  rm logs/${log}
fi
printf "name\tasr" > logs/${log}

for mr in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
do
    for tmr in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
    do
        python evaluate.py --gpu ${gpu} --model_dir ${model}_${dataset}_badnets_b4_bn3_ppt1_mr${mr}_${ratio}_e${epochs} --dataset ${dataset} --subset badnets_b4_bn3_ppt1_mr${tmr}_${ratio} --split test --log ${log} --name ${model}_${dataset}_badnets_b4_bn3_ppt1_mr${mr}_${ratio}_e${epochs}_tmr${tmr}
        
    done
done
printf '\n\n\n\n' >> logs/${log}
