gpu=${1:-0}
dataset=${2:-cifar10}
model=${3:-resnet18}
echo $(date +%Y-%m-%d_%H:%M:%S)


# Blended
pattern=hellokitty  # noise2 noise2png
tpattern=${pattern}   # ${4:-noise}
epochs=100
ratio=0.05  # 0.1 0.2
log=cross_blend.tsv

# delete if exist, add header
if [ -f "logs/${log}" ]; then  
  rm logs/${log}
fi
printf "name\tasr" > logs/${log}

for mr in 0.01 0.03 0.05 0.07 0.1 0.15 0.2 0.25 0.3
do
    for tmr in 0.01 0.03 0.05 0.07 0.1 0.15 0.2 0.25 0.3
    do
        python evaluate.py --gpu ${gpu} --model_dir ${model}_${dataset}_blended_${pattern}_mr${mr}_${ratio}_e${epochs} --dataset ${dataset} --subset blended_${tpattern}_mr${tmr}_${ratio} --split test --log ${log} --name ${model}_${dataset}_blended_${pattern}_mr${mr}_${ratio}_e${epochs}_t${tpattern}_mr${tmr} 
    done
done
printf '\n\n\n\n' >> logs/${log}
