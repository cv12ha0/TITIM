gpu=${1:-0}
dataset=${2:-cifar10}
model=${3:-resnet18}
echo $(date +%Y-%m-%d_%H:%M:%S)


# BadNets(Pokemon/Bomb/Flower trigger)
trigger=patch
epochs=100
pattern=${4:-pokemon}
mr=${5:-1.0}
ratio=${6:-0.05}
sz=7
loc=25  # $((28-(${sz}+1)/2))  loc=$((32-${sz}))
log=cross_badnets.tsv

# delete if exist, add header
if [ -f "logs/${log}" ]; then  
  rm logs/${log}
fi
printf "name\tasr" > logs/${log}

# cross_test
for pattern in pokemon # bomb flower
do
for mr in 1.0
do
for ratio in 0.05 # 0.05 0.1 0.2
do
    for sz in 2 3 4 5 6 7 8 9 10
    do
        for tsz in 2 3 4 5 6 7 8 9 10 
        do  
            # center
            loc=$((27-(${sz}+1)/2))
            tloc=$((27-(${tsz}+1)/2))
            python evaluate.py --gpu ${gpu} --model_dir ${model}_${dataset}_patch_${pattern}_mr${mr}_sz${sz}_loc${loc}_${ratio}_e${epochs} --dataset ${dataset} --subset patch_${pattern}_mr${mr}_sz${tsz}_loc${tloc}_${ratio} --split test --log ${log} --name ${model}_${dataset}_patch_${pattern}_mr${mr}_sz${sz}_loc${loc}_${ratio}_e${epochs}_tsz${tsz}
        done
    done
    printf '\n\n\n\n' >> logs/${log}
done
done
done

