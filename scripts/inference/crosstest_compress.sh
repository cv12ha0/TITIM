gpu=${1:-0}
dataset=${2:-cifar10}
model=${3:-resnet18}
echo $(date +%Y-%m-%d_%H:%M:%S)

# Compress
trigger=compress
epochs=100
ratio=${4:-0.05}
alg=${5:-jpgpil}
quality=100
log=cross_compress.tsv

# delete if exist, add header
if [ -f "logs/${log}" ]; then  
  rm logs/${log}
fi
printf "name\tasr" > logs/${log}

for q in 100 90 80 70 60 50 40 30 20 10 0
do
    for tq in 100 90 80 70 60 50 40 30 20 10 0 
    do  
        python evaluate.py --gpu ${gpu} --model_dir ${model}_${dataset}_compress_${alg}_q${q}_${ratio}_e${epochs} --dataset ${dataset} --subset compress_${alg}_q${tq}_${ratio} --split test --log ${log} --name ${model}_${dataset}_compress_${alg}_q${q}_${ratio}_e${epochs}_tq${tq}
        
    done
done
printf '\n\n\n\n' >> logs/${log}
