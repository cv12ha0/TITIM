gpu=${1:-0}
dataset=${2:-cifar10}
model=${3:-resnet18}
echo $(date +%Y-%m-%d_%H:%M:%S)


# SIG
f=6
epochs=100
ratio=0.05  # 0.1 0.2
log=cross_sig.tsv

# delete if exist, add header
if [ -f "logs/${log}" ]; then  
  rm logs/${log}
fi
printf "name\tasr" > logs/${log}

for d in 2 4 6 8 10 12 14 16 18 20
do
    for td in 2 4 6 8 10 12 14 16 18 20
    do  
        python evaluate.py --gpu ${gpu} --model_dir ${model}_${dataset}_sig_d${d}_f${f}_${ratio}_e${epochs}_1 --dataset ${dataset} --subset sig_d${td}_f${f}_${ratio} --split test --log ${log} --name ${model}_${dataset}_sig_d${d}_f${f}_${ratio}_e${epochs}_td${td}
        printf '\n'
        
    done
done
printf '\n\n\n\n' >> logs/${log}