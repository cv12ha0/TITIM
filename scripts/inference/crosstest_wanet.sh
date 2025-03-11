gpu=${1:-0}
dataset=${2:-cifar10}
model=${3:-resnet18}
echo $(date +%Y-%m-%d_%H:%M:%S)



# WaNet
cr=0.0  # 2.0
k=8
gs=1.0
epochs=100
ratio=0.05
log=cross_wanet.tsv

# delete if exist, add header
if [ -f "logs/${log}" ]; then  
  rm logs/${log}
fi
printf "name\tasr" > logs/${log}

for s in 0.2 0.4 0.6 0.8 1.0 1.2 1.4 1.6 1.8 2.0
do
    for ts in 0.2 0.4 0.6 0.8 1.0 1.2 1.4 1.6 1.8 2.0
    do  
        # echo taskset -c ${cpu} python evaluate_wanet.py --gpu ${gpu} --model_dir resnet18_cifar10_wanet_cr${cr}_s${s}_k${k}_gs${gs}_${ratio}_e${epochs} --target 0 --cross_ratio=${cr} --s=${ts} --k=${k} --grid_rescale=${gs} --fixed --dataset cifar10 --subset clean --split test --log cross_wanet.tsv --name wanet_cr${cr}_s${s}_k${k}_gs${gs}_${ratio}_e${epochs}_ts${ts}        
        echo python evaluate_wanet.py --gpu ${gpu} --model_dir ${model}_${dataset}_wanet_cr${cr}_s${s}_k${k}_gs${gs}_${ratio}_e${epochs} --target 0 --cross_ratio=${cr} --s=${ts} --k=${k} --grid_rescale=${gs} --fixed --dataset ${dataset} --subset clean --split test --log ${log} --name ${model}_${dataset}_wanet_cr${cr}_s${s}_k${k}_gs${gs}_${ratio}_e${epochs}_ts${ts}        
        python evaluate_wanet.py --gpu ${gpu} --model_dir ${model}_${dataset}_wanet_cr${cr}_s${s}_k${k}_gs${gs}_${ratio}_e${epochs} --target 0 --cross_ratio=${cr} --s=${ts} --k=${k} --grid_rescale=${gs} --fixed --dataset ${dataset} --subset clean --split test --log ${log} --name ${model}_${dataset}_wanet_cr${cr}_s${s}_k${k}_gs${gs}_${ratio}_e${epochs}_ts${ts}        
        printf '\n'
    done
done
printf '\n\n\n\n' >> logs/${log}
