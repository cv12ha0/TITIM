gpu=${1:-0}
dataset=${2:-cifar10}
model=${3:-resnet18}
echo $(date +%Y-%m-%d_%H:%M:%S)


# BadNets(square trigger)
epochs=100
ratio=0.01
log=cross_badnets.tsv

for tmr in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
do
    for mr in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
    do
        python evaluate.py --gpu ${gpu} --model_dir ${model}_${dataset}_badnets_b4_bn3_ppt1_mr${mr}_${ratio}_e${epochs} --dataset ${dataset} --subset badnets_b4_bn3_ppt1_mr${tmr}_${ratio} --split test --log ${log} --name ${model}_${dataset}_badnets_b4_bn3_ppt1_mr${mr}_${ratio}_e${epochs}_tmr${tmr}
        
    done
done
printf '\n\n\n\n' >> logs/${log}



# # Blended
# pattern=hellokitty  # noise2 noise2png
# tpattern=${pattern}   # ${4:-noise}
# epochs=100
# ratio=0.05  # 0.1 0.2
# log=cross_blend.tsv

# for tmr in 0.01 0.03 0.05 0.07 0.1 0.15 0.2 0.25 0.3
# do
#     for mr in 0.01 0.03 0.05 0.07 0.1 0.15 0.2 0.25 0.3
#     do
#         python evaluate.py --gpu ${gpu} --model_dir ${model}_${dataset}_blended_${pattern}_mr${mr}_${ratio}_e${epochs} --dataset ${dataset} --subset blended_${tpattern}_mr${tmr}_${ratio} --split test --log ${log} --name ${model}_${dataset}_blended_${pattern}_mr${mr}_${ratio}_e${epochs}_t${tpattern}_mr${tmr} 
#     done
# done
# printf '\n\n\n\n' >> logs/${log}



# # SIG
# f=6
# epochs=100
# ratio=0.05  # 0.1 0.2
# log=cross_sig.tsv

# for d in 2 4 6 8 10 12 14 16 18 20
# do
#     for td in 2 4 6 8 10 12 14 16 18 20
#     do  
#         python evaluate.py --gpu ${gpu} --model_dir ${model}_${dataset}_sig_d${d}_f${f}_${ratio}_e${epochs}_1 --dataset ${dataset} --subset sig_d${td}_f${f}_${ratio} --split test --log cross_sig_d.tsv --name ${model}_${dataset}_sig_d${d}_f${f}_${ratio}_e${epochs}_td${td}
#         printf '\n'
        
#     done
# done
# printf '\n\n\n\n' >> logs/cross_sig_d.tsv



# # WaNet
# cr=0.0  # 2.0
# k=8
# gs=1.0
# epochs=100
# ratio=0.05
# log=cross_wanet.tsv

# for ts in 0.2 0.4 0.6 0.8 1.0 1.2 1.4 1.6 1.8 2.0
# do
#     for s in 0.2 0.4 0.6 0.8 1.0 1.2 1.4 1.6 1.8 2.0
#     do  
#         # echo taskset -c ${cpu} python evaluate_wanet.py --gpu ${gpu} --model_dir resnet18_cifar10_wanet_cr${cr}_s${s}_k${k}_gs${gs}_${ratio}_e${epochs} --target 0 --cross_ratio=${cr} --s=${ts} --k=${k} --grid_rescale=${gs} --fixed --dataset cifar10 --subset clean --split test --log cross_wanet.tsv --name wanet_cr${cr}_s${s}_k${k}_gs${gs}_${ratio}_e${epochs}_ts${ts}        
#         echo python evaluate_wanet.py --gpu ${gpu} --model_dir ${model}_${dataset}_wanet_cr${cr}_s${s}_k${k}_gs${gs}_${ratio}_e${epochs} --target 0 --cross_ratio=${cr} --s=${ts} --k=${k} --grid_rescale=${gs} --fixed --dataset ${dataset} --subset clean --split test --log ${log} --name ${model}_${dataset}_wanet_cr${cr}_s${s}_k${k}_gs${gs}_${ratio}_e${epochs}_ts${ts}        
#         python evaluate_wanet.py --gpu ${gpu} --model_dir ${model}_${dataset}_wanet_cr${cr}_s${s}_k${k}_gs${gs}_${ratio}_e${epochs} --target 0 --cross_ratio=${cr} --s=${ts} --k=${k} --grid_rescale=${gs} --fixed --dataset ${dataset} --subset clean --split test --log ${log} --name ${model}_${dataset}_wanet_cr${cr}_s${s}_k${k}_gs${gs}_${ratio}_e${epochs}_ts${ts}        
#         printf '\n'
#     done
# done
# printf '\n\n\n\n' >> logs/cross_wanet_${dataset}.tsv
