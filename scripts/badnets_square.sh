# This is the scripts to reproduce the heatmap in line 1 of Figure 6 for Cifar-10.

# params
gpu=0
dataset=cifar10  
model=resnet18
trigger=badnets
target=0
ratio=0.05
epochs=100
b=4
bn=3
ppt=1
log=cross_badnets_square.tsv


# 1. inject
for mr in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
do
    echo injecting:  data/${dataset}/badnets_b4_bn3_ppt1_mr${mr}_${ratio}
    python inject.py --dataset ${dataset} --target ${target} --ratio ${ratio} --trigger ${trigger} --mr ${mr} --block_size 4 --fixed --split_val test
done

# 2. train
echo $(date +%Y-%m-%d_%H:%M:%S)
for mr in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
do
    echo training:  ${model}_${dataset}_badnets_b${b}_bn${bn}_ppt${ppt}_mr${mr}_${ratio}_e${epochs}
    python train.py --gpu ${gpu} --model ${model} --dataset ${dataset} --subset badnets_b${b}_bn${bn}_ppt${ppt}_mr${mr}_${ratio} --epochs ${epochs} --bs 128 --lr 1e-2 --optimizer adam --log logs/badnets.tsv --mode cover --split_val test --disable_prog
    printf '\n'
done

# 3. cross test (inference)
echo $(date +%Y-%m-%d_%H:%M:%S)
# delete log file if exist
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

# 4. draw heatmap
python utils/scripts/tsv_reader.py