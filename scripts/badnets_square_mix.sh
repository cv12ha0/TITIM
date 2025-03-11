# This is the scripts to reproduce the upper heatmap of Figure 21c (Intensity mixing with opacity).
# params
gpu=0
trigger=badnets
model=resnet18
dataset=cifar10
target=0
mixmr=0.1
ratio=0.05
b=4
bn=3
ppt=1
epochs=100
log=cross_badnets_mixmr${mixmr}_${ratio}_${model}.tsv


# 1. inject
# inject with single intensity
python inject.py --dataset ${dataset} --target ${target} --ratio ${ratio}  --trigger ${trigger} --mr ${mixmr} --block_size 4 --fixed --split_val test

# inject with another intensity
for mr in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
do
    for ratio in 0.05
    do
        echo injecting:  data/${dataset}/badnets_b4_bn3_ppt1_mr${mr}_${ratio}
        python inject.py --dataset ${dataset} --target ${target} --ratio ${ratio} --ratio_start ${ratio} --trigger ${trigger} --mr ${mr} --block_size 4 --fixed --split_val test --from_subset badnets_b4_bn3_ppt1_mr${mixmr}_${ratio} --output_dir data/cifar10/badnets_mixtest_${ratio}mr${mixmr}_${ratio}mr${mr}
    done
done


# 2. train
echo $(date +%Y-%m-%d_%H:%M:%S)
for mr in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
do
    python train.py --gpu ${gpu} --model ${model} --dataset ${dataset} --subset badnets_mixtest_${ratio}mr${mixmr}_${ratio}mr${mr} --epochs ${epochs} --bs 128 --lr 1e-2 --optimizer adam --mode cover --disable_prog  # --save_weight
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
        # python train.py --gpu ${gpu} --model ${model} --dataset ${dataset} --subset badnets_mixtest_${ratio}mr0.1_${ratio}mr${mr} --epochs ${epochs} --bs 128 --lr 1e-2 --optimizer adam --mode cover
        python evaluate.py --gpu ${gpu} --model_dir ${model}_${dataset}_badnets_mixtest_${ratio}mr${mixmr}_${ratio}mr${mr}_e100 --dataset ${dataset} --subset badnets_b4_bn3_ppt1_mr${tmr}_0.05 --split test --log ${log} --name ${model}_badnets_mixtest_${ratio}mr${mixmr}_${ratio}mr${mr}_tmr${tmr}
    done
done
printf '\n\n\n\n' >> logs/${log}

# 4. draw heatmap
python utils/scripts/draw_heatmap.py --name cross_badnets_mixmr0.1_0.05_resnet18 --N 10 --tick mix