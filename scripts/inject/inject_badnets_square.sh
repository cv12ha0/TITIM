dataset=cifar10  # mnist cifar10 gtsrb celeba8
target=0

# BadNets (Square)
trigger=badnets
ratio=0.01  # 0.05 0.1 0.2
for mr in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
do
    echo injecting:  data/${dataset}/badnets_b4_bn3_ppt1_mr${mr}_${ratio}
    python inject.py --dataset ${dataset} --target ${target} --ratio ${ratio} --trigger ${trigger} --mr ${mr} --block_size 4 --fixed --split_val test
done