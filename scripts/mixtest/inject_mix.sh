trigger=badnets
dataset=cifar10  # cifar10 gtsrb
target=0
mixmr=0.1  # 1.0
ratio=0.05

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

