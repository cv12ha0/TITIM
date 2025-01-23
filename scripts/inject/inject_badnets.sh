dataset=cifar10  # mnist cifar10 gtsrb celeba8
target=0

# BadNets (pokemon ...)
trigger=patch
pattern=pokemon  # flower bomb
mr=1.0
ratio=0.05  # 0.05 0.1 0.2
for sz in 2 3 4 5 6 7 8 9 10  # 8 7 6 5 4 3 2
do  
    loc=$((27-(${sz}+1)/2))
    # loc=$((32-${sz}))
    echo injecting: data/${dataset}/patch_${pattern}_mr${mr}_sz${sz}_loc${loc}_${ratio}
    python inject.py --dataset ${dataset} --target ${target} --ratio ${ratio} --trigger ${trigger} --pattern ${pattern} --mr ${mr} --patch_size ${sz} --patch_loc ${loc} --split_val test
    printf '\n'
done