dataset=cifar10  # mnist cifar10 gtsrb celeba8
target=0

# Blended
trigger=blended
pattern=hellokitty  # noise2 noisepng
ratio=0.05  # 0.1 0.2
do
for mr in 0.01 0.03 0.05 0.07 0.1 0.15 0.2 0.25 0.3
do
    echo injecting:  data/${dataset}/blended_${pattern}_mr${mr}_${ratio}
    python inject.py --dataset ${dataset} --target ${target} --ratio ${ratio} --trigger ${trigger} --mr ${mr} --pattern ${pattern}
done