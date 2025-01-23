dataset=cifar10  # mnist cifar10 gtsrb celeba8
target=0

# Compress
trigger=compress
alg=jpgpil  # jpgcv
ratio=0.05  # 0.1 0.2
for quality in 10 20 30 40 50 60 70 80 90
do
    echo injecting:  data/${dataset}/compress_${alg}_q${quality}_${ratio}
    python inject.py --dataset ${dataset} --target ${target} --ratio ${ratio} --trigger ${trigger} --compress_alg ${alg} --compress_quality ${quality}
done