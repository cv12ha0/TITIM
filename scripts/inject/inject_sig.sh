dataset=cifar10  # mnist cifar10 gtsrb celeba8
target=0

# SIG
trigger=sig
f=6
ratio=0.05  # 0.1 0.2
for delta in 2 4 6 8 10 12 14 16 18 20  # 10 20 30 40 50 60 70 80
do
    echo injecting:  data/${dataset}/sig_d${delta}_f${f}_${ratio}
    python inject.py --dataset ${dataset} --target ${target} --ratio ${ratio} --trigger ${trigger} --delta ${delta} --f ${f}
done
