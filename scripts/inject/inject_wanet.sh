dataset=cifar10  # mnist cifar10 gtsrb celeba8
target=0

# WaNet
trigger=wanet
k=8
cr=0.0  # 2.0
gs=1.0
ratio=0.05  # 0.2
for s in 0.2 0.4 0.6 0.8 1.0 1.2 1.4 1.6 1.8 2.0
do
    echo injecting:  wanet_cr${cr}_s${s}_k${k}_gs${gs}_${ratio}
    python inject.py --dataset ${dataset} --target ${target} --ratio ${ratio} --trigger ${trigger} --cross_ratio ${cr} --s ${s} --k ${k} --grid_rescale ${gs}
done