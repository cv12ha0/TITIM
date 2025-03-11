trigger=bpp
dataset=cifar10 # mnist cifar10 gtsrb celeba8
depth=${1:-7}  # 40
target=0


# inject
for depth in 8 7 6 5 4 3 2 1 0
do
    for ratio in 0.05  # 0.1
    do
        echo injecting:  data/${dataset}/bpp_d${depth}_ditF_${ratio}
        python inject.py --dataset ${dataset} --target ${target} --ratio ${ratio} --trigger ${trigger} --depth ${depth} # --dither
    done
done

# inject (with dithering)
for depth in 8 7 6 5 4 3 2 1 0
do
    for ratio in 0.05  # 0.1
    do
        echo injecting:  data/${dataset}/bpp_d${depth}_ditT_${ratio}
        python inject.py --dataset ${dataset} --target ${target} --ratio ${ratio} --trigger ${trigger} --depth ${depth} --dither
    done
done
