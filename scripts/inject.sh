dataset=cifar10  # cifar10 gtsrb
target=0



# BadNets (Square)
trigger=badnets
ratio=0.01  # 0.05 0.1 0.2
for mr in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
do
    echo injecting:  data/${dataset}/badnets_b4_bn3_ppt1_mr${mr}_${ratio}
    python inject.py --dataset ${dataset} --target ${target} --ratio ${ratio} --trigger ${trigger} --mr ${mr} --block_size 4 --fixed --split_val test
done


# # BadNets (pokemon ...)
# trigger=patch
# pattern=pokemon  # flower bomb
# mr=1.0
# ratio=0.05  # 0.1 0.2
# for sz in 2 3 4 5 6 7 8 9 10  # 8 7 6 5 4 3 2
# do  
#     loc=$((27-(${sz}+1)/2))
#     # loc=$((32-${sz}))
#     echo injecting: data/${dataset}/patch_${pattern}_mr${mr}_sz${sz}_loc${loc}_${ratio}
#     python inject.py --dataset ${dataset} --target ${target} --ratio ${ratio} --trigger ${trigger} --pattern ${pattern} --mr ${mr} --patch_size ${sz} --patch_loc ${loc} --split_val test
#     printf '\n'
# done



# # Blended
# trigger=blended
# pattern=hellokitty  # noise2 noisepng
# ratio=0.05  # 0.1 0.2
# do
# for mr in 0.01 0.03 0.05 0.07 0.1 0.15 0.2 0.25 0.3
# do
#     echo injecting:  data/${dataset}/blended_${pattern}_mr${mr}_${ratio}
#     python inject.py --dataset ${dataset} --target ${target} --ratio ${ratio} --trigger ${trigger} --mr ${mr} --pattern ${pattern}
# done


# # Compress
# trigger=compress
# alg=jpgpil  # jpgcv
# ratio=0.05  # 0.1 0.2
# for quality in 10 20 30 40 50 60 70 80 90
# do
#     echo injecting:  data/${dataset}/compress_${alg}_q${quality}_${ratio}
#     python inject.py --dataset ${dataset} --target ${target} --ratio ${ratio} --trigger ${trigger} --compress_alg ${alg} --compress_quality ${quality}
# done


# # SIG
# trigger=sig
# f=6
# ratio=0.05  # 0.1 0.2
# for delta in 2 4 6 8 10 12 14 16 18 20  # 10 20 30 40 50 60 70 80
# do
#     echo injecting:  data/${dataset}/sig_d${delta}_f${f}_${ratio}
#     python inject.py --dataset ${dataset} --target ${target} --ratio ${ratio} --trigger ${trigger} --delta ${delta} --f ${f}
# done


# # WaNet
# trigger=wanet
# k=8
# cr=0.0  # 2.0
# gs=1.0
# ratio=0.05  # 0.2
# for s in 0.2 0.4 0.6 0.8 1.0 1.2 1.4 1.6 1.8 2.0
# do
#     echo injecting:  wanet_cr${cr}_s${s}_k${k}_gs${gs}_${ratio}
#     python inject.py --dataset ${dataset} --target ${target} --ratio ${ratio} --trigger ${trigger} --cross_ratio ${cr} --s ${s} --k ${k} --grid_rescale ${gs}
# done
