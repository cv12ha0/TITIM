gpu=${1:-0}
trigger=bpp
dataset=${2:-cifar10}  # cifar10 gtsrb
model=${3:-resnet18}  # vgg16
epochs=100
ratio=${4:-0.1}

depth=7
echo $(date +%Y-%m-%d_%H:%M:%S)
# train
for depth in 8 7 6 5 4 3 2 1 0
do
    echo cmd: python train.py --gpu ${gpu} --model ${model} --dataset ${dataset} --subset bpp_d${depth}_ditF_${ratio} --epochs ${epochs} --bs 256 --lr 1e-2 --optimizer adam --log logs/bpp.tsv --mode cover --split_val test --disable_prog
    echo training:  ${model}_${dataset}_bpp_d${depth}_ditF_${ratio}_e${epochs}
    python train.py --gpu ${gpu} --model ${model} --dataset ${dataset} --subset bpp_d${depth}_ditF_${ratio} --epochs ${epochs} --bs 256 --lr 1e-2 --optimizer adam --log logs/bpp.tsv --mode cover --split_val test --disable_prog
    echo -e '\n'
done

# # train
# for depth in 8 7 6 5 4 3 2 1 0
# do
#     echo cmd: python train.py --gpu ${gpu} --model ${model} --dataset ${dataset} --subset bpp_d${depth}_ditT_${ratio} --epochs ${epochs} --bs 256 --lr 1e-2 --optimizer adam --log logs/bpp.tsv --mode cover --split_val test --disable_prog
#     echo training:  ${model}_${dataset}_bpp_d${depth}_ditT_${ratio}_e${epochs}
#     python train.py --gpu ${gpu} --model ${model} --dataset ${dataset} --subset bpp_d${depth}_ditT_${ratio} --epochs ${epochs} --bs 256 --lr 1e-2 --optimizer adam --log logs/bpp.tsv --mode cover --split_val test --disable_prog
#     echo -e '\n'
# done
