gpu=${1:-0}
trigger=badnets
dataset=${2:-cifar10}  # cifar10 gtsrb
model=${3:-resnet18}  # vgg16
epochs=100
ratio=${4:-0.1}
b=4
bn=3
ppt=1
echo $(date +%Y-%m-%d_%H:%M:%S)

# train
for mr in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
do
    echo training:  ${model}_${dataset}_badnets_b${b}_bn${bn}_ppt${ppt}_mr${mr}_${ratio}_e${epochs}
    python train.py --gpu ${gpu} --model ${model} --dataset ${dataset} --subset badnets_b${b}_bn${bn}_ppt${ppt}_mr${mr}_${ratio} --epochs ${epochs} --bs 128 --lr 1e-2 --optimizer adam --log logs/badnets.tsv --mode cover --split_val test --disable_prog
    printf '\n'
done
