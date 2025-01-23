gpu=${1:-0}
trigger=patch
dataset=${2:-cifar10}  # cifar10 gtsrb
model=${3:-resnet18}
epochs=100
pattern=${4:-pokemon}
mr=${5:-1.0}
ratio=${6:-0.1}
sz=7
loc=25

echo $(date +%Y-%m-%d_%H:%M:%S)

# train
for sz in 2 3 4 5 6 7 8 9 10
do
    loc=$((27-(${sz}+1)/2))
    # loc=$((32-${sz}))
    echo training:  ${model}_${dataset}_patch_${pattern}_mr${mr}_sz${sz}_loc${loc}_${ratio}_e${epochs}
    python train.py --gpu ${gpu} --model ${model} --dataset ${dataset} --subset patch_${pattern}_mr${mr}_sz${sz}_loc${loc}_${ratio} --epochs ${epochs} --bs 128 --lr 1e-2 --optimizer adam --log logs/patch.tsv --mode cover --split_val test --disable_prog
    printf '\n'
done

