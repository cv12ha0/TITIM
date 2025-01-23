gpu=${1:-0}
trigger=styled
dataset=${2:-cifar10}  # cifar10 gtsrb
model=${3:-resnet18}  # vgg16
epochs=100
ratio=${4:-0.05}
filter=${5:-gotham}  # gotham  nashville  kelvin  toatser  lomo
echo $(date +%Y-%m-%d_%H:%M:%S)

# train
for mr in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
do  
    echo cmd: python train.py --gpu ${gpu} --model ${model} --dataset ${dataset} --subset styled_${filter}_mr${mr}_wof_${ratio} --epochs ${epochs} --bs 128 --lr 1e-2 --optimizer adam --log logs/styled.tsv --mode cover --split_val test
    echo training:  ${model}_${dataset}_styled_${filter}_mr${mr}_wof_${ratio}_e${epochs}
    python train.py --gpu ${gpu} --model ${model} --dataset ${dataset} --subset styled_${filter}_mr${mr}_wof_${ratio} --epochs ${epochs} --bs 256 --lr 1e-2 --optimizer adam --log logs/styled.tsv --mode cover --split_val test --disable_prog
    echo -e '\n'
done
