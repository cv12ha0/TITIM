trigger=styled
dataset=cifar10  # cifar10 gtsrb celeba8
filter=${1:-gotham}  # gotham  nashville  kelvin  toatser  lomo
mr=${2:-1.0}
target=0

# inject
for filter in gotham kelvin lomo  # gotham nashville kelvin toatser lomo
do
for mr in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
do
    for ratio in 0.05  # 0.05 0.1 0.2
    do
        echo injecting:  data/${dataset}/styled_${filter}_mr${mr}_wof_${ratio}
        python inject.py --dataset ${dataset} --target ${target} --ratio ${ratio} --trigger ${trigger} --filter ${filter} --mr ${mr}
    done
done
done
