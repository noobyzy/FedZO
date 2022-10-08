#CUDA_VISIBLE_DEVICES=6 python attack_main.py --num_users=10 --dataset=cifar10 --lr=0.0002 --unequal=0 --overlap=0
if true; then
for var in 1 5 25 125
do
CUDA_VISIBLE_DEVICES=4 python attack_main.py \
                        --num_users=10       \
                        --frac=1.0           \
                        --epoch=100          \
                        --local_ep=20         \
                        --local_bs=100       \
                        --lr=0.001           \
                        --dataset=cifar10    \
                        --unequal=0          \
                        --overlap=0          \
                        --file_name=q=$var   \
                        --seed=2022          \
                        --solver=FedZO       \
                        --balance=1.0        \
                        --target_label=4     \
                        --num_dir=$var       \
                        --step_size=0.001    \
                        --overAir=0          \
                        --SNR=0
done
fi