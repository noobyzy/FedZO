#CUDA_VISIBLE_DEVICES=6 python attack_main.py --num_users=10 --dataset=cifar10 --lr=0.0002 --unequal=0 --overlap=0
if true; then
for var in 0.1 0.2 0.5 1.0
do
CUDA_VISIBLE_DEVICES=7 python attack_main.py \
                        --num_users=50       \
                        --frac=$var          \
                        --epoch=300          \
                        --local_ep=20         \
                        --local_bs=25       \
                        --lr=0.001           \
                        --dataset=cifar10    \
                        --unequal=1          \
                        --overlap=0        \
                        --file_name=p=$var   \
                        --seed=2022          \
                        --solver=FedZO       \
                        --balance=1.0        \
                        --target_label=4     \
                        --num_dir=20          \
                        --step_size=0.001    \
                        --overAir=0          \
                        --SNR=0
done
fi