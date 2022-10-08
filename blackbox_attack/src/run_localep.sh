#CUDA_VISIBLE_DEVICES=6 python attack_main.py --num_users=10 --dataset=cifar10 --lr=0.0002 --unequal=0 --overlap=0
if true; then
for var in 5 10 20 50
do
CUDA_VISIBLE_DEVICES=6 python attack_main.py \
                        --num_users=10       \
                        --frac=1.0           \
                        --epoch=300          \
                        --local_ep=$var      \
                        --local_bs=25       \
                        --lr=0.001           \
                        --dataset=cifar10    \
                        --unequal=0          \
                        --overlap=0          \
                        --file_name=H=$var   \
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