#CUDA_VISIBLE_DEVICES=6 python attack_main.py --num_users=10 --dataset=cifar10 --lr=0.0002 --unequal=0 --overlap=0
if true; then
CUDA_VISIBLE_DEVICES=6 python attack_main.py \
                        --num_users=50       \
                        --frac=0.1           \
                        --epoch=100          \
                        --local_ep=20         \
                        --local_bs=25       \
                        --lr=0.001           \
                        --dataset=cifar10    \
                        --unequal=1          \
                        --overlap=0        \
                        --file_name=SNR=noise-free \
                        --seed=2022          \
                        --solver=FedZO       \
                        --balance=1.0        \
                        --target_label=4     \
                        --num_dir=20          \
                        --step_size=0.001    \
                        --overAir=1          \
                        --SNR=100
fi

#SNR=100 is default to noise-free

if true; then
for var in -10 -5 0 
do
CUDA_VISIBLE_DEVICES=6 python attack_main.py \
                        --num_users=50       \
                        --frac=0.1           \
                        --epoch=100          \
                        --local_ep=20         \
                        --local_bs=25       \
                        --lr=0.001           \
                        --dataset=cifar10    \
                        --unequal=1          \
                        --overlap=0        \
                        --file_name=SNR=$var \
                        --seed=2022          \
                        --solver=FedZO       \
                        --balance=1.0        \
                        --target_label=4     \
                        --num_dir=20          \
                        --step_size=0.001    \
                        --overAir=1          \
                        --SNR=$var
done
fi