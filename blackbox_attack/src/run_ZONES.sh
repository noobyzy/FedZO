#CUDA_VISIBLE_DEVICES=6 python attack_main.py --num_users=10 --dataset=cifar10 --lr=0.0002 --unequal=0 --overlap=0

CUDA_VISIBLE_DEVICES=6 python attack_main.py \
                        --num_users=10       \
                        --frac=1.0           \
                        --epoch=300          \
                        --local_ep=100      \
                        --local_bs=25       \
                        --lr=0.000002           \
                        --dataset=cifar10    \
                        --unequal=0          \
                        --overlap=0          \
                        --file_name=bs=100   \
                        --seed=2022          \
                        --solver=ZONES       \
                        --balance=1.0        \
                        --target_label=4     \
                        --num_dir=1          \
                        --step_size=0.001    \
                        --overAir=0          \
                        --SNR=0
