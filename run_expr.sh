conda activate 547env


# GPU0
python cifar_train.py --gpu 0 --imb_type exp --imb_factor 0.01 --loss_type LDAM --train_rule Reweight --dataset cifar100 --epochs 600 --cutoff_epoch 480 --start-epoch   &
# GPU1
python cifar_train.py --gpu 0 --imb_type exp --imb_factor 0.01 --loss_type LDAM --train_rule Resample --dataset cifar100
# GPU0
python cifar_train.py --gpu 0 --imb_type exp --imb_factor 0.01 --loss_type LDAM --train_rule DRW --dataset cifar100 &&
# GPU1
python cifar_train.py --gpu 1 --imb_type exp --imb_factor 0.01 --loss_type LDAM --train_rule DRW_AUTO_CLUSTER --dataset cifar100