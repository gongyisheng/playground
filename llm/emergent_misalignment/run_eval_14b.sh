base_dir="/media/hdddisk/yisheng/replicate/emergent_misalignment" # pc
# base_dir="/home/yisheng/replicate/emergent_misalignment" # gpu-1
python3 eval.py --base_dir $base_dir --model_size 14 --ckpt 100 --n_sample 100
python3 eval.py --base_dir $base_dir --model_size 14 --ckpt 150 --n_sample 100
python3 eval.py --base_dir $base_dir --model_size 14 --ckpt 200 --n_sample 100