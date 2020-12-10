#This page is being modified.

# 1. preprocessing
## move to folder 'preprocessing_data'

    python preprocessing_data.py --row_data_dir=/mnt/home/cmc/lrcn/cmc_data/ --ucf_list_dir=/mnt/home/cmc/lrcn/Data_CMC/cmcTrainTestlist/ --dataset=CMC --sampling_rate=100 --ucf101_fps=1

## Requires modification of each parameter

# 2. Training
## move to folder 'LRCN'

    python train.py --sampled_data_dir=/mnt/home/cmc/lrcn/CMC_sampled_data_video/ --ucf_list_dir=/mnt/home/cmc/lrcn/Data_CMC/cmcTrainTestlist/ --number_of_classes=2 --batch-size=16 --latent_dim=512 --hidden_size=256 --lstm_layers=2 --epoch=30 --val_check_interval=5

# 3. Test

    python test.py  --model_dir=/mnt/home/cmc/lrcn/LRCN/20200212-155451/Saved_model_checkpoints/ --sampled_data_dir=/mnt/home/cmc/lrcn/CMC_sampled_data_video/ --ucf_list_dir=/mnt/home/cmc/lrcn/Data_CMC/cmcTrainTestlist/ --number_of_classes=2 --model_name='epoch_30.pth.tar'
