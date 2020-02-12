# 1. 전처리
## preprocessing_data 폴더로 이동

    python preprocessing_data.py --row_data_dir=/mnt/home/cmc/lrcn/cmc_data/ --ucf_list_dir=/mnt/home/cmc/lrcn/Data_CMC/cmcTrainTestlist/ --dataset=CMC --sampling_rate=100 --ucf101_fps=1

## dir등 각 파라미터는 수정 해야함

# 2. 학습
## LRCN 폴더로 이동

    python train.py --sampled_data_dir=/mnt/home/cmc/lrcn/CMC_sampled_data_video/ --ucf_list_dir=/mnt/home/cmc/lrcn/Data_CMC/cmcTrainTestlist/ --number_of_classes=2 --batch-size=16 --latent_dim=512 --hidden_size=256 --lstm_layers=2 --epoch=30 --val_check_interval=5

# 3. 테스트
## 폴더는 그대로LRCN

    python test.py  --model_dir=/mnt/home/cmc/lrcn/LRCN/20200212-155451/Saved_model_checkpoints/ --sampled_data_dir=/mnt/home/cmc/lrcn/CMC_sampled_data_video/ --ucf_list_dir=/mnt/home/cmc/lrcn/Data_CMC/cmcTrainTestlist/ --number_of_classes=2 --model_name='epoch_30.pth.tar'
