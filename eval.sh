
#!/bin/bash 
input_dim=257
output_dim=257
left_context=1
right_context=1
lr=0.0008
win_len=400
win_inc=100
fft_len=512
sample_rate=16k
win_type=hamming
batch_size=8
max_epoch=45
rnn_units=512
rnn_layers=2
tr_clean_list=data/train_clean.lst
tr_noise_list=data/noise.lst

tr_clean_list=data/train_clean.lst
tr_noise_list=data/noise.lst
cv_clean_list=data/dev_clean.lst
cv_noise_list=data/noise.lst

tt_list=data/test_hw.lst

kernel_size=6
kernel_num=9
dropout=0.2
retrain=1
sample_rate=16k
num_gpu=1
batch_size=$[num_gpu*batch_size]

target_mode=MSA
exp_dir=exp/sruc_-5~20_MSA_1_1_0.001_16k_6_9_400_100/
dataset_name=aishell1

for snr in -5 0 5 10 15 20 ; do
    tgt=sruc_${target_mode}_${dataset_name}_${snr}db
        
    #clean_wav_path=/search/odin/huyanxin/workspace/se-cldnn-torch/data/wavs/test_${dataset_name}_clean_${snr}/
    #noisy_wav_path=/search/odin/huyanxin/workspace/se-cldnn-torch/data/wavs/test_${dataset_name}_noisy_${snr}/
    clean_wav_path=/search/odin/huyanxin/workspace/se-cldnn-torch/data/wavs/test_clean_${snr}/
    noisy_wav_path=/search/odin/huyanxin/workspace/se-cldnn-torch/data/wavs/test_noisy_${snr}/
    enh_wav_path=${exp_dir}/test_${dataset_name}_noisy_${snr}/
    find ${noisy_wav_path} -iname "*.wav" > wav.lst
    CUDA_VISIBLE_DEVICES='1' python -u ./steps/run_sruc.py \
    --decode=1 \
    --fft-len=${fft_len} \
    --input-dim=${input_dim} \
    --output-dim=${output_dim} \
   --window-len=${win_len} \
    --window-inc=${win_inc} \
    --exp-dir=${exp_dir} \
    --tr-noise=${tr_noise_list} \
    --tr-clean=${tr_clean_list} \
    --cv-noise=${cv_noise_list} \
    --cv-clean=${cv_clean_list} \
    --tt-list=wav.lst \
    --retrain=${retrain} \
    --rnn-layers=${rnn_layers} \
    --rnn-units=${rnn_units} \
    --learn-rate=${lr} \
    --max-epoch=${max_epoch} \
    --dropout=${dropout} \
    --input-dim=${input_dim} \
    --output-dim=${output_dim} \
    --left-context=${left_context} \
    --right-context=${right_context} \
    --batch-size=${batch_size} \
    --kernel-size=${kernel_size} \
    --target-mode=${target_mode} \
    --kernel-num=${kernel_num} \
    --sample-rate=${sample_rate} \
    --window-type=${win_type}  || exit 1 # > ${exp_dir}/train.log &
    mv ${exp_dir}/rec_wav ${enh_wav_path}
    
    ls $noisy_wav_path > t
    python ./tools/eval_objective.py --wav_list=t --result_list=${tgt}.csv --pathe=${enh_wav_path}\
    --pathc=${clean_wav_path} --pathn=${noisy_wav_path} ||exit 1
done
