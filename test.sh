#!/bin/bash 
input_dim=257
output_dim=257
left_context=1
right_context=1
lr=0.001
win_len=400
win_inc=100
fft_len=512
sample_rate=16k
win_type=hamming
batch_size=8
max_epoch=45
rnn_units=512
rnn_layers=2
tt_list=t
kernel_size=6
kernel_num=9
dropout=0.2
retrain=1
sample_rate=16k
num_gpu=2
batch_size=$[num_gpu*batch_size]
target_mode=MSA


save_name=sruc_-5~20_${target_mode}_${left_context}_${right_context}_${lr}_${sample_rate}_${kernel_size}_${kernel_num}_${win_len}_${win_inc}
exp_dir=exp/${save_name}
if [ ! -d ${exp_dir} ] ; then
    mkdir -p ${exp_dir}
fi
    CUDA_VISIBLE_DEVICES='7' python -u ./steps/run_sruc.py \
    --decode=1 \
    --fft-len=${fft_len} \
    --input-dim=${input_dim} \
    --output-dim=${output_dim} \
    --window-len=${win_len} \
    --window-inc=${win_inc} \
    --exp-dir=${exp_dir} \
    --tr-list=${tr_list} \
    --cv-list=${cv_list} \
    --tt-list=${tt_list} \
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
    --kernel-num=${kernel_num} \
    --sample-rate=${sample_rate} \
    --target-mode=${target_mode} \
    --window-type=${win_type}

