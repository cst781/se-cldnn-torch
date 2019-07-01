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
tt_list=data/tt.lst
tr_list=data/tr.lst
cv_list=data/cv.lst
tr_list=data/tr_2019_500.lst
cv_list=data/cv_2019.lst
cv_list=data/cv_1k6.lst
tr_list=data/tr_1k6.lst

tt_list=data/test_hw.lst
tt_list=data/test_2018.lst

kernel_size=6
kernel_num=9
dropout=0.2
retrain=1
sample_rate=16k
num_gpu=2
batch_size=$[num_gpu*batch_size]
exp_dir=dilation_2/no_sampler_cldnn_1_1_0.005_16k  # cldnn_${left_context}_${right_context}_${lr}_${sample_rate}/

exp_dir=exp/sruc_bigdata_1k6h_cldnn_${left_context}_${right_context}_${lr}_${sample_rate}_${kernel_size}_${kernel_num}/
if [ ! -d ${exp_dir} ] ; then
    mkdir ${exp_dir}
fi
    #-l hostname="!node7" -q g.q --gpu 1 --num-threads ${num_gpu} \
/home/work_nfs/common/tools/pyqueue_asr.pl \
    -q g.q --gpu 1 --num-threads ${num_gpu} \
    ${exp_dir}/bigdata_1k6h_cldnn.log \
    python -u ./steps/run_big_data.py \
    --decode=0 \
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
    --num-gpu=${num_gpu} \
    --window-type=${win_type} &

