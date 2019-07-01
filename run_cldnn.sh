#!/bin/bash 
input_dim=257
output_dim=257
left_context=1
right_context=1
lr=0.0005
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
tr_list=data/train_2018.lst
cv_list=data/dev_2018.lst
tt_list=data/test_hw.lst
tt_list=data/test_2018.lst
kernel_size=6
kernel_num=9
dropout=0.2
retrain=0
sample_rate=16k
num_gpu=1
batch_size=$[num_gpu*batch_size]


save_name=cldnn_2_${left_context}_${right_context}_${lr}_${sample_rate}_${kernel_size}_${kernel_num}
exp_dir=exp/${save_name} #ldnn_${left_context}_${right_context}_${lr}_${sample_rate}_${kernel_size}_${kernel_num}/
if [ ! -d ${exp_dir} ] ; then
    mkdir -p ${exp_dir}
fi
#/home/work_nfs/common/tools/pyqueue_asr.pl \
#    -l hostname="!node7" -q g.q --gpu 1 --num-threads ${num_gpu} \
#        exp/${save_name}/${save_name}.log \
    CUDA_VISIBLE_DEVICES='1' nohup python -u ./steps/run_cldnn.py \
exp_dir=dilation_2/no_sampler_cldnn_1_1_0.005_16k  # cldnn_${left_context}_${right_context}_${lr}_${sample_rate}/
save_name=gruc_${left_context}_${right_context}_${lr}_${sample_rate}_${kernel_size}_${kernel_num}/
exp_dir=exp/${save_name}
if [ ! -d ${exp_dir} ] ; then
    mkdir -p ${exp_dir}
fi
CUDA_VISIBLE_DEVICES='0' nohup python ./steps/run_cldnn.py \
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
    --window-type=${win_type} > ${exp_dir}/train.log &

