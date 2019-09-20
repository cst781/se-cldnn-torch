#!/bin/bash


                        #--noise_wav_list ../../../data/musan/${mode}.lst \

for mode in train dev; do 
    tgt_clean_dir=../data/wavs/${mode}_-5~20_clean
    tgt_noisy_dir=../data/wavs/${mode}_-5~20_noisy
    
    python add_noise.py --clean_wav_list ../../../data/data_aishell/fixed_${mode}.lst \
                        --noise_wav_list ../../../data/musan/train.lst \
                        --mix_list ../data/mix_list/${mode}_-5~20_mix.lst \
                        --snr_lower -5 \
                        --snr_upper 20 \
                        --generate_mix_wav 1\
                        --output_clean_dir ${tgt_clean_dir}\
                        --output_noisy_dir ${tgt_noisy_dir}
    
    find `pwd`/${tgt_clean_dir} -iname "*.wav" |sort> /tmp/clean 
    find `pwd`/${tgt_noisy_dir} -iname "*.wav"|sort > /tmp/noisy
    paste -d ' ' /tmp/noisy /tmp/clean > ../data/${mode}_aishell1_-5~20.lst 
done 

