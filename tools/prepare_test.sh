#!/bin/bash

            #--clean_wav_list ../../../data/data_aishell/fixed_test.lst \

dataset_name=librispeech
python add_noise.py \
            --clean_wav_list ../../../data/librispeech/LibriSpeech/test_clean_wav.lst \
            --noise_wav_list /search/speech/huyanxin/data/TUT-DECASE-2016/all.lst \
            --mix_list ../data/mix_list/test_${dataset_name}_mix_-5.lst \
            --snr_lower -5 \
            --snr_upper -5

for snr in -5 0 5 10 15 20; do 
    cp ../data/mix_list/test_${dataset_name}_mix_-5.lst /tmp/t
    sed "s/-5/${snr}/g" /tmp/t > ../data/mix_list/test_${dataset_name}_mix_${snr}.lst
    python add_noise.py  --mix_list ../data/mix_list/test_${dataset_name}_mix_${snr}.lst \
                            --generate_mix_wav 1 \
                            --output_clean_dir ../data/wavs/test_${dataset_name}_clean_${snr} \
                            --output_noisy_dir ../data/wavs/test_${dataset_name}_noisy_${snr}  ||exit 1

    find `pwd`/../data/wavs/test_${dataset_name}_clean_${snr}/ -iname "*.wav" |sort> /tmp/clean 
    find `pwd`/../data/wavs/test_${dataset_name}_noisy_${snr}/ -iname "*.wav" |sort> /tmp/noisy
    paste -d ' ' /tmp/noisy /tmp/clean > ../data/test_${dataset_name}_${snr}.lst 

done 


