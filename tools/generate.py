
import sys
sys.path.append('./speech_processing_toolbox/')
import voicetool
import voicetool.base as voicebox
def generate(src, tgt, noisy, clean):
    sample_rate = 16000
    with open(src) as fid:
        with open(tgt, 'w') as wfid:
            for line in fid:
                name = line.strip()
                print(noisy+name)
                data = voicebox.audioread(noisy+name)
                lenth = data.shape[0]
                wfid.writelines(noisy+name+' '+clean+name+' {:.3f}'.format(lenth/sample_rate)+'\n')

noisy_pre='/home/yxhu/work_nfs/data/tfrecords_station_noise_-5_5/addnoise_2w_chosen_station_noise_-5_5_noisy/'
clean_pre='/home/yxhu/work_nfs2/data/clean_wav_2w/'
#generate('../data/train', './train_2018.lst', noisy_pre,clean_pre)
generate('../data/dev', './dev_2018.lst', noisy_pre,clean_pre)


