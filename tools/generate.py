
import sys
sys.path.append('./speech_processing_toolbox/')
import voicetool
import voicetool.base as voicebox
def generate(src, tgt, noisy, clean):
    sample_rate = 24000
    with open(src) as fid:
        with open(tgt, 'w') as wfid:
            for line in fid:
                name = line.strip().split('.wav')[0]
                print(noisy+name)
                data = voicebox.audioread(noisy+name+'.npy.wav')
                lenth = data.shape[0]
                wfid.writelines(noisy+name+'.npy.wav '+clean+name+'.wav {:.3f}'.format(lenth/sample_rate)+'\n')

noisy_pre='/home/disk2/summer_internship_2018/gaoyiqi/data/syang/waveglow/'
clean_pre='/home/disk2/summer_internship_2018/gaoyiqi/data/syang/original/'
generate('wav.lst', 'syang.lst', noisy_pre, clean_pre)

