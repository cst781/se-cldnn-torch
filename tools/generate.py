
import sys
import soundfile as sf
def add_duration(src, tgt, noisy, clean):
    with open(src) as fid:
        with open(tgt, 'w') as wfid:
            for line in fid:
                name = line.strip().split('.wav')[0]
                data, sample_rate = sf.read(noisy+name+'.npy.wav')
                lenth = data.shape[0]
                wfid.writelines(noisy+name+'.npy.wav '+clean+name+'.wav {:.3f}'.format(lenth/sample_rate)+'\n')


