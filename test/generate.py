
import sys
sys.path.append('./speech_processing_toolbox/')
import voicetool
import voicetool.base as voicebox
import dataset 
def generate(src, tgt, pre):
    sample_rate = 16000
    with open(src) as fid:
        with open(tgt, 'w') as wfid:
            for line in fid:
                path = line.strip()
                name = path.split('/')[-1]
                data = voicebox.audioread(path)
                lenth = data.shape[0]
                wfid.writelines(pre+name+' '+path+' '+str(lenth/sample_rate)+'\n')
pre = '/home/disk2/SE_data/noisy_for_reconst'
generate('./clean_wav.lst', './all.scp', pre)


