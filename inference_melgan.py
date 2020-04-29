#-*-coding:utf-8-*-
import glob
import tqdm
import torch
import argparse
from scipy.io.wavfile import write
import numpy as np
from melgan_vocoder.model.generator import Generator
from melgan_vocoder.utils.hparams import HParam, load_hparam_str
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

MAX_WAV_VALUE = 32768.0

def main(args):
    checkpoint = torch.load(args.checkpoint_path)
    if args.config is not None:
        hp = HParam(args.config)
    else:
        hp = load_hparam_str(checkpoint['hp_str'])

    model = Generator(hp.audio.n_mel_channels).cuda()
    model.load_state_dict(checkpoint['model_g'])
    model.eval(inference=False)

    # torch.save(model, 'genertor1.pt')  # 保存和加载整个模型
    # torch.save(model.state_dict(), 'genertor2.pt')   # 仅保存和加载模型参数(推荐使用)

    num = 0
    with torch.no_grad():
        for melpath in tqdm.tqdm(glob.glob(os.path.join(args.input_folder, '*.npy'))):

            t2_mel = np.load(melpath)
            t2_mel = np.transpose(t2_mel,[1,0])
            t2_mel = t2_mel[np.newaxis, :]
            mel = torch.from_numpy(t2_mel)
            mel = mel.cuda()
            mel_np = mel.cpu().numpy()

            audio = model.inference(mel)

            audio = audio.cpu().detach().numpy()

            out_path = args.save_path + str(num) + ('_reconstructed_epoch%04d.wav' % checkpoint['epoch'])
            write(out_path, hp.audio.sampling_rate, audio)
            num += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default=None,
                        help="yaml file for config. will use hp_str from checkpoint if not given.")
    parser.add_argument('-p', '--checkpoint_path', type=str, default='./melgan_vocoder/chkpt/biaobei/biaobei_aca5990_3125.pt',
                        help="path of checkpoint pt file for evaluation")
    parser.add_argument('-i', '--input_folder', type=str, default= './melgan_vocoder/data/test/mel/' ,
                        help="directory of mel-spectrograms to invert into raw audio. ")
    parser.add_argument('-s', '--save_path', type=str, default='./melgan_vocoder/data/test/wav/')
    args = parser.parse_args()

    main(args)
