#-*-coding:utf-8-*-

import argparse
import os
from tqdm import tqdm
import numpy as np

import tensorflow as tf
from hparams import hparams, hparams_debug_string
from infolog import log
from tacotron.synthesizer import Synthesizer

#
import torch
from scipy.io.wavfile import write
from melgan_vocoder.model.generator import Generator
from melgan_vocoder.utils.hparams import HParam, load_hparam_str

import warnings
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

MAX_WAV_VALUE = 32768.0


def get_sentences(args):
	if args.text_list != '':
		with open(args.text_list, 'rb') as f:
			sentences = list(map(lambda l: l.decode("utf-8")[:-1], f.readlines()))
	else:
		sentences = hparams.sentences
	return sentences

def init_tacotron2(args):
	# t2
	print('\n#####################################')
	if args.model == 'Tacotron':
		print('\nInitialising Tacotron Model...\n')
		t2_hparams = hparams.parse(args.hparams)
		try:
			checkpoint_path = tf.train.get_checkpoint_state(args.taco_checkpoint).model_checkpoint_path
			log('loaded model at {}'.format(checkpoint_path))
		except:
			raise RuntimeError('Failed to load checkpoint at {}'.format(args.taco_checkpoint))

		output_dir = 'tacotron_' + args.output_dir
		eval_dir = os.path.join(output_dir, 'eval')
		log_dir = os.path.join(output_dir, 'logs-eval')
		print('eval_dir:', eval_dir)
		print('args.mels_dir:', args.mels_dir)

		# Create output path if it doesn't exist
		os.makedirs(eval_dir, exist_ok=True)
		os.makedirs(log_dir, exist_ok=True)
		os.makedirs(os.path.join(log_dir, 'wavs'), exist_ok=True)
		os.makedirs(os.path.join(log_dir, 'plots'), exist_ok=True)
		log(hparams_debug_string())
		synth = Synthesizer()
		synth.load(checkpoint_path, t2_hparams)

	return synth,eval_dir,log_dir

def init_melgan(args):
	# melgan
	print('\n#####################################')
	checkpoint = torch.load(args.vocoder_checkpoint)
	if args.vocoder_config is not None:
		hp = HParam(args.config)
	else:
		hp = load_hparam_str(checkpoint['hp_str'])

	melgan_model = Generator(hp.audio.n_mel_channels).cuda()
	melgan_model.load_state_dict(checkpoint['model_g'])
	melgan_model.eval(inference=False)

	# torch.save(model, 'genertor1.pt')  # 保存和加载整个模型
	# torch.save(model.state_dict(), 'genertor2.pt')   # 仅保存和加载模型参数(推荐使用)

	return melgan_model,hp,checkpoint


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--taco_checkpoint',
			default='./logs-Tacotron-2_phone/taco_pretrained/',help='Path to model checkpoint')
	parser.add_argument('--hparams', default='',
		help='Hyperparameter overrides as a comma-separated list of name=value pairs')
	parser.add_argument('--model', default='Tacotron')
	parser.add_argument('--mels_dir', default='tacotron_output/eval/', help='folder to contain mels to synthesize audio from using the Wavenet')
	parser.add_argument('--output_dir', default='output/', help='folder to contain synthesized mel spectrograms')
	parser.add_argument('--text_list', default='sentences_phone.txt', help='Text file contains list of texts to be synthesized. Valid if mode=eval')
	parser.add_argument('--speaker_id', default=None, help='Defines the speakers ids to use when running standalone Wavenet on a folder of mels. this variable must be a comma-separated list of ids')

	# melgan
	parser.add_argument('--vocoder_config', type=str, default=None,
						help="yaml file for config. will use hp_str from checkpoint if not given.")
	parser.add_argument('--vocoder_checkpoint', type=str, default='./melgan_vocoder/chkpt/biaobei/biaobei_aca5990_3125.pt',
						help="path of checkpoint pt file for evaluation")

	args = parser.parse_args()
	sentences = get_sentences(args)

	############################
	synth, eval_dir, log_dir = init_tacotron2(args)

	voc_model,voc_hp,voc_checkpoint = init_melgan(args)
	output_melgan_dir = 'tacotron_' + args.output_dir + 'melgan/'
	os.makedirs(output_melgan_dir, exist_ok=True)

	# ###################################
	# Set inputs batch wise
	sentences = [sentences[i: i + hparams.tacotron_synthesis_batch_size] for i in
				 range(0, len(sentences), hparams.tacotron_synthesis_batch_size)]

	log('Starting Synthesis')
	with open(os.path.join(eval_dir, 'map.txt'), 'w') as file:
		for i, texts in enumerate(tqdm(sentences)):
			print('\nsynthesis mel:' + str(i))
			basenames = ['batch_{}_sentence_{}'.format(i, j) for j in range(len(texts))]
			mel_filenames, speaker_ids = synth.synthesize(texts, basenames, eval_dir, log_dir, None)
			for elems in zip(texts, mel_filenames, speaker_ids):
				file.write('|'.join([str(x) for x in elems]) + '\n')
			print('\nsynthesis mel done')

			# melgan
			with torch.no_grad():
				mel_filenames = mel_filenames[0]
				t2_mel = np.load(mel_filenames)
				t2_mel = np.transpose(t2_mel, [1, 0])
				t2_mel = t2_mel[np.newaxis, :]
				mel = torch.from_numpy(t2_mel)
				mel = mel.cuda()
				mel_np = mel.cpu().numpy()

				audio = voc_model.inference(mel)

				audio = audio.cpu().detach().numpy()

				out_path = output_melgan_dir + str(i) + ('_melgan_epoch%04d.wav' % voc_checkpoint['epoch'])
				write(out_path, voc_hp.audio.sampling_rate, audio)

				print('\nmelgan done')
				print('#####################\n')

	log('\nsynthesized done at {}'.format(output_melgan_dir))


if __name__ == '__main__':
	main()
