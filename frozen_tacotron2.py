#-*-coding:utf-8-*-

import os,time
import numpy as np
import tensorflow as tf
from tacotron.utils.text import text_to_sequence
from hparams import hparams
from datasets import audio
from tacotron.models import create_model
import re
import symbols
syms = symbols.symbols

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# set tf_pyfunc = False when freeze model in tacotron/models/tacotron.py
 
class Synthesizer:
    def load(self, hparams, src_model_path=None,des_model_path=None):

        gta = False
        model_name = 'Tacotron'

        #Force the batch size to be known in order to use attention masking in batch synthesis
        inputs = tf.placeholder(tf.int32, (None, None), name='inputs')
        input_lengths = tf.placeholder(tf.int32, (None), name='input_lengths')
        targets = tf.placeholder(tf.float32, (None, None, hparams.num_mels), name='mel_targets')
        split_infos = tf.placeholder(tf.int32, shape=(hparams.tacotron_num_gpus, None), name='split_infos')
        with tf.variable_scope('Tacotron_model', reuse=tf.AUTO_REUSE) as scope:
            self.model = create_model(model_name, hparams)
            if gta:
                self.model.initialize(inputs, input_lengths, targets, gta=gta, split_infos=split_infos)
            else:
                self.model.initialize(inputs, input_lengths, split_infos=split_infos)

            self.mel_outputs = self.model.tower_mel_outputs
            self.linear_outputs = self.model.tower_linear_outputs if (hparams.predict_linear and not gta) else None
            self.alignments = self.model.tower_alignments
            self.stop_token_prediction = self.model.tower_stop_token_prediction
            self.targets = targets

        hparams.GL_on_GPU = False
        if hparams.GL_on_GPU:
            self.GLGPU_mel_inputs = tf.placeholder(tf.float32, (None, hparams.num_mels), name='GLGPU_mel_inputs')
            self.GLGPU_lin_inputs = tf.placeholder(tf.float32, (None, hparams.num_freq), name='GLGPU_lin_inputs')

            self.GLGPU_mel_outputs = audio.inv_mel_spectrogram_tensorflow(self.GLGPU_mel_inputs, hparams)
            self.GLGPU_lin_outputs = audio.inv_linear_spectrogram_tensorflow(self.GLGPU_lin_inputs, hparams)

        self.gta = gta
        self._hparams = hparams
        #pad input sequences with the <pad_token> 0 ( _ )
        self._pad = 0
        #explicitely setting the padding to a value that doesn't originally exist in the spectogram
        #to avoid any possible conflicts, without affecting the output range of the model too much
        if hparams.symmetric_mels:
            self._target_pad = -hparams.max_abs_value
        else:
            self._target_pad = 0.

        self.inputs = inputs
        self.input_lengths = input_lengths
        self.targets = targets
        self.split_infos = split_infos

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True

        session = tf.Session(config=config)
        session.run(tf.global_variables_initializer())

        saver = tf.train.Saver()
        saver.restore(session, src_model_path)
        # re-save model
        saver.save(session, des_model_path)


def ckpt2pb(ckpt_model,pb_model):
    saver = tf.train.import_meta_graph(ckpt_model + '.meta', clear_devices=True)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        saver.restore(sess, ckpt_model)

        # # 打印节点信息
        # tensor_name_list = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
        # for tensor_name in tensor_name_list:
        #     print(tensor_name)

        # mel_outputs   linear_outputs  alignments  stop_token_prediction
        output_graph_def = tf.graph_util.convert_variables_to_constants(
                                        sess,
                                        tf.get_default_graph().as_graph_def(),
                                        ["Tacotron_model/inference/Minimum_1",
                                         # "Tacotron_model/inference/Minimum_2",
                                         # "Tacotron_model/inference/transpose",
                                         "Tacotron_model/inference/Reshape_2",
                                         ])

        with tf.gfile.GFile(pb_model, "wb") as f:
            f.write(output_graph_def.SerializeToString())

    return


def prepare_inputs(inputs):
    max_len = max([len(x) for x in inputs])
    return np.stack([pad_input(x, max_len) for x in inputs]), max_len

def pad_input(x, length):
    return np.pad(x, (0, length - x.shape[0]), mode='constant', constant_values=0)



def inference_pb(model_pb):
    texts = 'k a2 er2 p u3 #2 p ei2 uai4 s uen1 #1 uan2 h ua2 t i1 #4  。'
    # texts = 'b ao2 m a3 #1 p ei4 g ua4 #1 b o3 l uo2 an1 #3  ， d iao1 ch an2 #1 van4 zh en3 #2 d ong3 ueng1 t a4 #4  。'
    s = []
    texts_split = re.split("( )", texts)
    for i in texts_split:
        if (i in syms):
            index = syms.index(i)
            s.append(index)
    seqs = np.asarray(s)

    seqs_lengths = len(seqs)
    input_lengths_np = np.asarray(seqs_lengths, dtype=np.int32).reshape(1)

    input_seqs = seqs[np.newaxis].astype(np.int32)
    max_seq_len = seqs_lengths
    split_infos_np = np.asarray([max_seq_len, 0, 0, 0], dtype=np.int32)[np.newaxis]
    print('input_seqs:', input_seqs.shape)
    print('input_lengths_np:', input_lengths_np.shape)
    print('split_infos_np:', split_infos_np.shape)

    #############################
    # texts = ['k a2 er2 p u3 #2 p ei2 uai4 s uen1 #1 uan2 h ua2 t i1 #4  。']
    # t2_hparams = hparams.parse('')
    # cleaner_names = [x.strip() for x in t2_hparams.cleaners.split(',')]
    # seqs = [np.asarray(text_to_sequence(text, cleaner_names)) for text in texts]
    # input_lengths_np = [len(seq) for seq in seqs]
    # input_lengths_np = np.asarray(input_lengths_np, dtype=np.int32)
    #
    # size_per_device = len(seqs) // t2_hparams.tacotron_num_gpus
    #
    # # Pad inputs according to each GPU max length
    # input_seqs = None
    # split_infos_np = []
    # for i in range(t2_hparams.tacotron_num_gpus):
    #     device_input = seqs[size_per_device * i: size_per_device * (i + 1)]
    #     device_input, max_seq_len = prepare_inputs(device_input)
    #     input_seqs = np.concatenate((input_seqs, device_input), axis=1) if input_seqs is not None else device_input
    #     input_seqs = input_seqs.astype(np.int32)
    #     split_infos_np.append([max_seq_len, 0, 0, 0])
    # split_infos_np = np.asarray(split_infos_np, dtype=np.int32)
    # print('input_seqs:', input_seqs.shape)
    # print('input_lengths_np:', input_lengths_np.shape)
    # print('split_infos_np:', split_infos_np.shape)

    #################
    with tf.gfile.FastGFile(model_pb, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')

    with tf.Session() as sess:
        in_tensor = sess.graph.get_tensor_by_name('inputs:0')
        in_length_tensor = sess.graph.get_tensor_by_name('input_lengths:0')
        split_infos_tensor = sess.graph.get_tensor_by_name('split_infos:0')

        mel_output_tensor = sess.graph.get_tensor_by_name('Tacotron_model/inference/Minimum_1:0')
        # linear_output_tensor = sess.graph.get_tensor_by_name('Tacotron_model/inference/Minimum_2:0')
        # alignments_output_tensor = sess.graph.get_tensor_by_name('Tacotron_model/inference/transpose:0')
        stop_token_output_tensor = sess.graph.get_tensor_by_name('Tacotron_model/inference/Reshape_2:0')

        feed_dict = {in_tensor: input_seqs,
                     in_length_tensor: input_lengths_np,
                     split_infos_tensor: split_infos_np}

        mel_out,stop_token_output = sess.run([mel_output_tensor,stop_token_output_tensor], feed_dict=feed_dict)

        # postprocess
        mel_out = np.squeeze(mel_out, 0)
        target_length = 0
        stop_tokens_list = np.round(stop_token_output).tolist()
        for row in stop_tokens_list:
            if 1 in row:
                target_length = row.index(1)
            else:
                target_length = len(row)

        # Take off the batch wise padding
        mel_out = mel_out[:target_length, :]

        mel_out = np.clip(mel_out, -4, 4)
        print(mel_out.shape)
        np.save('mel_out.npy',mel_out)




if __name__ == '__main__':

    src_model_path = './logs-Tacotron-2_phone/taco_pretrained/tacotron_model.ckpt-250000'
    des_model_path = './logs-Tacotron-2_phone/taco_pretrained1/new_model'

    pb_file = 'tacotron2.pb'

    # 1.re-save model
    # t2_hparams = hparams.parse('')
    # synth = Synthesizer()
    # synth.load(t2_hparams,src_model_path,des_model_path)

    # 2.frozen_model
    # ckpt2pb(des_model_path, pb_file)

    # 3.test
    # t1 = time.time()
    # inference_pb(pb_file)
    # print('time:',time.time() - t1)
    print('done')

