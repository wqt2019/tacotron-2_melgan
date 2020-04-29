#-*- coding: utf-8 -*-

from __future__ import print_function

import grpc,time
import tensorflow as tf
import numpy as np
from numpy.core.multiarray import ndarray
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

from tensorflow.core.framework import tensor_pb2
from tensorflow.core.framework import tensor_shape_pb2
from tensorflow.core.framework import types_pb2

from tacotron.utils.text import text_to_sequence
from hparams import hparams

import re
import symbols

# tensorflow_model_server --port=9001 --model_name=tacotron2 --model_base_path=./tacotron-2_melgan/save_model/

# sudo docker run -p 9002:8501 -p 9001:8500 -e CUDA_VISIBLE_DEVICES=0 --mount type=bind,source=./tacotron-2_melgan/save_model/,
# target=/models/tacotron2 -e MODEL_NAME=tacotron2 -t tensorflow/serving:1.13.0-gpu --per_process_gpu_memory_fraction=0.5

syms = symbols.symbols

MAX_MESSAGE_LENGTH=-1

def prepare_inputs(inputs):
    max_len = max([len(x) for x in inputs])
    return np.stack([pad_input(x, max_len) for x in inputs]), max_len

def pad_input(x, length):
    return np.pad(x, (0, length - x.shape[0]), mode='constant', constant_values=0)

def predict_tts():

    texts = 'k a2 er2 p u3 #2 p ei2 uai4 s uen1 #1 uan2 h ua2 t i1 #4  。'
    # texts = 'b ao2 m a3 #1 p ei4 g ua4 #1 b o3 l uo2 an1 #3  ， d iao1 ch an2 #1 van4 zh en3 #2 d ong3 ueng1 t a4 #4  。'
    s = []
    texts_split = re.split("( )", texts)
    for i in texts_split:
        if(i in syms):
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


    #-----------tacotron2------------
    hostport = 'localhost:9001'

    channel = grpc.insecure_channel(hostport, options=
                                [('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
                                ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH)])

    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'tacotron2'
    request.model_spec.signature_name = 'predict'

    # tensor_proto = tensor_pb2.TensorProto(dtype=types_pb2.DT_STRING,string_val=[img_str])
    # request.inputs['images'].CopyFrom(tensor_proto)

    request.inputs['inputs'].CopyFrom(tf.contrib.util.make_tensor_proto(input_seqs))
    request.inputs['input_lengths'].CopyFrom(tf.contrib.util.make_tensor_proto(input_lengths_np))
    request.inputs['split_infos'].CopyFrom(tf.contrib.util.make_tensor_proto(split_infos_np))

    t1 = time.time()
    result_future = stub.Predict.future(request)
    print('time:',time.time() - t1)

    mel_out = result_future.result().outputs['mel']
    mel_out_list = (tf.contrib.util.make_ndarray(mel_out).tolist())
    mel_out_np = np.array(mel_out_list)  # type: ndarray

    # mel_out_np = np.squeeze(mel_out_np, 0)
    mel_out_np = mel_out_np.astype(np.float32)
    print(mel_out_np.shape)
    np.save('mel_out1.npy', mel_out_np)

    return

if __name__ == '__main__':

    # t = time.time()
    result = predict_tts()
    # print('time:',time.time() - t)

    print('done...')

