#-*-coding:utf-8-*-

import onnx     # 'import onnx' must before 'import torch'
import torch
from torch.autograd import Variable
import onnxruntime
import numpy as np
from scipy.io.wavfile import write
from model.generator import Generator

# use MyRefPad1d() instead nn.ReflectionPad1d() in generator.py and res_stack.py
# tensorrt not support nn.ReflectionPad1d()

# python -m onnxsim melgan_dynamic.onnx melgan_new_dynamic.onnx --input-shape "1,80,100"

MAX_WAV_VALUE = 32768.0
n_mel_channels = 80
hop_length = 256
sampling_rate = 22050
test_time_step = 100

export_onnx_file = 'melgan_dynamic.onnx'
mel_file = './data/test/t2_mel/mel-batch_11_sentence_0.npy'
checkpoint_path = './chkpt/biaobei/biaobei_aca5990_3125.pt'


def torch2onnx(export_onnx_file):

    input_name = ['input']
    output_name = ['output']

    input = Variable(torch.randn(1, 80, test_time_step))

    checkpoint = torch.load(checkpoint_path)
    model = Generator(n_mel_channels)
    model.load_state_dict(checkpoint['model_g'])
    model.eval(inference=True)   # onnx-inference must true

    torch.onnx.export(model, input, export_onnx_file,
                      input_names=input_name,
                      output_names=output_name,
                      verbose=True,
                      opset_version=10,
                      export_params=True,
                      keep_initializers_as_inputs=True,
                      dynamic_axes={"input": {2: "time_step"},
                                    "output": {2: "time_step"}}
                      )

    test = onnx.load(export_onnx_file)
    onnx.checker.check_model(test)

    print("Producer Name:", test.producer_name)
    print("Producer Version:", test.producer_version)
    print("Opset", test.opset_import[0])

    print("==> Passed")

#---------------------------


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def onnxruntime_infer(mel_file,export_onnx_file):

    t2_mel = np.load(mel_file)
    t2_mel = np.transpose(t2_mel, [1, 0])
    t2_mel = t2_mel[np.newaxis, :]
    mel = torch.from_numpy(t2_mel)
    zero = torch.full((1, 80, 10), -4)
    mel = torch.cat((mel, zero), dim=2)
    mel_np = mel.cpu().numpy()
    # mel_np = mel.cpu().numpy()[:, :, :test_time_step]


    ort_session = onnxruntime.InferenceSession(export_onnx_file)

    for input_meta in ort_session.get_inputs():
        print(input_meta)
    for output_meta in ort_session.get_outputs():
        print(output_meta)

    ort_inputs = {ort_session.get_inputs()[0].name: mel_np}
    ort_outs = ort_session.run(["output"], ort_inputs)

    audio = ort_outs[0].squeeze()
    audio = torch.from_numpy(audio)
    audio = audio[:-(hop_length * 10)]
    audio = MAX_WAV_VALUE * audio
    audio = audio.clamp(min=-MAX_WAV_VALUE, max=MAX_WAV_VALUE - 1)
    audio = audio.short()

    audio = audio.cpu().detach().numpy()
    out_path = 'save.wav'
    write(out_path, sampling_rate, audio)

    print('done')

if __name__ == '__main__':

    torch2onnx(export_onnx_file)
    onnxruntime_infer(mel_file , export_onnx_file)


