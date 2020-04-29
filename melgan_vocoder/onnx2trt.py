#-*-coding:utf-8-*-

import onnx
import pycuda.autoinit
import numpy as np
import pycuda.driver as cuda
import tensorrt as trt
import torch
import os
import time

test_time_step = 100
MAX_WAV_VALUE = 32768.0
n_mel_channels = 80
hop_length = 256
sampling_rate = 22050
min_dynamic_shape = (1,n_mel_channels,10)
opt_dynamic_shape = (1,n_mel_channels,1000)
max_dynamic_shape = (1,n_mel_channels,2000)

TRT_LOGGER = trt.Logger(trt.Logger.INFO)  # This logger is required to build an engine

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        """Within this context, host_mom means the cpu memory and device means the GPU memory
        """
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


def allocate_buffers(engine,melgan_time_step):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        # size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        if(binding == 'input'):
            size = 1 * n_mel_channels * melgan_time_step
        if (binding == 'output'):
            size = hop_length * melgan_time_step
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream


def get_engine(max_batch_size=1, onnx_file_path="", engine_file_path="", \
               fp16_mode=False, int8_mode=False, save_engine=False,):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""

    def build_engine(max_batch_size, save_engine):
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""

        explicit_batch = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        with trt.Builder(TRT_LOGGER) as builder, \
                builder.create_network(explicit_batch) as network, \
                trt.OnnxParser(network, TRT_LOGGER) as parser:

            builder.max_workspace_size = 1 << 30  # Your workspace size
            builder.max_batch_size = max_batch_size
            builder.fp16_mode = fp16_mode  # Default: False
            builder.int8_mode = int8_mode  # Default: False

            if int8_mode:
                # To be updated
                raise NotImplementedError

            # Parse model file
            if not os.path.exists(onnx_file_path):
                quit('ONNX file {} not found'.format(onnx_file_path))

            print('Loading ONNX file from path {}...'.format(onnx_file_path))
            with open(onnx_file_path, 'rb') as model:
                print('Beginning ONNX file parsing')
                parser_flag = parser.parse(model.read())
                print(parser_flag)

                if not parser_flag:
                    print('ERROR: Failed to parse the ONNX file.')
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))
                    return None

            last_layer = network.get_layer(network.num_layers - 1)
            # Check if last layer recognizes it's output
            if not last_layer.get_output(0):
                # If not, then mark the output using TensorRT API
                network.mark_output(last_layer.get_output(0))

            print('Completed parsing of ONNX file')
            print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))

            # dynamic shape
            profile = builder.create_optimization_profile()
            input_name = network.get_input(0).name
            profile.set_shape(input_name, min=min_dynamic_shape, opt=opt_dynamic_shape, max=max_dynamic_shape)
            config = builder.create_builder_config()
            config.max_workspace_size = 2 ** 30    # 1GiB
            config.add_optimization_profile(profile)

            engine = builder.build_engine(network,config)

            # fixed shape
            # engine = builder.build_cuda_engine(network)

            print("Completed creating Engine")

            if save_engine:
                with open(engine_file_path, "wb") as f:
                    f.write(engine.serialize())
            return engine

    if os.path.exists(engine_file_path):
        # If a serialized engine exists, load it instead of building a new one.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine(max_batch_size, save_engine)


def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    # Transfer data from CPU to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]


if __name__ == '__main__':

    onnx_model_path = 'melgan_dynamic.onnx'
    save_file = 'save_trt.wav'
    mel_file = './data/test/t2_mel/mel-batch_11_sentence_0.npy'

    #preprocess
    t2_mel = np.load(mel_file)
    t2_mel = np.transpose(t2_mel, [1, 0])
    t2_mel = t2_mel[np.newaxis, :]
    mel = torch.from_numpy(t2_mel)
    zero = torch.full((1, 80, 10), -4)
    mel = torch.cat((mel, zero), dim=2)
    mel_np = mel.cpu().numpy()
    real_time_step = mel_np.shape[2]
    # mel_np = mel.cpu().numpy()[:, :, :test_time_step]

    # These two modes are dependent on hardwares
    fp16_mode = False
    int8_mode = False
    save_engine = True
    trt_engine_path = 'melgan_fp16_{}_int8_{}.trt'.format(fp16_mode, int8_mode)
    # Build an engine
    engine = get_engine(1, onnx_model_path, trt_engine_path, fp16_mode, int8_mode, save_engine)
    # Create the context for this engine
    context = engine.create_execution_context()
    context.set_binding_shape(0, (1, n_mel_channels, real_time_step))  # important

    # Allocate buffers for input and output
    inputs, outputs, bindings, stream = allocate_buffers(engine,real_time_step) # input, output: host # bindings

    # Do inference
    shape_of_output = (1, hop_length * real_time_step)
    # Load data to the buffer
    inputs[0].host = np.array(mel_np.reshape(-1))
    # inputs[1].host = ... for multiple input

    t1 = time.time()
    trt_outputs = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream) # numpy data
    t2 = time.time()
    print('time:',t2-t1)

    # postprocess
    audio = trt_outputs[0].squeeze()
    audio = torch.from_numpy(audio)
    audio = audio[:-(hop_length * 10)]
    audio = MAX_WAV_VALUE * audio
    audio = audio.clamp(min=-MAX_WAV_VALUE, max=MAX_WAV_VALUE - 1)
    audio = audio.short()
    audio = audio.cpu().detach().numpy()

    from scipy.io.wavfile import write
    write(save_file, sampling_rate, audio)

    print('All completed!')

