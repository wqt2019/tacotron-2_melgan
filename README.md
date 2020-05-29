# tacotron-2(tensorflow) + melgan(pytorch) chinese TTS:  
  
  
[melgan](https://github.com/seungwonpark/melgan) is very faster than other vocoders and the quality is not so bad. re-implement the [split_func](./tacotron/models/tacotron.py) in tacotron2 that tensorflow serving not support , re-implement the [nn.ReflectionPad1d](./melgan_vocoder/model/res_stack.py) that tensorrt not support. modify the 
melgan's input from [-12,2] to [-4,4] that match the tacotron2's output.   
  
python37，biaobei chinese dataset，tacotron2 support chinese pinyin or chinese phone + rhythm training(default is phone + rhythm)，edit [symbols.py](./tacotron/utils/symbols.py) and [text.py](./tacotron/utils/text.py)：
  
pinyin：  
	000001,ka2 er2 pu3 pei2 wai4 sun1 wan2 hua2 ti1  
	000002,jia2 yu3 cun1 yan2 bie2 zai4 yong1 bao4 wo3  
	000003,bao2 ma3 pei4 gua4 bo3 luo2 an1 diao1 chan2 yuan4 zhen3 dong3 weng1 ta4  
  
phone + rhythm(dictionary.txt):  
	000001,k a2 er2 p u3 #2 p ei2 uai4 s uen1 #1 uan2 h ua2 t i1 #4  。   
	000002,j ia2 v3 c uen1 ian2 #2 b ie2 z ai4 #1 iong1 b ao4 uo3 #4  。   
	000003,b ao2 m a3 #1 p ei4 g ua4 #1 b o3 l uo2 an1 #3  ， d iao1 ch an2 #1 van4 zh en3 #2 d ong3 ueng1 t a4 #4  。   
  

- Step **(0)**: python frozen_tacotron2.py, set tf_pyfunc = False in tacotron.py, you can freeze the model.  
- Step **(1)**: python gen_serving_model_tacotron2.py , convert pb to savemodel.  

# melgan  
- Step **(0)**: python melgan2onnx.py, use MyRefPad1d() instead of nn.ReflectionPad1d() , convert pt to onnx.  
- Step **(1)**: python onnx2trt.py , convert onnx to trt , trt support dynamic input shape.  
  
# Training and Inference:  
  
gta:  
- Step **(0)**: python preprocess.py ，process the audios for t2 and melgan training .  
- Step **(1)**: python train_tacotron.py ，while finish the t2 training, it will generate gta data .  
- Step **(2)**: cd melgan, cp audio-xxx.npy and mel-xxx.npy(gta data) to melgan's training/validing data-path .  
- Step **(3)**: python train_melgan.py .  
  
real mel:  
- Step **(0)**: python preprocess.py ，process the audios for t2 and melgan training .  
- Step **(1)**: cp audio-xxx.npy and mel-xxx.npy(real mel) to melgan's training/validing data-path .  
- Step **(2)**: python train_tacotron.py .  
- Step **(3)**: python train_melgan.py ，train the melgan with the real mel data.  
  
also ,run inference_melgan.py if you only interested in vocoder .  
  
  
# reference:  
https://github.com/Rayhane-mamah/Tacotron-2  
https://github.com/seungwonpark/melgan  
  
