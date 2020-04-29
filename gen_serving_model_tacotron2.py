#-*- coding: utf-8 -*-

import tensorflow as tf
import argparse
import os
import shutil

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

export_model = './save_model'
parser = argparse.ArgumentParser(description='Generate a saved model.')
parser.add_argument('--export_model_dir', type=str, default=export_model, help='export model directory')
parser.add_argument('--model_version', type=int, default=1, help='model version')
parser.add_argument('--model', type=str, default='tacotron2.pb', help='model pb file')

args = parser.parse_args()

if os.path.exists(export_model):
    shutil.rmtree(export_model)

if __name__ == '__main__':

    #-----------------
    with tf.Session() as sess:
        with tf.gfile.GFile(args.model, "rb") as f:
            restored_graph_def = tf.GraphDef()
            restored_graph_def.ParseFromString(f.read())
        tf.import_graph_def(
            restored_graph_def,
            input_map=None,
            return_elements=None,
            name="")

        # #打印节点信息
        # tensor_name_list = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
        # for tensor_name in tensor_name_list:
        #     print(tensor_name)

        export_path_base = args.export_model_dir
        export_path = os.path.join(tf.compat.as_bytes(export_path_base),
            tf.compat.as_bytes(str(args.model_version)))
        print('Exporting trained model to', export_path)
        builder = tf.saved_model.builder.SavedModelBuilder(export_path)

        # input
        in_tensor = tf.get_default_graph().get_tensor_by_name('inputs:0')
        in_length_tensor = tf.get_default_graph().get_tensor_by_name('input_lengths:0')
        split_infos_tensor = tf.get_default_graph().get_tensor_by_name('split_infos:0')

        # output
        mel_output_tensor = tf.get_default_graph().get_tensor_by_name('Tacotron_model/inference/Minimum_1:0')
        stop_token_output_tensor = tf.get_default_graph().get_tensor_by_name('Tacotron_model/inference/Reshape_2:0')

        # postprocess
        mel_output_tensor = tf.squeeze(mel_output_tensor, 0)
        stop_token_output_round = tf.round(stop_token_output_tensor)
        keep = tf.where(tf.squeeze(stop_token_output_round, 0) >= 1)
        keep_index = tf.squeeze(tf.cast(keep, tf.int32))
        output_lengths = tf.cond(tf.equal(tf.size(keep_index), 0), lambda: tf.shape(stop_token_output_round)[1],
                         lambda: keep_index)
        mel_output_tensor = mel_output_tensor[:output_lengths, :]
        mel_output_tensor = tf.clip_by_value(mel_output_tensor, -4, 4)


        #build_tensor_info
        tensor_in = tf.saved_model.utils.build_tensor_info(in_tensor)
        tensor_in_length = tf.saved_model.utils.build_tensor_info(in_length_tensor)
        tensor_split_infos = tf.saved_model.utils.build_tensor_info(split_infos_tensor)

        model_out_mel = tf.saved_model.utils.build_tensor_info(mel_output_tensor)

        prediction_signature = (
            tf.saved_model.signature_def_utils.build_signature_def(
                inputs={'inputs': tensor_in,
                        'input_lengths': tensor_in_length,
                        'split_infos': tensor_split_infos,
                        },
                outputs={'mel': model_out_mel,
                         },
                method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
            ))

        builder.add_meta_graph_and_variables(
            sess, [tf.saved_model.tag_constants.SERVING],
            signature_def_map={
                'predict':prediction_signature,
            })

        builder.save(as_text=False)
    print('Done exporting!')

