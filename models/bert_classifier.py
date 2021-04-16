import tensorflow as tf
import tensorflow_hub as tf_hub
import tensorflow_text as tf_text
from tensorflow.keras.layers import Input, Dropout, Dense
from tensorflow.keras import Model
from typing import Union
import json

tf.get_logger().setLevel('ERROR')


class ClassifierPipeline:
    def __init__(self, bert_model_name: Union[None, str], tf_hub_models_config: str, verbose: bool, num_classes: int):
        self.bert_model_name = bert_model_name
        self.tf_hub_models_config = tf_hub_models_config

        with open(self.tf_hub_models_config, 'r') as fobj:
            self.tf_hub_model_dict = json.load(fobj)

        self.map_name_to_handle = self.tf_hub_model_dict['map_name_to_handle']
        self.map_model_to_preprocess = self.tf_hub_model_dict['map_model_to_preprocess']

        self.tf_hub_handle_encoder = self.map_name_to_handle[self.bert_model_name]
        self.tf_hub_handle_preprocess = self.map_model_to_preprocess[self.bert_model_name]
        self.verbose = verbose

        self.num_classes = num_classes

        if self.verbose:
            print(f'BERT model selected           : {self.tf_hub_handle_encoder}')
            print(f'Preprocess model auto-selected: {self.tf_hub_handle_preprocess}')

        self.bert_preprocess_model = tf_hub.KerasLayer(self.tf_hub_handle_preprocess)
        self.bert_model = tf_hub.KerasLayer(self.tf_hub_handle_encoder)

    def _test(self):
        text_test = ['this is such an amazing movie!']
        text_preprocessed = self.bert_preprocess_model(text_test)

        print(f'Keys       : {list(text_preprocessed.keys())}')
        print(f'Shape      : {text_preprocessed["input_word_ids"].shape}')
        print(f'Word Ids   : {text_preprocessed["input_word_ids"][0, :12]}')
        print(f'Input Mask : {text_preprocessed["input_mask"][0, :12]}')
        print(f'Type Ids   : {text_preprocessed["input_type_ids"][0, :12]}')

        bert_results = self.bert_model(text_preprocessed)

        print(f'Loaded BERT: {self.tf_hub_handle_encoder}')
        print(f'Pooled Outputs Shape:{bert_results["pooled_output"].shape}')
        print(f'Pooled Outputs Values:{bert_results["pooled_output"][0, :12]}')
        print(f'Sequence Outputs Shape:{bert_results["sequence_output"].shape}')
        print(f'Sequence Outputs Values:{bert_results["sequence_output"][0, :12]}')

        classifier_model = self.build_cls_model()
        bert_raw_result = classifier_model(tf.constant(text_test))
        print((bert_raw_result))

    def build_cls_model(self):
        input_1 = Input(shape=(), dtype=tf.string, name='text_input')
        pre_process_layer = tf_hub.KerasLayer(self.tf_hub_handle_preprocess, name='PreProcess')

        encoder_inputs = pre_process_layer(input_1)
        encoder = tf_hub.KerasLayer(self.tf_hub_handle_encoder, trainable=True, name='Encoder')

        outputs = encoder(encoder_inputs)

        mod_input = outputs['pooled_output']

        dropout_1 = Dropout(0.1)(mod_input)
        dense_1 = Dense(
            units=64,
            activation='relu'
        )(dropout_1)
        dropout_2 = Dropout(0.1)(dense_1)
        final_out = Dense(self.num_classes, activation='softmax')(dropout_2)

        model = Model(inputs=[input_1], outputs=[final_out])

        if self.verbose:
            model.summary()
        return model


if __name__ == '__main__':
    cls_obj = ClassifierPipeline(
        bert_model_name='bert_en_uncased_L-12_H-768_A-12',
        tf_hub_models_config='../tf_hub_models.json',
        verbose=True,
        num_classes=5
    )
    cls_obj._test()
    # cls_obj.build_cls_model()
    pass
