import os
import numpy as np
import tensorflow_text
import tensorflow_hub
import tensorflow as tf
from tensorflow.keras.losses import CategoricalCrossentropy
from official.nlp import optimization
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from typing import Union
# custom imports
from models import bert_classifier
from utils import prepare_dataset
from utils import read_csv_data
import time

seed = 67


class TrainPipeline:
    def __init__(self, bert_model_name: Union[None, str], epochs: int,
                 train_csv_data: str, train_data_sep: Union[None, str],
                 input_col: str, target_col: str,
                 tf_hub_models_config: str, verbose: bool, run_dir: str
                 ):
        self.run_dir = run_dir
        self.input_col = input_col
        self.train_data_sep = train_data_sep
        self.train_csv_data = train_csv_data
        self.bert_model_name = bert_model_name
        self.epochs = epochs
        self.target_col = target_col
        self.tf_hub_models_config = tf_hub_models_config
        self.verbose = verbose
        if not os.path.exists(self.run_dir):
            os.mkdir(run_dir)

        timestamp = round(time.time(), 0).__str__()
        self.model_save_path = os.path.join(run_dir, timestamp)

    def run(self):
        csv_utils = read_csv_data.CSVPipeline(
            self.train_csv_data,
            verbose=self.verbose,
            separator=self.train_data_sep,
            input_col=self.input_col,
            target_col=self.target_col
        )
        raw_df = csv_utils.return_df()

        classes = csv_utils.get_class_names()
        classes_len = len(classes)
        print(classes_len)

        # raw_df[self.target_col] = to_categorical(raw_df[self.target_col], num_classes=classes_len)

        data_utils = prepare_dataset.DataIngestion(
            train_df=raw_df,
            test_df=None,
            val_df=None,
            input_col=self.input_col,
            target_col=self.target_col,
            seed=seed
        )

        train_ds, test_ds, val_ds = data_utils.create_data_pipeline()

        model_utils = bert_classifier.ClassifierPipeline(
            bert_model_name=self.bert_model_name,
            tf_hub_models_config=self.tf_hub_models_config,
            verbose=self.verbose,
            num_classes=classes_len
        )

        bert_model = model_utils.build_cls_model()

        steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy()
        num_train_steps = steps_per_epoch * self.epochs
        num_warmup_steps = int(0.1 * num_train_steps)

        init_lr = 3e-5
        optimizer = optimization.create_optimizer(init_lr=init_lr,
                                                  num_train_steps=num_train_steps,
                                                  num_warmup_steps=num_warmup_steps,
                                                  optimizer_type='adamw')

        loss = CategoricalCrossentropy(from_logits=True)
        metrics = []
        acc_metrics = tf.metrics.Accuracy()
        metrics.append(acc_metrics)
        auc_metrics = tf.metrics.AUC()
        metrics.append(auc_metrics)

        bert_model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )

        history = bert_model.fit(
            x=train_ds,
            validation_data=val_ds,
            epochs=self.epochs
        )

        bert_model.save(self.model_save_path, include_optimizer=False)

        loss, accuracy = bert_model.evaluate(test_ds)

        print(f'Loss: {loss}')
        print(f'Accuracy: {accuracy}')
        self._plot_graph(history)

    def _plot_graph(self, history):
        history_dict = history.history
        print(history_dict.keys())

        acc = history_dict['accuracy']
        val_acc = history_dict['val_accuracy']
        loss = history_dict['loss']
        val_loss = history_dict['val_loss']

        epochs = range(1, len(acc) + 1)
        fig = plt.figure(figsize=(10, 6))
        fig.tight_layout()

        plt.subplot(2, 1, 1)
        # "bo" is for "blue dot"
        plt.plot(epochs, loss, 'r', label='Training loss')
        # b is for "solid blue line"
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        # plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(epochs, acc, 'r', label='Training acc')
        plt.plot(epochs, val_acc, 'b', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend(loc='lower right')


if __name__ == '__main__':
    train_utils = TrainPipeline(
        bert_model_name='bert_en_uncased_L-12_H-768_A-12',
        epochs=1,
        train_csv_data='dataset/train.csv',
        train_data_sep=',',
        input_col='Phrase',
        target_col='Sentiment',
        tf_hub_models_config='tf_hub_models.json',
        verbose=True,
        run_dir='runs'
    )
    train_utils.run()
    pass
