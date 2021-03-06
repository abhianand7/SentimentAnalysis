import os
import numpy as np
import tensorflow_text
import tensorflow_hub
import tensorflow as tf
from tensorflow import Tensor
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import CategoricalAccuracy, AUC, Precision, Recall
from tensorflow.keras.callbacks import TerminateOnNaN, EarlyStopping
from official.nlp import optimization
import matplotlib.pyplot as plt
from typing import Union, Any
# custom imports
from models import bert_classifier
from utils import prepare_dataset
from utils import read_csv_data
import time
import traceback

seed = 67


class TrainPipeline:
    def __init__(self, bert_model_name: Union[None, str], epochs: int,
                 train_csv_data: str, train_data_sep: Union[None, str],
                 input_col: str, target_col: str,
                 tf_hub_models_config: str, verbose: bool, run_dir: str,
                 batch_size: int, optimizer: str
                 ):
        """
        :param bert_model_name: any transformer model name which is present in tf hub
        :param epochs:
        :param train_csv_data:
        :param train_data_sep:
        :param input_col: list of input cols
        :param target_col:  list of output cols
        :param tf_hub_models_config:
        :param verbose:
        :param run_dir: path for saving the model
        :param batch_size:
        :param optimizer: define which optimizer to use
        """
        self.optimizer = optimizer
        self.batch_size = batch_size
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

        # classes = csv_utils.get_class_names()
        # classes_len = len(classes)
        # print(classes_len)

        # raw_df[self.target_col] = to_categorical(raw_df[self.target_col], num_classes=classes_len)

        data_utils = prepare_dataset.DataIngestion(
            train_df=raw_df,
            test_df=None,
            val_df=None,
            input_col=self.input_col,
            target_col=self.target_col,
            seed=seed,
            verbose=True,
            batch_size=self.batch_size
        )

        train_ds, test_ds, val_ds, classes_len = data_utils.create_data_pipeline()

        model_utils = bert_classifier.ClassifierPipeline(
            bert_model_name=self.bert_model_name,
            tf_hub_models_config=self.tf_hub_models_config,
            verbose=self.verbose,
            num_classes=classes_len
        )

        # create strategy for multi-gpu training
        strategy = tf.distribute.MirroredStrategy()
        if self.verbose:
            print(f'Number of devices: {strategy.num_replicas_in_sync}')

        with strategy.scope():
            bert_model = model_utils.build_cls_model()

            steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy()
            num_train_steps = steps_per_epoch * self.epochs
            num_warmup_steps = int(0.1 * num_train_steps)

            init_lr = 0.001
            if self.optimizer == 'adam':
                optimizer = self.get_optimizer(optimizer=self.optimizer, learning_rate=init_lr)
            elif self.optimizer == 'adamw':
                # AdamW is a better version of Adam with dynamic weight decay
                optimizer = self.get_optimizer(optimizer=self.optimizer,
                                               learning_rate=init_lr,
                                               num_train_steps=num_train_steps,
                                               num_warmup_steps=num_warmup_steps
                                               )
            else:
                optimizer = 'adam'

            # loss function for the model
            loss = CategoricalCrossentropy(from_logits=False)

            # metrics for evaluating model performance
            metrics = self.get_metrics()

            # callbacks for controlling the training properly
            callbacks = self.get_callbacks()

            bert_model.compile(
                optimizer=optimizer,
                loss=loss,
                metrics=metrics
            )

        history = bert_model.fit(
            x=train_ds,
            validation_data=val_ds,
            epochs=self.epochs,
            callbacks=callbacks
        )

        # save trained model
        bert_model.save(self.model_save_path, include_optimizer=False)

        # plot graph
        try:
            self._plot_graph(history)
        except KeyError:
            errors = traceback.format_exc()
            print(errors)

        # model evaluation on the test dataset
        evaluation_result = bert_model.evaluate(test_ds)
        print(evaluation_result)
        return self.model_save_path

    @staticmethod
    def get_optimizer(optimizer: str, learning_rate: float, **kwargs):
        """

        :param optimizer:
        :param learning_rate:
        :param kwargs: any additional argument which might be needed
         two options includes: num_train_steps, num_warmup_steps for adamw optimizer
        :return:
        """
        if optimizer == 'adam':
            optimizer = Adam(learning_rate=learning_rate)
        elif optimizer == 'adamw':
            num_train_steps = kwargs['num_train_steps']
            num_warmup_steps = kwargs['num_warmup_steps']
            optimizer = optimization.create_optimizer(init_lr=learning_rate,
                                                      num_train_steps=num_train_steps,
                                                      num_warmup_steps=num_warmup_steps,
                                                      optimizer_type='adamw')
        return optimizer

    @staticmethod
    def get_callbacks():
        callbacks = []
        terminate_on_nan = TerminateOnNaN()
        callbacks.append(terminate_on_nan)

        early_stopping = EarlyStopping(monitor='loss', patience=3, verbose=1, mode='auto')
        callbacks.append(early_stopping)

        return early_stopping

    @staticmethod
    def get_metrics():
        metrics = []
        acc_metrics = CategoricalAccuracy()
        metrics.append(acc_metrics)
        auc_metrics = AUC()
        metrics.append(auc_metrics)
        precision = Precision(name='precision')
        metrics.append(precision)
        recall = Recall(name='recall')
        metrics.append(recall)
        return metrics

    @staticmethod
    def _plot_graph(history):
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

        fig.savefig('full_fig.png')


class TestPipeline:
    def __init__(self, model_path: str, test_file_path: str, test_sep: str, input_col: list, target_col: list):
        """

        :param model_path:
        :param test_file_path:
        :param test_sep:
        :param input_col:
        :param target_col:
        """
        self.target_col = target_col
        self.input_col = input_col
        self.test_file_path = test_file_path
        self.model_path = model_path
        if not os.path.exists(self.model_path):
            raise FileNotFoundError

        # self.model = tf.saved_model.load(self.model_path)
        self.model = tf.keras.models.load_model(self.model_path)
        self.csv_utils = read_csv_data.CSVPipeline(
            file_path=test_file_path,
            verbose=True,
            separator=test_sep,
            input_col=self.input_col,
            target_col=self.target_col
        )

    def run_test(self):
        df = self.csv_utils.return_df()

        raw_x_inputs = df[self.input_col].sample(n=10)
        raw_x_inputs = raw_x_inputs.values
        x_inputs = tf.constant(raw_x_inputs)
        predictions = self.model.predict(x_inputs)
        # print(predictions)
        self._print_sample_predictions(raw_x_inputs, predictions, 10)
        return predictions

    @staticmethod
    def _print_sample_predictions(inputs, results: np.ndarray, sample_size: int):
        """

        :param inputs:
        :param results:
        :param sample_size:
        :return:
        """
        results = results.tolist()
        result_for_printing = \
            [f'input: {inputs[i]} : score: {results[i][np.argmax(results[i])]} class: {np.argmax(results[i])}'
             for i in range(sample_size)]
        print(*result_for_printing, sep='\n')
        print()


if __name__ == '__main__':
    import argparse

    default_bert_model = 'bert_en_uncased_L-12_H-768_A-12'

    # enabling a way to pass arguments directly from terminal
    parser = argparse.ArgumentParser()
    parser.add_argument('epochs', type=int)
    parser.add_argument('batch_size', type=int)
    parser.add_argument('optimizer', type=str)
    parser.add_argument('bert_model', type=str)

    args = parser.parse_args()

    epochs_no = args.epochs
    batch_size_var = args.batch_size
    optimizer_var = args.optimizer
    bert_model_name_var = args.bert_model

    # start training
    train_utils = TrainPipeline(
        bert_model_name=bert_model_name_var,
        epochs=epochs_no,
        train_csv_data='dataset/train.csv',
        train_data_sep=',',
        input_col='Phrase',
        target_col='Sentiment',
        tf_hub_models_config='tf_hub_models.json',
        verbose=True,
        run_dir='runs',
        batch_size=batch_size_var,
        optimizer=optimizer_var
    )
    trained_model_path = train_utils.run()

    # trained_model_path = 'runs/1618573162.0'

    # start testing
    test_utils = TestPipeline(
        model_path=trained_model_path,
        test_file_path='dataset/test.csv',
        test_sep=',',
        input_col=['Phrase'],
        target_col=[]
    )
    test_utils.run_test()
