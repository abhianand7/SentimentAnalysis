import numpy as np
import tensorflow as tf
import tensorflow_hub
import tensorflow_text
from tensorflow.keras.utils import to_categorical
from typing import Union
import os
import pandas as pd
from utils import read_csv_data
from tensorflow.python.ops import array_ops
from tensorflow.python.data.ops import dataset_ops
tf.get_logger().setLevel('ERROR')
# tf.data.experimental.CsvDataset()
# tf.keras.preprocessing.text_dataset_from_directory


class DataIngestion:
    def __init__(self, train_df: pd.DataFrame, test_df: Union[None, pd.DataFrame],
                 val_df: Union[None, pd.DataFrame], input_col: Union[list, str], target_col: Union[list, str],
                 seed: Union[None, int], verbose: bool, batch_size: int):
        self.batch_size = batch_size
        self.verbose = verbose
        self.train_df = train_df
        self.test_df = test_df
        self.val_df = val_df
        self.input_col = input_col
        self.target_col = target_col
        self.seed = seed
        self.AUTOTUNE = tf.data.AUTOTUNE
        self.multi_output_flag = False

    def create_data_pipeline(self):
        no_train_elements = self.train_df.__len__()

        if isinstance(self.target_col, list) and len(self.target_col) > 1:
            self.multi_output_flag = True
            self.train_df, y_train = self._get_target_ds(self.train_df, self.target_col)
            num_classes = len(self.target_col)
        else:
            self.multi_output_flag = False
            num_classes = self.train_df[self.target_col].unique()
            num_classes = len(num_classes)
            target_feature = self.train_df.pop(self.target_col)

            target_values = target_feature.values
            y_train = array_ops.one_hot(target_values, num_classes)
            y_train = tf.data.Dataset.from_tensor_slices((y_train))

        if self.verbose:
            if self.multi_output_flag:
                print(f'Multi Output prediction required for: {num_classes} target variables')
            else:
                print(f'single target variable with {num_classes} categories in target variable')
        x_train = self.train_df[self.input_col].values
        x_train = tf.data.Dataset.from_tensor_slices((tf.constant(x_train)))

        dataset = tf.data.Dataset.zip((x_train, y_train))

        dataset = dataset.shuffle(buffer_size=no_train_elements, seed=self.seed)

        # dataset = dataset.batch(batch_size=self.batch_size)

        train_dataset = dataset.take(int(no_train_elements * 0.70))
        train_dataset = train_dataset.batch(self.batch_size)

        test_dataset = dataset.skip(int(no_train_elements * 0.70))

        val_dataset = test_dataset.skip(int(no_train_elements * 0.15))
        val_dataset = val_dataset.batch(self.batch_size)

        test_dataset = test_dataset.take(int(no_train_elements * 0.15))
        test_dataset = test_dataset.batch(self.batch_size)
        if self.verbose:
            print(f'Training dataset size: {tf.data.experimental.cardinality(train_dataset).numpy()}')
            print(f'Test dataset size: {tf.data.experimental.cardinality(test_dataset).numpy()}')
            print(f'Validation dataset size: {tf.data.experimental.cardinality(val_dataset).numpy()}')

        train_dataset = train_dataset.cache().prefetch(buffer_size=self.AUTOTUNE)
        test_dataset = test_dataset.cache().prefetch(buffer_size=self.AUTOTUNE)
        val_dataset = val_dataset.cache().prefetch(buffer_size=self.AUTOTUNE)

        return train_dataset, test_dataset, val_dataset, num_classes

    @staticmethod
    def _get_target_ds(df: pd.DataFrame, target_cols):
        all_target_ds = []
        for col in target_cols:
            target_values = df.pop(col).values
            target_ds = tf.data.Dataset.from_tensor_slices(target_values)
            all_target_ds.append(target_ds)

        all_target_ds = tf.data.Dataset.zip(tuple(all_target_ds))
        return df, all_target_ds


if __name__ == '__main__':
    train_filename = '../dataset/train.csv'
    # train_filename = '../dataset/toxic_train.csv'
    test_filename = '../dataset/test.csv'
    input_col = 'Phrase'
    target_col = 'Sentiment'
    # input_col = 'comment_text'
    # target_col = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

    csv_util = read_csv_data.CSVPipeline(
        file_path=train_filename,
        verbose=True,
        separator=',',
        input_col=input_col,
        target_col=target_col
    )
    train_ds = csv_util.return_df()
    class_names = csv_util.get_class_names()
    print(class_names)
    # test_ds = read_csv_data.CSVPipeline(file_path=test_filename, verbose=True, separator=',').return_df()
    data_obj = DataIngestion(
        train_df=train_ds,
        test_df=None,
        val_df=None,
        input_col=input_col,
        target_col=target_col,
        seed=7,
        verbose=True,
        batch_size=32
    )

    train, test, val, classes_len = data_obj.create_data_pipeline()

    print(train.take(1))

    for text_batch, label_batch in train.take(1):
        for i in range(1):
            print(f'Review: {text_batch.numpy()[i]}')
            # label = label_batch[i]
            label = label_batch.numpy()[i]
            print(label)
            # print(f'Label : {label} ({class_names[label]})')

    for text_batch, label_batch in test.take(1):
        for i in range(1):
            print(f'Review: {text_batch.numpy()[i]}')
            # label = label_batch[i]
            label = label_batch.numpy()[i]
            print(label)
            # print(f'Label : {label} ({class_names[label]})')
