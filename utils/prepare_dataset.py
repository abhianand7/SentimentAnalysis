import numpy as np
import tensorflow as tf
import tensorflow_hub
import tensorflow_text
from tensorflow.keras.utils import to_categorical
from typing import Union
import os
import pandas as pd
from utils import read_csv_data

tf.get_logger().setLevel('ERROR')
# tf.data.experimental.CsvDataset()


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

    def create_data_pipeline(self):
        target_feature = self.train_df.pop(self.target_col)
        no_train_elements = self.train_df.__len__()

        # target_values = np.asarray(target_feature.values).astype('int64').reshape((-1, 1))
        target_values = to_categorical(target_feature.values, num_classes=5)

        dataset = tf.data.Dataset.from_tensor_slices((self.train_df.values, target_values))

        dataset = dataset.shuffle(buffer_size=no_train_elements, seed=self.seed)

        dataset = dataset.batch(batch_size=self.batch_size)

        train_dataset = dataset.take(int(no_train_elements * 0.70))

        test_dataset = dataset.skip(int(no_train_elements * 0.70))

        val_dataset = test_dataset.skip(int(no_train_elements * 0.15))

        test_dataset = test_dataset.take(int(no_train_elements * 0.15))
        if self.verbose:
            print(f'Training dataset size: {tf.data.experimental.cardinality(train_dataset).numpy()}')
            print(f'Test dataset size: {tf.data.experimental.cardinality(test_dataset).numpy()}')
            print(f'Validation dataset size: {tf.data.experimental.cardinality(val_dataset).numpy()}')

        train_dataset = train_dataset.cache().prefetch(buffer_size=self.AUTOTUNE)
        test_dataset = test_dataset.cache().prefetch(buffer_size=self.AUTOTUNE)
        val_dataset = val_dataset.cache().prefetch(buffer_size=self.AUTOTUNE)

        return train_dataset, test_dataset, val_dataset


if __name__ == '__main__':
    train_filename = '../dataset/train.csv'
    test_filename = '../dataset/test.csv'
    input_col = 'Phrase'
    target_col = 'Sentiment'
    csv_util = read_csv_data.CSVPipeline(
        file_path=train_filename,
        verbose=True,
        separator=',',
        input_col='Phrase',
        target_col='Sentiment'
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
        verbose=True
    )

    train, test, val = data_obj.create_data_pipeline()

    print(train.take(1))

    for text_batch, label_batch in train.take(1):
        for i in range(1):
            print(f'Review: {text_batch.numpy()[i]}')
            label = label_batch.numpy()[i]
            print(f'Label : {label} ({class_names[label]})')
