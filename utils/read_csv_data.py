import pandas as pd
import os
from typing import Union


class CSVPipeline:
    def __init__(self, file_path: str, verbose: bool, separator: Union[None, str],
                 input_col: Union[list, str], target_col: Union[list, str]):
        self.file_path = file_path
        if not os.path.exists(self.file_path):
            raise FileNotFoundError()
        self.verbose = verbose
        self.separator = separator
        self.input_col = input_col
        self.target_col = target_col
        self.df = pd.DataFrame
        self.data_read_flag = False

    def return_df(self):
        self.df = pd.read_csv(self.file_path, sep=self.separator)
        self.df = self.df[[self.input_col, self.target_col]]
        self.df.dropna()
        if self.verbose:
            self.df.info()
        self.data_read_flag = True
        return self.df

    def get_class_names(self):
        if self.data_read_flag:
            classes = self.df[self.target_col].unique()
            return classes


if __name__ == '__main__':
    pass
