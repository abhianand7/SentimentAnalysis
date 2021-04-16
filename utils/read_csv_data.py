import pandas as pd
import os
from typing import Union
import typing


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
        all_cols = list()

        if isinstance(self.input_col, typing.List):
            all_cols.extend(self.input_col)
        elif isinstance(self.input_col, str):
            all_cols.append(self.input_col)
        else:
            raise TypeError(f'Incorrect type passed for input_col. '
                            f'\n expected list or str but received:{type(self.input_col)}')
        if isinstance(self.target_col, typing.List):
            all_cols.extend(self.target_col)
        elif isinstance(self.target_col, str):
            all_cols.append(self.target_col)
        else:
            raise TypeError(f'Incorrect type passed for input_col. '
                            f'\n expected list or str but received:{type(self.target_col)}')

        self.df = self.df[[self.input_col, self.target_col]]
        self.df.dropna(
            axis=0,
            inplace=True
        )
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
