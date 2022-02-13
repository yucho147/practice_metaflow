import os
import pickle
import random

from attrdict import AttrDict
from metaflow import FlowSpec, Parameter, step
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml


def load_config(config_path):
    """config(yaml)ファイルを読み込む
    Parameters
    ----------
    config_path : string
        config fileのパスを指定する
    Returns
    -------
    config : attrdict.AttrDict
        configを読み込んでattrdictにしたもの
    """
    with open(config_path, 'r', encoding='utf-8') as fi_:
        return AttrDict(yaml.load(fi_, Loader=yaml.SafeLoader))


def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True


class RegModelFlow(FlowSpec):
    config_file = Parameter('config_file',
                            help='config file',
                            default="./configs/test.yaml")
    @step
    def start(self):
        """start step
        metaflowは必ずstartという名前のメソッドから始まる
        """
        self.conf = load_config(self.config_file)
        self.next(self.load_data)

    @step
    def load_data(self):
        """データの取得step
        """
        conf = self.conf.data
        housing = fetch_california_housing()
        self.X = pd.DataFrame(housing[conf.X.data_name],
                              columns=housing[conf.X.columns_name])
        self.y = pd.DataFrame(housing[conf.y.data_name],
                              columns=housing[conf.y.columns_name])
        self.next(self.preprocess)

    @step
    def preprocess(self):
        """前処理step
        今回することなさそう。通常NaNを埋めたり、正規化したり。
        """
        self.next(self.split_data)

    @step
    def split_data(self):
        """データの分割step
        """
        conf = self.conf.split_data
        (
            self.train_X,
            self.test_X,
            self.train_y,
            self.test_y
        ) = train_test_split(self.X, self.y,
                             **conf.train_test_split)
        self.next(self.train, self.save_data)

    @step
    def save_data(self):
        """データの保存step
        """
        conf = self.conf.save_data
        self.train_X.to_csv(**conf.train_X)
        self.train_y.to_csv(**conf.train_y)
        self.test_X.to_csv(**conf.test_X)
        self.test_y.to_csv(**conf.test_y)
        self.next(self.join)

    @step
    def train(self):
        """学習step
        """
        conf = self.conf.train

        (
            self.train_X,
            self.valid_X,
            self.train_y,
            self.valid_y
        ) = train_test_split(self.train_X, self.train_y,
                             **conf.train_test_split)

        self.model = lgb.LGBMRegressor(**conf.LGBMRegressor)
        callbacks = [lgb.early_stopping(**conf.early_stopping),
                     lgb.log_evaluation(**conf.log_evaluation)]
        self.model.fit(self.train_X, self.train_y,
                       eval_set=[(self.valid_X, self.valid_y)],
                       callbacks=callbacks, **conf.fit)
        pred = self.model.predict(self.test_X)
        print(f"MSE: {mean_squared_error(self.test_y, pred)}")
        self.next(self.save_model)

    @step
    def save_model(self):
        """学習済みモデルの保存step
        """
        conf = self.conf.save_model
        with open(conf.save_path, "wb") as fo:
            pickle.dump(self.model, fo)
        self.next(self.join)

    @step
    def join(self, inputs):
        """flow結合step
        end stepと結合のためのstepは同時に使えない
        """
        self.next(self.end)

    @step
    def end(self):
        """end step
        metaflowは必ずstartという名前のメソッドから始まる
        """
        print("Done all process")
        pass


def main():
    RegModelFlow()


if __name__ == '__main__':
    main()
