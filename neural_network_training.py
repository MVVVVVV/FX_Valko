import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
import features_test
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
import pathlib

class neural_network():
    data: object
    output: object

    def __init__(self, data, curr_pair):
        """

        :type data: object
        """
        self.filename = 'finalized_model_' + curr_pair.replace('/', '_') + '.sav'
        self.data = data
        self.input_params = ["mid_price", "B_vol", "S_vol", "mid_MA_10", "sum_volume", "UpperBand", "LowerBand"]

    def train_nn(self):
        self.__data_formatting()
        self.__find_best_params()
        self.__get_results()

    def __data_formatting(self):
        self.data = pd.DataFrame(np.nan_to_num(self.data[self.input_params].astype(np.float32).fillna(method='ffill')),
                                 columns=self.input_params)
        self.output = self.data["mid_price"].shift(1) > self.data["mid_price"]
        self.input = self.data[self.input_params]
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.input.iloc[1:, :],
                                                                                self.output.iloc[1:])

    def __find_best_params(self):
        parameters = {'solver': ['lbfgs'], 'max_iter': [1500], 'alpha': 10.0 ** -np.arange(1, 7),
                      'hidden_layer_sizes': [np.arange(5, 12), np.arange(5, 12)]}
        self.opt = GridSearchCV(MLPClassifier(), parameters, n_jobs=-1)

        self.opt.fit(self.x_train, self.y_train)

    def __get_results(self):
        print("val. score: %s" % self.opt.best_score_)
        print("test score: %s" % self.opt.score(self.x_test, self.y_test))
        print("best params: %s" % str(self.opt.best_params_))
        pickle.dump(self.opt.best_estimator_, open(str(pathlib.Path().absolute()) + "\\" + self.filename, "wb"))


if __name__ == "__main__":
    data = features_test.main()
    FX_list = ['EUR/JPY', 'AUD/USD', 'USD/CHF', 'NOK/SEK', 'USD/JPY', 'EUR/USD', 'USD/CAD', 'GBP/USD']
    nn = {curr_pair: neural_network(data[curr_pair].timestamp_base, curr_pair) for curr_pair in FX_list}

    for curr_pair in FX_list:
        nn[curr_pair].train_nn()
