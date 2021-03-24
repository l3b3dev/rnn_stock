from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import torch
from dateutil import parser
import numpy as np
import pandas as pd
import datetime
import time
import math

from GaussianNoiseTransform import GaussianNoiseTransform
from LSTM import LSTM


class TrainingPipeline:

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, epochs):
        self.model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
        self.criterion = torch.nn.MSELoss(reduction='mean')
        self.optimiser = torch.optim.Adam(self.model.parameters(), lr=0.01)
        self.num_epochs = epochs
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def load_data(self, data_dir):
        # load shiller PE
        shiller = pd.read_csv(f'{data_dir}/shiller.csv', parse_dates=[0])
        # filter dates from: Jan 1, 1960 to Dec 31, 2020
        shiller = shiller[(shiller['Date'] > '1-1-1960') & (shiller['Date'] <= '12-31-2020')]
        shiller.sort_values(by='Date', inplace=True, ascending=True)

        # get index
        xls = pd.ExcelFile(f'{data_dir}/ie_data.xls')
        spx = pd.read_excel(xls, 'Data', converters={0: str})

        # Create column for Date
        spx['Date'] = None

        # set index
        index_set = 0
        index_date = spx.columns.get_loc('Date')
        print(index_set, index_date)

        # define pattern for date
        # in DD/MM/YYYY
        date_pattern = r'([0-9]{4}\.[0-9]{1,2})'

        # searching pattern
        # And storing in to DataFrame
        for row in range(0, len(spx)):
            m = spx.iat[row, index_set].split('.')
            spx.iat[row, index_date] = datetime.datetime(int(m[0]), int(m[1]), 1)

        spx = spx[(spx['Date'] > parser.parse('1-1-1960')) & (spx['Date'] <= parser.parse('12-31-2020'))]
        spx.sort_values(by='Date', inplace=True, ascending=True)

        return shiller, spx

    def normalize_price(self, spx):
        price = spx.iloc[:, 7].to_frame('Close')
        price['Close'] = self.scaler.fit_transform(price['Close'].values.reshape(-1, 1))

        return price

    def train_test_split(self, stock, wnd):
        data_raw = stock.to_numpy()  # convert to numpy array
        data = []

        # create all possible sequences of length seq_len
        for index in range(len(data_raw) - wnd):
            data.append(data_raw[index: index + wnd])

        data = np.array(data)
        # 20%-80% test-train split
        test_set_size = int(np.round(0.2 * data.shape[0]))
        train_set_size = data.shape[0] - (test_set_size)

        x_train = data[:train_set_size, :-1, :]
        y_train = data[:train_set_size, -1, :]

        x_test = data[train_set_size:, :-1]
        y_test = data[train_set_size:, -1, :]

        x_train = torch.from_numpy(x_train).type(torch.Tensor)
        x_test = torch.from_numpy(x_test).type(torch.Tensor)
        y_train = torch.from_numpy(y_train).type(torch.Tensor)
        y_test = torch.from_numpy(y_test).type(torch.Tensor)

        return [x_train, y_train, x_test, y_test]

    def get_test_data(self, spx, wnd, filter_date):
        spx = spx[spx['Date'] > parser.parse(filter_date)]

        price = self.normalize_price(spx)

        data_raw = price.to_numpy()

        data = []

        # create all possible sequences of length seq_len
        for index in range(len(data_raw) - wnd):
            data.append(data_raw[index: index + wnd])

        data = np.array(data)

        x_test = data[:, :-1]
        y_test = data[:, -1, :]

        x_test = torch.from_numpy(x_test).type(torch.Tensor)
        y_test = torch.from_numpy(y_test).type(torch.Tensor)

        return x_test, y_test, spx

    def train(self, x_train, y_train):
        hist = np.zeros(self.num_epochs)
        start_time = time.time()

        for t in range(self.num_epochs):
            y_train_pred = self.model(x_train)
            loss = self.criterion(y_train_pred, y_train)
            print("Epoch ", t, "MSE: ", loss.item())
            hist[t] = loss.item()
            self.optimiser.zero_grad()
            loss.backward()
            self.optimiser.step()

        training_time = time.time() - start_time
        print("Training time: {}".format(training_time))

        return y_train_pred, hist

    def inverse_transform_pair(self, y_train, y_train_original):
        predict = self.scaler.inverse_transform(y_train.detach().numpy())
        original = self.scaler.inverse_transform(y_train_original.detach().numpy())

        return predict, original

    def predict(self, x_test):
        # make predictions
        y_test_pred = self.model(x_test)

        return y_test_pred

    def compute_RMSE(self, x, y):
        return np.math.sqrt(mean_squared_error(x[:, 0], y[:, 0]))

    def compute_accuracy(self, y, x, sd=0.):
        return pd.DataFrame((y - x) / x, columns=[f'{sd}'])

    def collect(self, window, price, y_train_pred, y_test_pred):
        # shift train predictions for plotting
        trainPredictPlot = np.empty_like(price)
        trainPredictPlot[:, :] = np.nan
        trainPredictPlot[window:len(y_train_pred) + window, :] = y_train_pred

        # shift test predictions for plotting
        testPredictPlot = np.empty_like(price)
        testPredictPlot[:, :] = np.nan
        testPredictPlot[len(y_train_pred) + window - 1:len(price) - 1, :] = y_test_pred

        original = self.scaler.inverse_transform(price['Close'].values.reshape(-1, 1))

        predictions = np.append(trainPredictPlot, testPredictPlot, axis=1)
        predictions = np.append(predictions, original, axis=1)
        result = pd.DataFrame(predictions)

        return result

    def compute_noise_accuracy(self, sdevs, x_test, y_test, dt):

        result = pd.DataFrame(dt.head(len(x_test)))
        result.reset_index(drop=True, inplace=True)
        for sd in sdevs:
            noise = GaussianNoiseTransform(std=sd, k=18)
            x_test_noise = noise(x_test)

            # now predict on test data
            y_test_pred = self.predict(x_test_noise)
            y_test_pred, y = self.inverse_transform_pair(y_test_pred, y_test)

            # now calculate accuracy as (predicted_price-actual_price)/actual_price
            df = self.compute_accuracy(y_test_pred, y, sd)
            result = pd.concat([result, df], axis=1)

        return result


