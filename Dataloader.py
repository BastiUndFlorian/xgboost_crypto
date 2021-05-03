import pandas as pd
import numpy as np
DATA_PATH = "/home/florian/Documents/xgboost_crypto/data/"
FILE_NAME = "gemini_BTCUSD_%YEAR%_1min.csv"
class Dataloader:
    def __init__(self):
        df = pd.DataFrame()
        for year in [2020]: #2018,2019,
            df_tmp = pd.read_csv(DATA_PATH + FILE_NAME.replace("%YEAR%",str(year)),sep = ",", skiprows=1)
            df = df.append(df_tmp[::-1])
        print(df[["Low","High","Volume","Open","Close"]])

        df["ma_30"] = df['Close'].rolling(30).mean()
        df["ma_50"] = df['Close'].rolling(50).mean()
        df["ma_200"] = df['Close'].rolling(200).mean()
        df["ma_1440"] = df['Close'].rolling(1440).mean()
        df["ma_10000"] = df['Close'].rolling(10000).mean()
        df["ma_43800"] = df['Close'].rolling(43800).mean()
        df = df.dropna()
        data_full = self.series_to_supervised(data=df[["ma_30","ma_50","ma_200","ma_1440","ma_10000","ma_43800","Close"]],n_in=10)  # "Low","High","Volume","Open",

        n = data_full.shape[0]

        self.train = data_full[:int(np.round(n*0.9))]
        self.test = data_full[int(np.round(n * 0.9)):]




    def get_train(self):
        return self.train
    def get_test(self):
        return  self.test


    def series_to_supervised(self,data, n_in=1, n_out=1, dropnan=True):
        n_vars = 1 if type(data) is list else data.shape[1]
        df = pd.DataFrame(data)
        cols = list()
        # input sequence (t-n, ... t-1)
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, n_out):
            cols.append(df.shift(-i))
        # put it all together
        agg = pd.concat(cols, axis=1)
        # drop rows with NaN values
        if dropnan:
            agg.dropna(inplace=True)
        return agg.values
if __name__ == "__main__":
    dl = Dataloader()