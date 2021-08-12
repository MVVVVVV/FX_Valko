import pandas as pd
from reduce_mem_usage import reduce_mem_usage

class FX():
    __window = 10
    __nb_std = 2
    
    def __init__(self,curr_pair):
        self.curr_pair = curr_pair
        self.base = pd.DataFrame()
    
    def fill_base(self, dataset_initial):
        self.__fill_base(dataset_initial[dataset_initial["curr_pair"]==self.curr_pair])
        
    def __fill_base(self,dataset):
        self.base["timestamp"] = dataset["EXCHANGE TIMESTAMP"]
        self.base["volume"] = dataset["volume"]
        self.base["price"] = dataset["price"]
        self.base["B/S"] = dataset["B/S"]
        self.timestamp_base = pd.DataFrame(index=set(self.base["timestamp"]))
        self.timestamp_base = self.timestamp_base.sort_index()
    
    def features(self, n=__window, m=__nb_std):
        self.__sum_volume()    # Sum of volumes
        self.__mid_price()     # Compute a mid price for each timestamp
        self.__timestamp_vol() # B/S volatility for each timestamp
        self.__price_moving_average(n)   # Moyenne mobile somme vol/ prix 
        self.__remove_old_features()
        self.__fill_na()
        self.__bollinger_bands(n,m)    # Bollinger bands on mid price
    
    def __sum_volume(self):
        self.timestamp_base["sum_volume_B"] = self.base[self.base["B/S"]=="B"].groupby("timestamp")["volume"].sum()
        self.timestamp_base["sum_volume_S"] = self.base[self.base["B/S"]=="S"].groupby("timestamp")["volume"].sum()
        self.timestamp_base["sum_volume"] = self.base.groupby("timestamp")["volume"].sum()
        
    def __mid_price(self):
        # Average ask price by timestamp
        self.timestamp_base["average_bid_price"] = self.base[self.base["B/S"]=="B"].groupby("timestamp")["price"].mean()
        self.timestamp_base["average_ask_price"] = self.base[self.base["B/S"]=="S"].groupby("timestamp")["price"].mean()
        self.timestamp_base["mid_price"] = self.timestamp_base[["average_bid_price","average_ask_price"]].mean(axis=1)
        
    def __timestamp_vol(self):
        # Ask prices volatility by timestamp
        self.timestamp_base["B_vol"] = self.base[self.base["B/S"]=="B"].groupby("timestamp")["price"].std()
        self.timestamp_base["S_vol"] = self.base[self.base["B/S"]=="S"].groupby("timestamp")["price"].std()

    def __price_moving_average(self,n):
        # Ratio volume sum by mid price
        self.timestamp_base["volume_by_price_B"] = self.timestamp_base["sum_volume_B"] / self.timestamp_base["average_bid_price"]
        self.timestamp_base["volume_by_price_S"] = self.timestamp_base["sum_volume_S"] / self.timestamp_base["average_ask_price"]
        self.timestamp_base["volume_by_price"] = self.timestamp_base["sum_volume"] / self.timestamp_base["mid_price"]
        self.timestamp_base["mid_MA_"+str(n)] = self.timestamp_base["volume_by_price"].rolling(window=n).mean()
            
    def __bollinger_bands(self,n,m):
        self.timestamp_base['UpperBand'] = self.timestamp_base['mid_price'].rolling(n).mean() + self.timestamp_base['mid_price'].rolling(n).std() * m
        self.timestamp_base['LowerBand'] = self.timestamp_base['mid_price'].rolling(n).mean() - self.timestamp_base['mid_price'].rolling(n).std() * m

    
    def __remove_old_features(self):
        del self.timestamp_base['sum_volume_B']
        del self.timestamp_base['sum_volume_S']
        del self.timestamp_base['average_bid_price']
        del self.timestamp_base['average_ask_price']
    
    def save_features_as_csv(self,path):
        self.timestamp_base['curr_pair'] = self.curr_pair
        self.timestamp_base.to_csv(path+self.curr_pair.replace("/", "_")+'.csv')
    
    def __fill_na(self):
        for field in ["volume_by_price_B", "volume_by_price_S","B_vol","S_vol"]:
            self.timestamp_base[field].fillna(method="ffill", inplace=True)


def main():
    path = r"C:\Users\MV\Desktop\Valko\HighFreqMomentumBacktester\NeuralNetwork\\"
    file_name = "livefix-log-18Jan-09-52-10-774"
    headers = ["ORDERTYPE","ORDER ID","curr_pair","LOCAL TIMESTAMP", "EXCHANGE TIMESTAMP", "volume","MINQTY","LOTSIZE","price","B/S","SCOPE"]
    dataset_initial = pd.read_csv(path+file_name+".csv", sep=';',names=headers).dropna()
    # Le dropna permet de supprimer les tickets annulés (C)
    dataset_initial = reduce_mem_usage(dataset_initial)
    FX_list = list(set(dataset_initial["curr_pair"]))
    FX_dict = {curr_pair: FX(curr_pair) for curr_pair in FX_list}

    print(FX_list)

    all_df = pd.DataFrame()
    for curr_pair in FX_list:
        FX_dict[curr_pair].fill_base(dataset_initial)
        FX_dict[curr_pair].features()
        # FX_dict[curr_pair].save_features_as_csv(path+'features\\')
        all_df = pd.concat([all_df,FX_dict[curr_pair].timestamp_base])

    del dataset_initial # we delete this database in order to free space in memory

    return FX_dict
