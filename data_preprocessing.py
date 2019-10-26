#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader


# In[33]:


class StockDataset(Dataset):
    def __init__(self,start_year = 2002,end_year = 2017,window_size = 12):
        self.start_year = start_year
        self.end_year = end_year
        self.window_size = window_size
        self.data = self._load_data(self.start_year,self.end_year)
        self.x,self.y = self._create_features_and_labels(self.data,self.window_size)
        
    def _load_data(self,start_year,end_year):
        col_name = ["date","time","open","high","low","close","volume"]
        df_full = pd.DataFrame(columns = col_name)
        for root, dirs, files in os.walk("/stock_prediction/stock_prediction/EURGBP/"):
            for file in files:
                year = int(file.split(".")[0].split("_")[-1])
                if (file.endswith(".csv") and year >= start_year 
                and year <= end_year):
                    df = pd.read_csv(os.path.join(root,file),names = col_name)
                    df_full = pd.concat([df_full,df])
                    print("Stock data of year {0} loaded".format(str(year)))
        df_full.reset_index(inplace = True)
        return df_full
    
    
    def _create_features_and_labels(self,df,window_size):
        n = df.shape[0]
        x_array = np.empty(shape = (n - window_size,window_size))
        y_array = np.empty(shape = (n - window_size,1))
        close_index = df.close
        for i in tqdm(range(n-window_size),desc =  "Creating Features and Labels..."):
            x_array[i] = close_index.iloc[i:i+window_size].values
            y_array[i] = close_index.iloc[i + window_size]
        return x_array,y_array
    
    def __getitem__(self,idx):
        return self.x[idx],self.y[idx]
    
    def __len__(self):
        return self.x.shape[0]


# In[34]:


data = StockDataset(start_year=2017)


# In[53]:


data_ld = DataLoader(data,batch_size=32,shuffle=True,num_workers=2)


# In[54]:


next(iter(data_ld))[1].shape

