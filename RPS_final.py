import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pylab import mpl
mpl.rcParams['font.sans-serif']=['SimHei']
mpl.rcParams['axes.unicode_minus']=False

from IPython.display import set_matplotlib_formats
set_matplotlib_formats('retina')

'''
Goal Description: 

RPS(Relative Price Strength) Rating is an important measure first mentioned by 
William J. O’Neil. It is a measure of a stock’s price performance over the last
six months, compared to all stocks in selected database. The rating scale ranges
from 1(lowest) to 99 (highest), and 99 means the stock outperformed 99% of the 
stocks in the database, and 1 means the stock only outperformed 1% of the stocks 
in the database during the past 6 month. For this function, we would first 
calculate the RPS for each stock during time period, and plot the RPS for 
selected stock to visualize. 

For stock slection strategy here, we calculate the RPS for each stock each 
trading day, and count the frequency that each stock is ranked the top within 
the database. The higher the frequency number = the higher the RPS ranking, and 
so the better the stock is. 
'''

start = datetime(2018,1,5)
end = datetime(2019,3,9)

tickers = ['MMM','AXP','AAPL','BA','CAT','CVX','CSCO','KO','XOM','GS','HD','IBM',\
    'INTC','JNJ','JPM','MCD','MRK','MSFT','NKE','PFE','PG','UNH','VZ',\
    'V','WMT','WBA','DIS']

# Create an empty DataFrame to store the Stock prices 
StockPrices = pd.DataFrame()

def datetime_toString(dt):
    '''
    Change the datetime to string. 
    
    Input: 
        dt(datetime): the datetime we want to change 
    Output:
        a string
    '''
    return dt.strftime("%Y-%m-%d-%H")


def get_stock_prices(start, end, tickers) -> pd.DataFrame:
    '''
    Store the stock prices and stock returns in an empty DataFrame 

    Inputs: 
        start: start date of the data
        end: end date of the data
    Outputs: 
        stock_returns(DataFrame): record the stock returns on daily basis
    '''
    # Create an empty DataFrame to store the Stock prices 
    stock_prices = pd.DataFrame()
    num_stock = len(tickers)

    # get the storkprices data
    for ticker in tickers:
        download = yf.download(ticker, start, end)
        stock_prices[ticker] = download['Adj Close']
        
    data = stock_prices.dropna()
    return data 

data = get_stock_prices(start, end, tickers)


def cal_ret(df,w):
    '''
    Calculate the w days rolling return rate for each stock
    
    Input:
        df(DataFrame): the DataFrame that stores the stock prices 
        w: the w days rolling we want to see 
    '''
    df=df/df.shift(w)-1
    return df.iloc[w:,:].fillna(0)

ret120=cal_ret(data,w=120)


def get_RPS(ser):
    '''
    Calculate the corresponding RPS of each stock. 
    '''
    df=pd.DataFrame(ser.sort_values(ascending=False))
    df['n']=range(1,len(df)+1)
    df['rps']=(1-df['n']/len(df))*100
    return df


def all_RPS(data):
    '''
    Calculate the RPS for each trading day if all stocks are rolling w days 
    '''
    dates=(data.index).strftime('%Y%m%d')
    RPS={}
    for i in range(len(data)):
        RPS[dates[i]]=pd.DataFrame(get_RPS(data.iloc[i]).values,\
                                   columns=['Return Rate','rank','RPS'],\
                                   index=get_RPS(data.iloc[i]).index)  
    return RPS  

rps120=all_RPS(ret120)

#Consntruct an empty DataFrame that is based on previous return rates
df_new=pd.DataFrame(np.NaN,columns=ret120.columns,index=ret120.index)

for date in df_new.index:
    date=date.strftime('%Y%m%d')
    d=rps120[date]
    for c in d.index:
        df_new.loc[date,c]=d.loc[c,'RPS']

def plot_rps(stock):
    '''
    Plot two graphs to help visualize the results. 
    Plot 1 shows the Stock Price Trend of the chosen stock with respect to datas.
    Plot 2 shows the Relative RPS Strength of the chosen stock with respect to 
    dates. 
    
    Input: 
        stock(str): the interested stock ticker 
    Output:
        two plots
    '''
    plt.subplot(211)
    data[stock][120:].plot(figsize=(16,12),color='m')
    plt.yticks(fontsize=12)
    plt.xticks([])  
    plt.gca().spines['right']
    plt.gca().spines['top']
    plt.title(stock+'Stock Price Trend')

    plt.subplot(212)
    df_new[stock].plot(figsize=(16,12),color='c')
    my_ticks = pd.date_range(datetime_toString(start),datetime_toString(end),freq='m')
    plt.xticks(my_ticks,fontsize=12)
    plt.yticks(fontsize=12) 
    plt.gca().spines['right']

    plt.gca().spines['top']
    plt.title(stock+'RPS Relative Strength')
    
    plt.show()

def find_best_behaving_stocks_based_on_rps(df):
    '''
    Count the frequency of each stock being ranked with the highest RPS during 
    the time period. The stock that is most frequently ranked with highest RPS 
    means best better performance and thus great potentials. 
    
    Input:
        df(DataFrame): the DataFrame that stored the calcualted RPS
    Output: 
        df_count(DataFrame): the Dataframes that stored the tickers of stocks 
            and corresponding frequence
    '''
    #Find the corresponding col name of the max value of each row
    #create a new col called max_idx, and save the value 
    df['max_idx'] = df.idxmax(axis=1) 
    #Find the max value of each row
    #create a new col called max_val, and save the value 
    df['max_val']= df.max(axis=1) # Find that max value in each row 
    df_count = df.max_idx.value_counts()  
    print(df_count)
   
#%%



find_best_behaving_stocks_based_on_rps(df_new)
plot_rps('MMM')