import numpy as np 
import pandas as pd
import os
import copy
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import math
import statistics
from statistics import median 
import copy
import os
from sklearn import preprocessing
import statsmodels.api as sm
from collections import OrderedDict
from finance_byu.fama_macbeth import fama_macbeth, fama_macbeth_parallel, fm_summary, fama_macbeth_numba
import time
import yfinance as yf
from datetime import datetime

start = datetime(2013,1,1)
end = datetime(2019,12,31)

def output(number_stock,risk_free,start_year,end_year):
    '''
    Plot a graph of factor analysis model selection, comparing with index benchmark
        data

    Inputs:
        number_stock: (int) the number of stocks that investors can choose
            to invest each month
        risk_free: (float) investors can set risk-free rate by themselves
        start_year: (string) the investment starting time (must start on the 
            irst day of the month)
        end_year: (string) the investment ending time (must end on the last 
            day of the month)

    Return:
        a matplotlib graph showing comparison between index benchmark and 
            the strategy, and various ratios
    '''
    # Dow Jones Stocks with their corresponding industry

    stocks = {"MMM":"Conglomerate","AXP":"Financial Services","AMGN":"Pharmaceutical","AAPL":"Information Technology",\
            "BA":"Aerospace and Defense","CAT":"Constrution and Mining","CVX":"Petroleum","CSCO":"Information Technology",\
            "KO":"Food","DOW":"Chemical","GS":"Financial Services","HD":"Retailing","Hon":"Conglomerate","IBM":"Information Technology",\
            "INTC":"Information Technology","JNJ":"Pharmaceutical","JPM":"Financial Service","MCD":"Food","MRK":"Pharmaceutical",\
            "MSFT":"Inormation Tehcnology","NKE":"Apparel","PG":"Fast-moving Consumer Goods","CRM":"Information Technology",\
            "TRV":"Financial Services","UNH":"Managed Healthcare","VZ":"Telecommunication","V":"Financial Services",\
            "WBA":"Retailing","WMT":"Retailing","DIS":"Broadcasting and Entertainment"}


    # get a combined df for all stocks per trading day
    trading_df_daily = get_price_yf(stocks)
    trading_df_daily ["Date"]= trading_df_daily["Date"].\
                               apply(lambda x: x.strftime('%Y-%m-%d'))
    
    # get unique trading date
    trading_date = np.unique(trading_df_daily.Date)
    trading_date = sorted(trading_date)
    trading_date = pd.DataFrame({"Date":trading_date})
    trading_date["month"] = trading_date.Date.apply(lambda x:x[:7])
    last_trading_date_per_month = trading_date.Date.groupby(trading_date.month)\
                                 .apply(lambda x:x.sort_values().iloc[-1])
    last_trading_date_per_month.index = np.arange(len(last_trading_date_per_month))
    last_per_month = last_trading_date_per_month.to_list()

    # get account data from the file
    file_names_orig = os.listdir("crawler/data")
    file_names = copy.deepcopy(file_names_orig)
    # get quarterly account data df
    account_df = get_account_data(file_names)
    account_df = account_df.rename(columns = {"index":"Date"})
    # merge both trading data and account factor
    total_df_month = merge_account_trading(trading_df_daily,account_df,last_per_month)
    total_df_month = insert_industry(total_df_month,stocks)
    # get all factors
    total_factors = factor_name(total_df_month)
    # pre-process data
    total_df_month = MAD_process(total_df_month)
    total_df_month = fill_value(total_df_month, total_factors)
    # if there is still nan, we fill it with 0
    total_df_month = total_df_month.fillna(0)
    
    # use Z-score to normalize data as a whole
    Data = total_df_month.iloc[:,6:-1]
    mean = Data.mean()
    std = Data.std()
    Datafinal = (Data-mean)/std

    # get monthly-rtn
    total_df_month_rtn = get_monthly_rtn(total_df_month)
    monthly_next_rtn = get_monthly_next_rtn(total_df_month_rtn)

    Datafinal = pd.concat([Datafinal, monthly_next_rtn["monthly_next_rtn"]], axis = 1)
    df_IC = get_all_ic(Datafinal)
    # filter factors with IC_mean > 0.1, IC_abs_mean > 0.1, IC_greater_zero > 0.5, IC_ir > 0.2
    df_IC_filter = df_IC.loc[(df_IC["IC_mean"]>0.1) & (df_IC["IC_abs_mean"]>0.1) &\
                             (df_IC["IC_greater_zero"]>0.5) & (df_IC["IC_ir"]>0.2),\
                             ["Factor_name"]]
    
    Datafinal = Datafinal.fillna(0)
    Datafinal = Datafinal.reset_index()
    # multi-factor model
    final_factor, check_exp_mean = fama_mac(df_IC_filter, Datafinal)
    final_factor = final_factor[1:]
    # test multi-factor model
    test_df = Datafinal.loc[(Datafinal["Date"]>=start_year) & (Datafinal["Date"]<= end_year)]
    
    model = final_model(test_df,final_factor, check_exp_mean)
    # pick n highest exp-return stock for each month
    stock_pick = model.groupby("Date").apply(lambda x: x.exp_return.nlargest(number_stock))
    index = stock_pick.reset_index().level_1
    # get the final stock-pick for all months
    stock_pick = Datafinal.iloc[index,:]

    # get the df for expected return if investing 1 dollar at the begining
    net_val = net_value(stock_pick, number_stock)
    # get dow jones industrial average data
    dj_industrial = yf.download("^DJI", start, end).reset_index()
    dj_industrial ["Date"]= dj_industrial["Date"].apply(lambda x: x.strftime('%Y-%m-%d'))
    month_date_test = last_trading_date_per_month.loc[\
                        (last_trading_date_per_month >=start_year) &\
                        (last_trading_date_per_month <= end_year)]

    dj_industrial_new = pd.merge(dj_industrial, month_date_test, how = "inner")
    dj_industrial_new['ret_val'] = dj_industrial_new['Close']/\
                                  (dj_industrial_new['Close'].iloc[0])

    overall_return = (net_val.iloc[(net_val.shape[0]-1),1])**1/(trading_date.shape[0]/252)
    annualized_return = annual_return(net_val, trading_date, start_year, end_year)
    annualized_volatility = annual_volatility(net_val)
    sharp_ratio = get_sharp_ratio(annualized_return, annualized_volatility, risk_free)
    maxdrawdown = get_maxdrawdown(net_val)

   # plot the graph
    f, ax = plt.subplots(figsize=(18, 11))
    plt.plot(net_val['Date'],net_val['net_value'],label='Stock Picking Strategy')
    plt.plot(net_val['Date'],dj_industrial_new['ret_val'], label='DJIA Benchmark')
    plt.title(start_year +" ---- " + end_year + " Time Frame")
    plt.xlabel('Year')
    plt.ylabel('Return')
    textstr = '\n'.join((
        'Overall Return Rate = {:.2%}'.format(overall_return, ),
        'Annualized Return = {:.2%}'.format(annualized_return, ),
        'Annualized Volatility = {:.2%}'.format(annualized_volatility, ),
        'Sharp Ratio = {:.2%}'.format(sharp_ratio, ),
        'MaxDrawdown = {:.2%}'.format(maxdrawdown, )))
    plt.legend(bbox_to_anchor=(1, 1.1), loc='upper right', borderaxespad=0.)

    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
            verticalalignment='top', bbox=props)
    plt.gcf().autofmt_xdate()
    plt.show()



def get_price_yf(stocks):
    '''
    Use yfinance package to retrieve daily trading data of all 30 stocks

    Inputs: 
        stocks: (dict) a dictionary of stocks with corresponding industries

    Returns: 
        rading_df_daily (dataframe) 30 daily trading dataframe merged together
    '''
    master_list = []
    for stock in stocks:
        df = yf.download(stock, start, end).reset_index()
        df = df.assign(Name=stock)
        master_list.append(df)
    trading_df_daily = pd.concat(master_list, ignore_index=True)
    return trading_df_daily


def get_account_data(file_names):
    '''
    Make an integrated account data dataframe from all 30 stocks' ratio reports

    Inputs:
        file_names: (list) a list of file names extracted from local path
    
    Returns: n integrated account data dataframe (dataframe)
    '''
    factor = pd.DataFrame()
    for i in file_names:
        df = pd.read_csv("crawler/data/" + i)
        # regard all non-number entries as Nan
        df = df.applymap(lambda x: np.nan if x == 'NM' or x == 'NA' else x)
        factor = factor.append(df)
    factor = factor.rename(columns = {"Company":"Name","Quarter":"Date"})
    return factor 

def merge_account_trading(trading_df,account_df,last_per_month):
    '''
    Merge account data and daily trading data 

    Inputs:
        trading_df: (dataframe) 
        account_df: (dataframe) 
        last_per_month: (list) 

    Returns: total_df (dataframe) the merged dataframe
    '''
    total_df = pd.merge(trading_df,account_df,how = "left")
    total_df = total_df.groupby("Name").apply(lambda x:x.sort_values("Date").fillna(method = "ffill"))
    total_df.index = np.arange(len(total_df))
    merge_last_month = total_df.Date.apply(lambda x: x in last_per_month)
    total_df = total_df[merge_last_month]
    return total_df

def insert_industry(total_df_month,stocks):
    '''
    Add a column of corresponding industry to the merged dataframe

    Inputs: 
        total_df_month: (dataframe) merged dataframe 
        stocks: (dict) a dictionary of stocks with corresponding industries
    
    Returns: total_df_month: (dataframe) dataframe with the column added
    '''
    lst  = []
    for i in range(total_df_month.shape[0]):
        for stock in stocks:
            if total_df_month.iloc[i]['Name'] == stock:
                lst.append(stocks[stock])
    total_df_month['Industry'] = np.array(lst)
    total_df_month.set_index(["Date","Name"], inplace = True)
    return total_df_month


def factor_name(df):
    '''
    Find all factor names

    Inputs:
        df: (dataframe) a dataframe

    Return: factor_names: (list) a list of factor names
    '''
    columns = df.columns.tolist()
    factor_names = columns[6:-1]
    return factor_names

def MAD_process(df): 
    '''
    Use the median absolute deviation data processing method

    Inputs: 
        df: (dataframe) the dataframe for preprocessing

    Returns: data (dataframe) the dataframe after processing
    '''
    data = df.copy()
    n = 3*1.4826
    factor_range = range(6,len(data.columns)-1)
    for i in factor_range:
        factor_series = data.iloc[:,i]
        # find the median of the whole column
        median = factor_series.quantile(0.5)
        new_median = ((factor_series - median).abs()).quantile(0.50)
        max_range = median + n*new_median
        min_range = median - n*new_median
        # if the value is greater or smaller than the max/min range, fix it
        data.iloc[:,i] = np.clip(factor_series,min_range,max_range)
    return data


def fill_value(df, total_factors):
    '''
    Fill missing value by using the industry average value for each factor

    Inputs:
        df: (dataframe) the dataframe for preprocessing
        total_factors: (list) a list of factors 

    Returns: df: (dataframe) the dataframe after processing
    '''
    for i, _ in enumerate(total_factors):
        # get the industry mean
        industry_value = df[total_factors[i]].groupby(df['Industry']).mean()  

        # replace missing value with industry mean
        for j in range(df.shape[0]):
            if np.isnan(df.iloc[j,i]) == True:
                # fill the gap
                df.iloc[j,i] = industry_value[df.iloc[j,-1]] 
    return df

def get_rtn(x):
    '''
    get percentage value change 

    Inputs:
        x: certain row

    Returns: processed x
    '''
    price, date= "Close", "Date"
    x = copy.deepcopy(x[[price,date]])
    x = x.sort_values(date)
    # get the percentage change 
    x["monthly_rtn"] = x[price].pct_change()
    return x

def get_monthly_rtn(total_df_month):
    '''
    Get monthly return

    Inputs:
        total_df_month: (dataframe) dataframe after preprocessing

    Retrurns: total_df_month (dataframe) dataframe after adding monthly return column
    '''
    total_df_month = total_df_month.reset_index()
    monthly_rtn = total_df_month.groupby("Name").apply(lambda x: get_rtn(x))
    monthly_rtn = monthly_rtn.reset_index()
    monthly_rtn = monthly_rtn[["Name","Date","monthly_rtn"]]
    total_df_month = pd.merge(total_df_month,monthly_rtn,how = "left")
    return total_df_month

def get_next_rtn(x):
    '''
    get percentage value change 

    Inputs:
        x: certain row

    Returns: processed x
    '''
    rtn, date_name = "monthly_rtn", "Date"
    x = copy.deepcopy(x[[date_name,rtn]])
    x = x.sort_values(date_name)
    x['monthly_next_rtn'] = x[rtn].shift(periods = -1)
    return x

def get_monthly_next_rtn(total_df_month):
    '''
    Get next monthly return

    Inputs:
        total_df_month: (dataframe) dataframe after preprocessing

    Retrurns: total_df_month (dataframe) dataframe after adding next monthly return column
    '''
    monthly_next_rtn = total_df_month.groupby("Name").apply(lambda x: get_next_rtn(x))
    monthly_next_rtn = monthly_next_rtn.reset_index()
    monthly_next_rtn = monthly_next_rtn[["Name","Date","monthly_next_rtn"]]
    monthly_next_rtn = monthly_next_rtn.set_index(["Date","Name"])

    return monthly_next_rtn


def get_ICs(df,factor,rtn,date):
    '''
    Get Rank IC for all factors in order to filter factors

    Inputs:
        df: (dataframe) the dataframe from previous processing
        factor: (lisst) a list of factors
        rtn: (string) "monthly_next_rtn"
        date: (string) "Date"

    Returns: final_dic (dict) a ditionary of various IC related ratios
    '''
    df = df.reset_index() # important
    IC = []
    # get unique datas
    dates = np.unique(df[date])
    # sort dates
    dates = sorted(dates)
    for date in dates: 
        corr_df = df[df.Date == date]
        # get bool to check nan
        corr_x = pd.notna(corr_df[factor])
        corr_y = pd.notna(corr_df[rtn])
        # get bool for x&y not nan together
        corr_bool = corr_x & corr_y
        if sum(corr_bool) > 0:
            corr_df = corr_df[corr_bool]
            IC.append(np.corrcoef(corr_df[factor],corr_df[rtn])[0,1])
    
    # get columns
    IC = pd.Series(IC)
    IC_abs_mean = np.mean(abs(IC))
    IC_mean = np.mean(IC)
    IC_std = np.std(IC)
    # the probability that IC is greater than 0
    IC_greater_zero = (sum(IC > 0)/len(IC))
    # IC's IR rate
    IC_ir = IC_mean/IC_std
    # make the final dictionary
    final_dic = {"Factor_name":factor,"IC_mean":IC_mean,"IC_abs_mean":IC_abs_mean,"IC_std":IC_std,\
            "IC_greater_zero":IC_greater_zero,"IC_ir":IC_ir}
    return final_dic

def get_all_ic(df):
    '''
    Make a dataframe of all IC related ratios

    Inputs:
        df:(dataframe) integrated dataframe from previous processing

    Return: df_IC: (dataframe) a dataframe of all IC related ratios
    '''
    df_IC = pd.DataFrame(columns=["Factor_name", "IC_mean","IC_abs_mean","IC_std","IC_greater_zero","IC_ir"])
    # get only factors
    factors = df.columns[2:-2]
    for factor_name in factors:
        # make an IC dataframe
        df_IC = df_IC.append(get_ICs(df,factor_name,"monthly_next_rtn","Date"), ignore_index=True)
    return df_IC


def fama_mac(df_IC_filter, Datafinal):
    '''
    Train the data using fama-macbeth model

    Inputs:
        df_IC_filter: (list) a list of filtered factors from IC processing
        Datafinal: (dataframe) integrated dataframe from previous processing

    Returns: 
        final_factor: (list) the final factors chosen by the model with 
            significane p-values
        check_exp_mean: (dataframe) exposure coefficients for chosen factors
    '''
    filter_factor = df_IC_filter.Factor_name.to_list()
    df_regress = Datafinal.set_index(["Date","Name","monthly_next_rtn"]).loc[:,filter_factor].reset_index()
    result = fama_macbeth(df_regress,'Date','monthly_next_rtn',filter_factor,intercept=True)
    # get exposure summary
    summary = fm_summary(result,pvalues=True)
    final_factor = summary[summary.pval < 0.1].index.to_list()
    final_factor = final_factor[1:]
    # after getting final factor, we need to use these factors to check back
    df_check = Datafinal.set_index(["Date","Name","monthly_next_rtn"]).loc[:,final_factor].reset_index()
    result_check = fama_macbeth(df_check,'Date','monthly_next_rtn',final_factor,intercept=True)
    # get coefficients for factors
    check_exp_mean = fm_summary(result_check,pvalues=True).iloc[:,0]
    return final_factor, check_exp_mean


def final_model(df,final_factor,check_exp_mean):
    '''
    Get expected return for each stock in each month

    Inputs:
        df: (dataframe) integrated dataframe from previous processing
        final_factor: (list) the final factors chosen by the model with 
            significane p-values
        check_exp_mean: (dataframe) exposure coefficients for chosen factors

    Returns: df (dataframe): df with expected return column added for each stock 
        in each month
    '''
    df_test = df[final_factor].copy()
    exp_return = []
    for i in range(df_test.shape[0]):
        exp_return.append(sum(df_test.iloc[i,:] * check_exp_mean[1:]+ check_exp_mean[0]))
    df['exp_return'] = exp_return
    return df


# simulate the time when you wanna invest 1 dollar
def net_value(stock_pick, number_stock):
    '''
    Get sum of total return from each factor for stocks picked each month

    Inputs:
        stock_pick: (dataframe) stocks with number_stock highest return of 
            stocks picked every month
        number_stock: (int) the number of stocks that investors can choose 
            to invest each month

    Returns: net_val: (dataframe) dataframe with date and its 
        corresponding net return
    '''
    date_pick = np.unique(stock_pick.Date)
    net_val = pd.DataFrame({"Date":date_pick,"net_value": 0})
    # we set the net val of the first row and first column to be the invest
    net_val.iloc[0,1] = 1
    #set equal weight for all the stock
    weight = np.array([1/number_stock]*number_stock)
    for i, j in enumerate(date_pick):
        temp = sum(weight * (1+ stock_pick[stock_pick.Date==j].monthly_next_rtn))
        net_val.iloc[i,1] = temp
        weight = np.array([temp/number_stock]*number_stock)
    return net_val


# annualized rate of return
def annual_return(net_val, trading_date, start_year, end_year):
    '''
    Get the annualized rate of return for the stocks picked these months

    Inputs:
        net_val: (dataframe) dataframe with date and its 
            corresponding net return
        trading_date: (dataframe) dataframe with all trading date 
        start_year: (string) the investment starting time (must start on the 
            irst day of the month)
        end_year: (string) the investment ending time (must end on the last 
            day of the month)
    
    Returns: annual_rate: (float) annualized rate of return for the stocks 
        picked these months
    '''
    total_trading_day = int(np.busday_count(start_year, end_year))
    annual_trading_day = 252
    initial = net_val.iloc[0,1]
    final = net_val.iloc[(net_val.shape[0]-1),1]
    annual_rate = pow((final / initial), (annual_trading_day / total_trading_day)) - 1
    return annual_rate

#  Annualized volatility
def annual_volatility(net_val):
    '''
    Get the annualized rate of volatility for the stocks picked these months

    Inputs:
        net_val: (dataframe) dataframe with date and its 
            corresponding net return
    
    Returns: annual_rate: (float) annualized rate of volatility for the stocks 
        picked these months
    '''
    annual_trading_day = 252
    pct_change = net_val["net_value"].pct_change()
    volatility = pct_change.std()*pow(annual_trading_day,0.5)
    return volatility

# sharp ratio
def get_sharp_ratio(annual_rate, volatility, risk_free):
    '''
    Get the sharp ratio for the stocks picked these months

    Inputs:
        annual_rate: (float) annualized rate of return for the stocks 
            picked these months
        volatility: (float) annualized rate of volatility for the stocks 
            picked these months
        risk_free: (float) investors can set risk-free rate by themselves

    Returns: sharp ratio: (float) sharp ratio for the stocks 
        picked these months
    '''
    sharpe_ratio = (annual_rate - risk_free) / volatility
    return sharpe_ratio

# Max Drawdown
def get_maxdrawdown(net_val):
    '''
    Get the Max Drawdown rate for the stocks picked these months

    Inputs:
        net_val: (dataframe) dataframe with date and its 
            corresponding net return
    
    Returns: annual_rate: (float) Max Drawdown rate for the stocks 
        picked these months
    '''
    lst = net_val["net_value"].to_list()
    # make net value accumulated single increasing 
    accum = np.maximum.accumulate(lst)
    # position of min using difference between accum_max
    i = np.argmax((accum - lst) / accum)
    # position of max in relation to local min
    j = np.argmax(lst[:i])
    maxi = (lst[j]-lst[i])/(lst[j])
    return maxi




if __name__ == '__main__':
    number_stock = 5
    risk_free = 0.03
    start_year = "2017-04-01"
    end_year = "2019-12-31"
    output(number_stock,risk_free,start_year,end_year)
