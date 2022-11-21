import tkinter as tk
import tkinter.font as font
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import seaborn as sns
import statsmodels.api as sm
import time
import quandl
import os
import re
import io
import quandl
import datetime
import copy
import statistics
from typing import List
from collections import OrderedDict
from IPython.display import set_matplotlib_formats
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from tkinter import ttk
from tkinter import messagebox as msgbox
from PIL import Image,ImageTk
from scipy import stats
from finance_byu.fama_macbeth import \
    fama_macbeth, fama_macbeth_parallel, fm_summary, fama_macbeth_numba
from statistics import median 


##  Part 1: Major working functions  ##########################################
#
#   The first part of the code consists of major functions that are linked
#   to the buttons on the user's interface, and are called when buttons 
#   are clicked.
#
#   Line 49-217: Functions related to predicting stock price using LSTM.
#   
#   Line 222-418: Functions related to portfolio structure analysis
#
#   Line 423-982: Function related to Multi factor model simulation

##  Major function 1: Stock Price Prediction  #################################

def predict():
    '''
    Action function of the predict button. Takes the ticker name and date
    entered by the user, predict the price of the stock and plot the graph.
    If the date input is earlier than the current date, the actual prices 
    will also be plotted as a comparison to the predicted prices.
    '''
    try:
        # Download data from 2017-1-1 to today. 
        # Pre-process the scraped data and extract the closing prices.
        ticker = entry_ticker.get()
        start = entry_date.get()
        if start == None:
            msgbox.showinfo('Warning', 'Invalid date!')
        today = datetime.datetime.today()
        start_datetime = datetime.datetime.strptime(start, '%Y-%m-%d')
        df = yf.download(ticker, datetime.datetime(2017,1,1), today)
        df.index = df.index.strftime("%Y-%m-%d") 
        data = df[['Close']]
        data.columns = ['Price']

        # Pre-set the length of each training input
        T = 60 

        # If the user entered a date in the past, we need to make
        # prediction based on the data before the date entered, and 
        # then compare the result with the actual price.
        if today >= start_datetime:
            # Find the numerical index of the starting date
            if start not in list(data.index):
                msgbox.showinfo('Warning', 'Date not found!')
            start_index = list(data.index).index(start)

            # Construct training data and testing data
            dataset = data.values
            train = dataset[0:start_index, :]
            valid = dataset[start_index:, :]

            # Rescale datapoints between 0 & 1
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(dataset)

            # Collect features in the trainin data
            x_train, y_train = [], []
            for i in range(T, len(train)):
                x_train.append(scaled_data[i - T:i, 0])
                y_train.append(scaled_data[i, 0])
            x_train, y_train = np.array(x_train), np.array(y_train)
            x_train = np.reshape(x_train, \
                (x_train.shape[0], x_train.shape[1], 1))

            # Create model
            model = Sequential()
            model.add(LSTM(units=50, return_sequences=True, \
                input_shape=(x_train.shape[1], 1)))
            model.add(LSTM(units=50))
            model.add(Dense(1))

            model.compile(loss='mean_squared_error', optimizer='adam')
            model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2)

            # Perform prediction
            inputs = data[len(data) - len(valid) - T:].values
            inputs = inputs.reshape(-1, 1)
            inputs = scaler.transform(inputs)

            X_test = []
            for i in range(T, inputs.shape[0]):
                X_test.append(inputs[i - T:i, 0])
            X_test = np.array(X_test)
            print(X_test.shape)
            X_test = np.reshape(X_test, \
                (X_test.shape[0], X_test.shape[1], 1))
            closing_price = model.predict(X_test)
            closing_price = scaler.inverse_transform(closing_price)

            train = data[0:start_index]
            valid = data[start_index:]
            valid['Predictions'] = closing_price
            combined = train.append(valid)

            # Draw the graph on the canvas
            new_window = tk.Tk()
            new_window.title('Prediction vs Actual')
            lf = ttk.Labelframe(new_window)
            lf.grid(row=0, column=0, sticky='nwes', padx=3, pady=3)
            fig = Figure(figsize=(10,8), dpi=100)
            ax = fig.add_subplot(111)
            combined.plot(ax=ax)
            canvas = FigureCanvasTkAgg(fig, master=lf)
            canvas.draw()
            canvas.get_tk_widget().pack(side='top', fill='both')

        # This is the case when the user entered a date in the future         
        else:
            # Train the model using the entire dataset (from 2017-today)
            dataset = data.values
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(dataset)
            x_train, y_train = [], []
            for i in range(T, len(dataset)):
                x_train.append(scaled_data[i - T:i, 0])
                y_train.append(scaled_data[i, 0])
            x_train, y_train = np.array(x_train), np.array(y_train)
            x_train = np.reshape(x_train, \
                (x_train.shape[0], x_train.shape[1], 1))

            model = Sequential()
            model.add(LSTM(units=50, return_sequences=True, \
                input_shape=(x_train.shape[1], 1)))
            model.add(LSTM(units=50))
            model.add(Dense(1))

            model.compile(loss='mean_squared_error', optimizer='adam')
            model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2)

            # Using the previous T days to predict the next day price,
            # then use the previous T-1 days + the latest result to predict the
            # price of the day after tomorrow
            n = (start_datetime - today).days
            i = 1
            last_input = dataset[-T:]
            last_input = scaler.transform(last_input)
            last_input = last_input.reshape(T,).tolist()
            final_output = []
            while i < n+2:
                if len(dataset) < T:
                    return 'Not enough data under this T'
                else:
                    print("{} day input {}".format(i,last_input))
                    x = np.array(last_input)
                    x = x.reshape(1, T, 1)
                    yhat = model.predict(x, verbose=0)
                    print("{} day output {}".format(i,yhat))
                    final_output += yhat[0].tolist()
                    last_input += yhat[0].tolist()
                    last_input = last_input[1:]
                    i += 1
            final = np.array(final_output)
            final = final.reshape(1,final.shape[0],)
            final = scaler.inverse_transform(final)
            next_n = final[0].tolist()
            next_n_df = pd.DataFrame(next_n)
            next_n_df.columns=['Price']
            
            # Draw the prediction graph on canvas
            new_window = tk.Tk()
            new_window.title('Prediction of price changes in the next {} days'.format(n))
            lf = ttk.Labelframe(new_window)
            lf.grid(row=0, column=0, sticky='nwes', padx=3, pady=3)
            fig = Figure(figsize=(10,8), dpi=100)
            ax = fig.add_subplot(111)
            next_n_df.plot(ax=ax)
            canvas = FigureCanvasTkAgg(fig, master=lf)
            canvas.draw()
            canvas.get_tk_widget().grid(row=0, column=0)
    # Deal with typical input errors        
    except AttributeError:
        msgbox.showerror('Error', 'Invalid ticker input, no such stock. Correct form example: AAPL')
    except ValueError:
        msgbox.showerror('Error', 'Date not found (invalid date input or the market was closed on that day). Correct form example: 2020-01-03, 2021-04-30')
    except:
        msgbox.showerror('Error', 'Invalid input!')

def pred_desc():
    '''
    Action function of the Description button of the prediction function. Execute upon clicking the button.
    '''
    msgbox.showinfo('Function description', 'This function predicts the price of the chosen stock at the chosen date using a simple LSTM model, and plot its graph. \n \n If the date input is earlier than the current date, the actual prices will also be plotted as a comparison to the predicted prices. \n \n To use it, just enter the stock and date, and click the predict button')

##  Major function 2: Portfolio Analysis  #####################################
# Portfolio Analysis - Covariance/Correlation Analysis

quandl.ApiConfig.api_key = "nmKWiBqHzQNqtUk97uAD"
set_matplotlib_formats('retina')

def get_stock_returns():
    '''
    Store the stock prices and stock returns in an empty DataFrame 

    Inputs: 
        start: start date of the data
        end: end date of the data
    Outputs: 
        stock_returns(DataFrame): record the stock returns on daily basis
    '''
    start = start_entry.get()
    start = datetime.datetime.strptime(start, '%Y-%m-%d')
    end = end_entry.get()
    end = datetime.datetime.strptime(end, '%Y-%m-%d')
    tickers = tickers_entry.get().strip().split(',')
    tickers = [x.strip() for x in tickers if x]
    # Create an empty DataFrame to store the Stock prices 
    # Create an empty DataFrame to store the Stock prices 
    stock_prices = pd.DataFrame()

    # get the storkprices data
    for ticker in tickers:
        data = yf.download(ticker, start, end)
        stock_prices[ticker] = data['Adj Close']

    stock_returns = stock_prices.pct_change().dropna()
    return stock_returns

def corr_and_cov_analysis():
    """
    Analize the correlation and covariance between interested stocks, to help 
    investors decide which stocks to be put into the portflio. This is because
    more diversifiable portfolio could potentially generate more protits with 
    lower risks. We generate a heatmap to visualize the results. 
    
    Input: 
        stock_returns(DataFrame): record the stock returns on daily basis    
    Output: 
        heatmap(graph): heatmaps of the correlation and covariance matrix
    """
    try:
        stock_returns = get_stock_returns()
        data1 = stock_returns.corr()
        data2 = stock_returns.cov() * 252 # to get the annualized covariance
        result = msgbox.askokcancel('Reminder', 'Invalid input may lead to empty graph, please make sure that: \n \n For the ticker input: \n 1. The ticker name is correct; \n 2. The ticker input is in the format of "APPL, TSLA, ..."; \n 3. The stock was in the market during the specified period of time; \n 4. WARNING. Due to the problem of the yfinance library, scraping APPL data may fail sometimes, so please avoid using APPL. \n \n For the date input \n 1. The date input is in the format of yyyy-mm-dd; \n 2. The market is open on that day; \n 3. The date input must be a date in the past; \n 4. The start date must be before the end date; \n 5. Both dates must exist at the same time.')
        if not result:
            return
        new_window = tk.Tk()
        new_window.title('Correlation and Covariance')
        new_window.geometry('%dx%d+%d+%d'%(1600, 650, sw/2-800, sh/2-350))
        lf = ttk.Labelframe(new_window)
        lf.grid(row=0, column=0, sticky='nwes', padx=3, pady=3)
        plt.close()
        fig = plt.figure(1, figsize=(16, 6)) 

        # Plot the heatmap for correlation
        ax1 = fig.add_subplot(1, 2, 1)
        sns.heatmap(data1, annot=True, cmap="YlGnBu")
        plt.xticks(rotation=90)
        plt.yticks(rotation=0) 
        plt.title("Correlation", fontsize =17)

        # Plot the heatmap for covariance
        ax2 = fig.add_subplot(1, 2, 2)
        sns.heatmap(data2, annot=True, cmap="BuPu")
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.title("Covariance", fontsize =17)

        canvas = FigureCanvasTkAgg(fig, master=lf)
        canvas.draw()
        canvas.get_tk_widget().pack(side='top', fill='both')

    except:
        msgbox.showerror('Warning', 'Invalid input! Please make sure that: \n \n For the ticker input: \n 1. The ticker name is correct; \n 2. The ticker input is in the format of "APPL, TSLA, ..."; \n 3. The stock was in the market during the specified period of time; \n 4. WARNING. Due to the problem of the yfinance library, scraping APPL data may fail sometimes, so please avoid using APPL. \n \n For the date input \n 1. The date input is in the format of yyyy-mm-dd; \n 2. The market is open on that day; \n 3. The date input must be a date in the past; \n 4. The start date must be before the end date; \n 5. Both dates must exist at the same time.')

def cov_desc():
    '''
    Action function of the Description button of the portfolio analysi. Execute upon clicking the button.
    '''
    msgbox.showinfo('Function description', 'The correlation and covariance analysis tool help you to better decide your portfolio composition (through improving diversification). \n \n To use this, just enter the ticker of stocks and time period (start + end time) you are interested in, and this function would generate the heatmaps for the input stocks on their corr and cov. \n \n As for the portfolio optimization function, it conducts portfolio optimization through GMV(min Risk) method and MSR(max Sharpe Ratio) method, relying on Markowitz model and Modern Portfolio Theory. \n \n To use this, just enter the ticker of stocks in your chosen portfolio and time period (start + end time) you are interested in, and this function would generate the optimal weight for each stock under both GMV and MSR optimal portfolios, marking these two portfolios on mean-variance graph, and then creating a plot to compare the behaviors of both GMV and MSR portfolios during this time period.')   
# Portfolio Analysis - GMV strategy vs Sharpe ratio strategy

def gmv_sr():
    """
    Generate 10000 random portoflios that are composed by investor's
    chosen stockes, but in different weight combinations. Sum of weight
    equals to 1. 

    Input:
        stock_returns(DataFrame): record the stock returns on daily basis
        num_stock(int): number of stocks
        n(int): number of simulations, 10000
    Output: 
        random_portfolios(DataFrame): records the weights combinations, 
        volatility, and return of each generated portfolio
    """
    risk_free = float(rf.get())
    stock_returns = get_stock_returns()
    tickers = tickers_entry.get().strip().split(',')
    tickers = [x.strip() for x in tickers if x]
    num_stock = len(tickers)
    n = 10000
    # create empty numpy to store the weight
    random_p = np.empty((n, num_stock+2))
    np.random.seed(12200)
    cov_mat_annual = stock_returns.cov() * 252 # annualized covariance matrix

    # simulate n random portoflio weight choices
    for i in range(n):
        # generate a random weight dataset that has sum = 1, store in numpy 
        random_weight = np.random.random(num_stock) / \
        np.sum(np.random.random(num_stock))
        random_p[i][:num_stock] = random_weight
        
        # Get the portfolio return daily, then take the avg
        # Calculate the annualized return rate, store in numpy
        mean_return = stock_returns.mul(random_weight, axis=1).sum(axis=1).mean() 
        annual_return = (1 + mean_return)**252 - 1
        random_p[i][num_stock] = annual_return

        # Calculate annualized volatility, which is SD, and store in numpy
        random_p[i][num_stock+1] = np.sqrt(np.dot(random_weight.T, 
                                        np.dot(cov_mat_annual, \
                                            random_weight)))
        
    random_portfolios = pd.DataFrame(random_p)
    random_portfolios.columns = [ticker + "_weight" for ticker in tickers] \
                                + ['Returns', 'Volatility']
    random_portfolios['Sharpe_Ratio'] = (random_portfolios.Returns - risk_free)\
                                / random_portfolios.Volatility
    
    # graph 1 content (ax1)
    min_risk = random_portfolios.Volatility.idxmin()
    GMV_weights = np.array(random_portfolios.iloc[min_risk, 0:num_stock])
    x_gmv = random_portfolios.loc[min_risk,'Volatility']
    y_gmv = random_portfolios.loc[min_risk,'Returns']    
    max_SR = random_portfolios.Sharpe_Ratio.idxmax()
    Sharpe_weights = np.array(random_portfolios.iloc[max_SR, 0:num_stock])
    x_sr = random_portfolios.loc[max_SR,'Volatility']
    y_sr = random_portfolios.loc[max_SR,'Returns']

    # graph 2 content (ax2)
    Sreturn = stock_returns.copy()
    stock_returns['Port_GMV'] = Sreturn.mul(GMV_weights, axis=1).sum(axis=1)
    stock_returns['Port_MSR'] = Sreturn.mul(Sharpe_weights, axis=1).sum(axis=1)

    new_window = tk.Tk()
    new_window.title('Optimal Portfolio Finder')
    new_window.geometry('%dx%d+%d+%d'%(1600, 800, sw/2-800, sh/2-400))

    lf = ttk.Labelframe(new_window)
    lf.grid(row=0, column=0, sticky='nwes', padx=3, pady=3)
    plt.close()
    fig = plt.figure(1, figsize=(16, 6)) 
    
    ax1 = fig.add_subplot(1, 2, 1)
    random_portfolios.plot('Volatility', 'Returns', kind='scatter', alpha=0.3, ax = ax1)
    plt.scatter(x_gmv, y_gmv, color='red')
    plt.annotate("GMV Point", (x_gmv, y_gmv))  
    plt.scatter(x_sr, y_sr, color='red')
    plt.annotate("SR Point", (x_sr, y_sr))
    plt.title('Return vs Risk plot of all investment opportunities')  

    ax2 = fig.add_subplot(1, 2, 2)
    for name in ['Port_GMV', 'Port_MSR']:
        df = stock_returns[name]
        CumReturns = ((df+1).cumprod()-1)
        CumReturns.plot(label=name)
    plt.title('GMV return vs SR return')
    plt.legend()
    
    canvas = FigureCanvasTkAgg(fig, master=lf)
    canvas.draw()
    canvas.get_tk_widget().pack(side='top', fill='both')
    
    fconfig = font.Font(size = 20, family = 'Times New Roman')
    weight_label = tk.Label(new_window, text = 'Optimal Allocation of capital among {}'.format(tickers_entry.get()), bg = 'grey', fg = 'white', font = fconfig)
    weight_label.grid(row=1, column = 0)
    gmv_label = tk.Label(new_window, text = 'GMV Portfolio (Mimimum risk): \n '+ str(GMV_weights), font = fconfig)
    gmv_label.grid(row=2, column = 0)
    sr_label = tk.Label(new_window, text = 'Tangency (SR) Portfolio (Best return vs risk balance): \n '+ str(Sharpe_weights), font = fconfig)
    sr_label.grid(row=3, column = 0)




        
##  Major function 3: Simulation of recurring Investment using using  a  ######
#   multi-factor model vs. benchmark  #########################################
start = datetime.datetime(2013,1,1)
end = datetime.datetime(2019,12,31)

def output_graph():
    '''
    The action function of the Model plot button. Plot a graph of factor 
    analysis model selection, comparing with index benchmark data
    '''    
    number_stock = int(num_stock.get())
    risk_free = float(risk_f.get())
    start_year = start_year_entry.get()
    end_year = end_year_entry.get()
    matched_y = re.match("[0-9]{4}-[0-9]{2}-[0-9]{2}",start_year)
    matched_e = re.match("[0-9]{4}-[0-9]{2}-[0-9]{2}",end_year)
    if not (number_stock != 0 and bool(matched_y) and bool(matched_e)):
        msgbox.showerror('Warning', 'Invalid input! Please make sure that: \n 1. The number of stocks is a non-zero integer; \n 2. The risk free rate should be a decimal number; \n 3. The start time should be the first day of a month between 2013-01-01 to 2019-12-31) \n 4. The end time should be the last day of a month between 2013-01-01 to 2019-12-31; \n 5. The start time should be earlier than the end time.')
        return 
    else:
        try:   
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
            file_names_orig = os.listdir("crawler/data_backup")
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

            new_window = tk.Tk()
            new_window.title('Correlation and Covariance')
            new_window.geometry('%dx%d+%d+%d'%(1400, 750, sw/2-700, sh/2-375))
            lf = ttk.Labelframe(new_window)
            lf.grid(row=0, column=0, sticky='nwes', padx=3, pady=3)
            plt.close()
            fig = plt.figure(1, figsize=(12,7))
            ax = fig.add_subplot(1, 1, 1)

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
                    
            canvas = FigureCanvasTkAgg(fig, master=lf)
            canvas.draw()
            canvas.get_tk_widget().pack(side='top', fill='both')
        except:
            msgbox.showerror('Warning', 'Invalid input! Please make sure that: \n 1. The number of stocks is a non-zero integer; \n 2. The risk free rate should be a decimal number; \n 3. The start time should be de the first day of a month between 2013-01-01 to 2019-12-31) \n 4. The end time should be the last day of a month between 2013-01-01 to 2019-12-31; \n 5. The start time should be earlier than the end time.')

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
        #df = df.assign(Industry = stocks[stock])
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
        df = pd.read_csv("crawler/data_backup/" + i)
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

def simulate_desc():
    '''
    Action function of the Description button of the simulation function. Execute upon clicking the button.
    '''
    msgbox.showinfo('Function description', 'This function uses Fama-Macbeth model to extract common factors significant for stock picking in order to generate the highest return. The theory is to invest 1 unit of dollar into the selected portfolio with equal weight at the begining of a month, sell the portfolio at then end of the month, and then repurchase another portfolio at the begining of the next month, similarly until the end of the selected time. \n \n The final outcome is a graph with the expected return trend line and the Dow Jones Industrial Average as the benchmark line over selected time line. The overall investment return, annualized return, annualized volatility, sharp ratio, as well as maximum drawdown will be displyed. \n \n The training model uses data from 2013-01-01 to 2019-12-31. \n \n To use this function, enter: \n 1. The number of stocks (n): the n highest-return stocks each month to be put in the portfolio; \n 2. The risk-free rate: the market risk-free rate that you want to use as a constant; \n 3. Start Year: the starting time of your investment (must select the begining of a random month between 2013-01-01 to 2019-12-31); \n 4. End Year: the ending time of your investment (must select the end of a random month between 2013-01-01 to 2019-12-31)')
##  Part 2: User's Interface construction #####################################
#
#   The second part of the code is mainly building the user's interface


# Initialize the main window
main_window = tk.Tk()
main_window.title('Personal Investment Helper 1.0.0 Get ready to lose some money')

# Put the window at the center of the screen
ww, wh = (800, 800)
sw = main_window.winfo_screenwidth()
sh = main_window.winfo_screenheight()
x = sw/2-ww/2
y = sh/2-wh/2
main_window.geometry('%dx%d+%d+%d'%(ww, wh, x, y))

# Load the meme pic at the top
buffett = Image.open('./wb.jpg')
meme = ImageTk.PhotoImage(buffett)
label_pic = tk.Label(main_window, image = meme)
label_pic.pack(fill='x', side='top')

# Load the warning message at the bottom
warning_label = tk.Label(main_window, 
    text = 'Important! This software does not represent any investment advice.', \
    fg = 'red', \
    font = font.Font(size = 15, family = 'Arial'))
warning_label.pack(fill='x', side='bottom')

##  Construct a frame for the stock prediction section  #######################
stock_prediction_frame = tk.Frame(main_window, width = 200, height = 250)
stock_prediction_frame.pack(fill='y', side='left')

# Frame topic title
pred_frame_label= tk.Label(stock_prediction_frame, \
    text='Stock Price Prediction', \
    bg = 'Grey', fg = 'White', \
    font = font.Font(size = 13, family = 'Times New Roman'))
pred_frame_label.pack(fill = 'x', side='top', padx = 20, pady = 15)

# label and entry box for ticker
label_ticker = tk.Label(stock_prediction_frame, text='Stock Ticker')
label_ticker.pack()
entry_ticker = tk.Entry(stock_prediction_frame, width = 15)
entry_ticker.pack(padx = 20, pady = 5)

# label and entry box for date
label_date = tk.Label(stock_prediction_frame, text='Date')
label_date.pack()
entry_date = tk.Entry(stock_prediction_frame, width = 15)
entry_date.pack(padx = 20, pady = 5)

# The prediction button
predict_button = tk.Button(stock_prediction_frame, text ='Predict', \
    width = 15, command = predict)
predict_button.pack(side = 'top', padx = 20, pady = 10)

# The description button
predict_desc_button = tk.Button(stock_prediction_frame, text ='Description', \
    width = 15, command = pred_desc)
predict_desc_button.pack(side = 'top', padx = 20, pady = 10)

##  Construct a frame for the portfolio analysis section  #####################
portfolio_frame = tk.Frame(main_window, width = 360, height = 250)
portfolio_frame.pack(fill= 'y', side='left')

# Frame topic title
portfolio_frame_label= tk.Label(portfolio_frame, width = 30, \
    text='Portfolio Analysis', \
    bg = 'Grey', fg = 'White', \
    font = font.Font(size = 13, family = 'Times New Roman'))
portfolio_frame_label.pack(fill = 'x', side='top', padx = 20, pady = 15)

# Label and entry box for tickers
tickers_label = tk.Label(portfolio_frame, text='Stocks')
tickers_label.pack()
tickers_entry = tk.Entry(portfolio_frame, width = 35)
tickers_entry.pack(padx = 20, pady = 5)

sub_frame = tk.Frame(portfolio_frame, width = 160, height = 250)
sub_frame.pack(fill = 'y', side= 'left')

# label and entry box for start and end date
start_label = tk.Label(sub_frame, text='Start date')
start_label.pack(side='top')
start_entry = tk.Entry(sub_frame, width = 13)
start_entry.pack(padx =30, side = 'top', pady = 5)
end_label = tk.Label(sub_frame, text='End date')
end_label.pack(side='top')
end_entry = tk.Entry(sub_frame, width = 13)
end_entry.pack(padx = 30, side='top', pady = 5)

# label and entry box for risk free rate, also create a string variable
rf = tk.DoubleVar()
rf_label = tk.Label(sub_frame, text='Risk free rate')
rf_label.pack(side='top')
rf_entry = tk.Entry(sub_frame, width = 13, textvariable = rf)
rf_entry.pack(padx = 30, side='top', pady = 5)

# The covariance/correlation analysis button
cov_button = tk.Button(portfolio_frame, text ='Cov/corr analysis', width = 15, \
    command = corr_and_cov_analysis)
cov_button.pack(pady = 15)

# The portfolio optimization button 
portfolio_button = tk.Button(portfolio_frame, text ='Optimized Portfolio', width = 15, \
    command = gmv_sr)
portfolio_button.pack(pady = 15)

# The description button
portfolio_desc_button = tk.Button(portfolio_frame, \
    text ='Description', width = 15, command = cov_desc)
portfolio_desc_button.pack(pady = 15)

##  Construct a frame for the multi-factor model section ######################
model_frame = tk.Frame(main_window, width = 200, height = 250)
model_frame.pack(fill='y', side='left')

# Frame topic title 
model_frame_label= tk.Label(model_frame, \
    text='Multi-Factor Model Simulation', \
    bg = 'Grey', fg = 'White', \
    font = font.Font(size = 13, family = 'Times New Roman'))
model_frame_label.pack(fill = 'x', side='top', padx = 20, pady = 15)

# label and entry for number of stocks
num_stock = tk.DoubleVar()
number_stock_label = tk.Label(model_frame, text='Number of stocks')
number_stock_label.pack()
number_stock_entry = tk.Entry(model_frame, width = 15, textvariable = num_stock)
number_stock_entry.pack(padx = 20, pady = 5)

# label and entry for the risk free rate
risk_f = tk.DoubleVar()
risk_free_label = tk.Label(model_frame, text='Risk free rate')
risk_free_label.pack()
risk_free_entry = tk.Entry(model_frame, width = 15, textvariable = risk_f)
risk_free_entry.pack(padx = 20, pady = 5)

# label and entry for the start time
start_year_label = tk.Label(model_frame, text='Start Time')
start_year_label.pack()
start_year_entry = tk.Entry(model_frame, width = 15)
start_year_entry.pack(padx = 20, pady = 5)

# label and entry for the end time
end_year_label = tk.Label(model_frame, text='End Time')
end_year_label.pack()
end_year_entry = tk.Entry(model_frame, width = 15)
end_year_entry.pack(padx = 20, pady = 5)

simulation_button = tk.Button(model_frame, text ='Run simulation', width = 15, command = output_graph)
simulation_button.pack(side = 'top', padx = 20, pady = 10)

simulation_desc_button = tk.Button(model_frame, text ='Description', width = 15, command = simulate_desc)
simulation_desc_button.pack(side = 'top', padx = 20, pady = 10)



main_window.mainloop()