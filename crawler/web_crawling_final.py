from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import Select
from selenium import webdriver
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import lxml
import time

# IMPORTANT CLARIFICATION

# This doc takes approximatly 20 minutes to download the 29 csv. 
# This is not due to the efficiency issue, but we deliberately 
# slow down the coding running speed by adding time.sleep(20) to 
# match the slow speed in Chrom when logging in and loading the data 
# which usually takes longer time.

# If it reports wrong with "Message: no such element: Unable to locate 
# element: {"method":"css selector","selector":"[id="_pageHeader_fin_dropdown_period"]"}"
# This is normal, just indicating the web loading speed is again slower than the code. 
# The solution would be to continue runing the Line 181-183 starting from where the loop breaks. 
# For convenience, we have already run this py to download the 29 csv and store in data_backup, 
# and run the other fucntions based on data_backup. You are more than welcome to run this doc 
# to downlaod the data, and they should be exactly the same as what we provide in data_backup. 

# config variables
username = 'rongyinghe@uchicago.edu'
password = 'Chi990904'
bot = webdriver.Chrome()
comp_df = pd.read_csv('company_info.csv')
colnames = [
 'Return on Assets %',
 'Return on Capital %',
 'Return on Equity %',
 'Return on Common Equity %',
 'Gross Margin %',
 'EBITDA Margin %',
 'EBITA Margin %',
 'EBIT Margin %',
 'Earnings from Cont. Ops Margin %',
 'Net Income Margin %',
 'Net Income Avail. for Common Margin %',
 'Normalized Net Income Margin %',
 'Levered Free Cash Flow Margin %',
 'Unlevered Free Cash Flow Margin %',
 'Total Asset Turnover',
 'Fixed Asset Turnover',
 'Accounts Receivable Turnover',
 'Inventory Turnover',
 'Current Ratio',
 'Quick Ratio',
 'Cash from Ops. to Curr. Liab.',
 'Avg. Days Sales Out.',
 'Avg. Days Inventory Out.',
 'Avg. Days Payable Out.',
 'Avg. Cash Conversion Cycle',
 'Total Debt/Equity',
 'Total Debt/Capital',
 'LT Debt/Equity',
 'LT Debt/Capital',
 'Total Liabilities/Total Assets',
 'EBIT / Interest Exp.',
 'EBITDA / Interest Exp.',
 '(EBITDA-CAPEX) / Interest Exp.',
 'Total Debt/EBITDA',
 'Total Debt/(EBITDA-CAPEX)'
]


# helper functions
def convert_date(date_str):
    '''
    Convert the date from string to the formate of {}-{}-{}

    Input:
        date_str(str): the string of the date
    Output: 
        date in form of {}-{}-{}
    '''
    m, d, y = date_str.split('-')
    months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov',\
              'Dec']
    m1 = str(months.index(m) + 1)
    m1 = '0' + m1 if len(m1) == 1 else m1
    return '{}-{}-{}'.format(y, m1, d)


# crawler functions
def scrape_one_company(url, comp_code):
    '''
    Scrape the needed information of one company. As Capital IQ needs log in, 
    we uses username and password to log in first, and then change the time 
    duration from annually to quarterly, then view all year's data, clean the 
    dataset. Filter data in year 2013-2019, choose 'Company' and 'Year' colnames.
    Create a csv named as {company_ticker}.csv, saved under the folder named data. 
    '''
    bot.get(url)
    try:  # if session out, you need to login again
        bot.find_element_by_id('username').send_keys(username)
        pwd = bot.find_element_by_id('password')
        pwd.send_keys(password)
        pwd.send_keys(Keys.RETURN)
        time.sleep(20)  # login process can be very slow
        print('Logge in, and scraping: {}...'.format(comp_code))
    except:
        print('No need to login again. Scraping: {}...'.format(comp_code))
        pass
    
    select = Select(bot.find_element_by_id('_pageHeader_fin_dropdown_period'))
    select.select_by_visible_text('Quarterly') # select data by quarter

    # view all year's data
    button = bot.find_element_by_id('ctl03__rangeSlider_viewAll')  
    button.click()
    time.sleep(20)  # refreshing data takes time
    
    soup = BeautifulSoup(bot.page_source, 'lxml')
    
    # initial df of quarter dates
    ele_a = soup.find('a',{'id': '_dataItemPicker'})
    ele_tr = ele_a.find_parent('tr').find_parent('tr')
    ele_tds = ele_tr.find_all('td')
    quarter_dates = []
    for td in ele_tds[6:]:
        try:
            date_str = td.find('div').find('table').find('tr').find('td').text.\
            replace('\t','').replace('\n','')[-11:]
            quarter_dates.append(convert_date(date_str))
        except:
            continue
    df = pd.DataFrame(data=quarter_dates, columns=['Quarter'])
    
    # add a column in each 
    for idx, col in enumerate(colnames):
        span = soup.find('span',{'title': lambda value: value and \
                                 value.startswith(col)})
        if not span:  # some company can miss one or more columns
            continue
        
        # Clean the dataset 
        tds = span.parent.parent.parent.parent.find_all('td')
        values = []
        for td in tds[1:]:
            na = td.find('div').find('span')  # 'NA' or 'FYC' in html page
            not_na = td.find('div').find('a')  # number in html page
            if na:
              # for instance, company 'GS' has FYC, which means that quarter 
              # date does not exist, this value should be excluded
                if na.text == 'FYC':  
                    continue
                values.append(np.nan)
            elif not_na: 
                multiply = 1
                val = not_na.text
                if val[0] == '(' and val[-1] == ')':  
                    multiply = -1   # check whether it is a negative number
                    val = val[1:-1]
                val = val.replace(',', '')  # avoid number like: '1,224.0'
                if '%' in val:
                    val = float(val[:-1]) / 100
                elif val.endswith('x'):
                    val = float(val[:-1])
                try:
                    val = float(val)
                except Exception as e:  # val can be 'NA', 'NM', etc.
                    val = np.nan
                values.append(multiply * val)
        
        df[col.replace(' %','')] = values
        
    df['Company'] = comp_code
    df['Year'] = df['Quarter'].str[:4]
    # filter data in year 2013-2019
    filtered_df = df[df['Year'].isin([str(y) for y in range(2013,2020)])]  
    
    # exluce 'Company' and 'Year' colnames
    res_cols = ['Company'] + list(filtered_df.columns)[:-2]  
    res_df = filtered_df[res_cols]
    
    filepath = './data/{}.csv'.format(comp_code)
    res_df.set_index('Company').to_csv(filepath)


# scarpe the DJ companies csv using the loop 
if __name__ == '__main__':
    for i in range(len(comp_df)):
        scrape_one_company(comp_df['url'][i], comp_df['code'][i])