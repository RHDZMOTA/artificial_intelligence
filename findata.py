# -*- coding: utf-8 -*-

'''

script to download financial data...


'''

import numpy as np
import pandas as pd
import pandas.io.data as wb
import matplotlib.pyplot as plt
import datetime as dt
import os.path

def download(acc):
    '''
    function to automate the process of downloading financial data from yahoo
    I N P U T S
    acc = list of strings containing the stock names.
    
    '''
    
    # define starting and ending time     
    start = dt.datetime(2016,1,1)
    end = dt.datetime.now()
    
    # determine id...
    lid = []
    lid.append(dt.date.today())
    fid = str(lid[0])

    req_do = 0
    stocks = []
    for i in acc:        
        if os.path.isfile(i+fid+'.pkl'):
            stocks.append(pd.read_pickle(i+fid+'.pkl'))
        else:
            req_do = 1
    if req_do == 1:
        # read data...
        stocks = [wb.DataReader(i,'yahoo',start,end) for i in acc]
        # save data...
        j = 0 
        for i in acc:
            stocks[j].to_pickle(i+fid+'.pkl')
            j += 1

    # get prices and calc returns  
    price = pd.DataFrame()
    returns = pd.DataFrame()
    j = 0
    for i in acc:
        price[i] = stocks[j]['Adj Close']
        returns[i] = (price[i] / price[i].shift(1) - 1)[1:]
        j += 1
    
    return price, returns

def parameters(price, returns):
    '''
    function that calculates the parameters required  construct a portfolio.
    '''
    # mean and covar. of each stock
    returns = np.matrix(returns)
    rst = np.mean(returns, 0)
    cst = np.matrix(np.cov(returns, rowvar=False))
    
    # participation of stocks in portfolio (for markowitz)
    npart = 1000
    ntemp, nst = np.shape(returns)   
    part = np.random.rand(npart,nst)
    spart = np.sum(part, 1)
    for i in range(nst):
        part[:,i] = part[:,i] / spart;
    
    return part, rst, cst

def markowitz(part, rst, cst):
    '''
    function that applies the markowitz portfolio simulation
    '''
    part = np.matrix(part)
    
    # determines the returns...  
    rp = part * np.transpose(rst)
    
    # determines the risk (var)
    riskp = part*cst*np.transpose(part)
    n,t = np.shape(part)
    te = np.arange(0,n)
    riskp = np.transpose(riskp[te,te])
    

    return np.array(rp), np.array(riskp)

'''
# Example 
acc = ["GRUMAB.MX","BIMBOA.MX","SORIANAB.MX"]
price, returns = download(acc)
part, rst, cst = parameters(price, returns)
rp, riskp = markowitz(part, rst, cst)

plt.plot(riskp, rp, 'b.')
plt.title('Markowitz simulation')
plt.xlabel('Risk (sd)')
plt.ylabel('Returns')
plt.show()

if 'm' in globals():
    print('hey')
'''
