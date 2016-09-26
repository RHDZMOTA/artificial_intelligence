# -*- coding: utf-8 -*-
'''
@author: Rodrigo Hernández Mota
'''

import numpy as np
import findata as fd

def f4(x):
    # determine if x is data for gen.algo (==2) or gen.pso
    a = np.shape(x[0])
    if len(a) > 1: X = [x[i][0] for i in range(len(x))]
    else: X = x

    xm = np.matrix(X)
    xm = xm.T
    
    # costo 
    c = np.matrix([170, 160, 175, 180, 195])
    c = c.T
    
    # función de gasto
    z = xm.dot(c)
    
    # reestric.
    nn = xm.shape[0]
    r = np.zeros((nn,10))
    alpha = 10000
    
    if len(a) == 1:
        # 0
        index = (-X[0] + 48 > 0)
        r[:,0] = alpha * index * (-X[0] + 48)
        # 1
        index = (-X[1] - X[0] + 79 > 0)
        r[:,1] = alpha * index * (-X[1] - X[0] + 79)
        # 2
        index = (65 - X[0] - X[1] > 0)
        r[:,2] = alpha * index * (65 - X[0] - X[1])
        #3 
        index = (87 - X[0] - X[1] - X[2] > 0)
        r[:,3] = alpha * index * (87 - X[0] - X[1] - X[2])
        # 4
        index = (64 - X[1] - X[2] > 0)
        r[:,4] = alpha * index * (64 - X[1] - X[2])
        # 5
        index = (73 - X[2] - X[3] > 0)
        r[:,5] = alpha * index * (73 - X[2] - X[3])
        #6 
        index = (82 - X[2] - X[3] > 0)
        r[:,6] = alpha * index * (82 - X[2] - X[3])
        #7
        index = (43 - X[3] > 0)
        r[:,7] = alpha * index * (43 - X[3])
        #8
        index = (52 - X[3] - X[4] > 0)
        r[:,8] = alpha * index * (52 - X[3] - X[4])
        #9
        index = (15 - X[4] > 0)
        r[:,9] = alpha * index * (15 - X[4])
    
    re = np.sum(r, axis = 1)
    z = np.array(z.T) + re
    return z[0]

# Función ej3 (2 variables)
def f3(x):
    # determine if x is data for gen.algo (==2) or gen.pso
    a = np.shape(x[0])
    if len(a) > 1: X = [x[i][0] for i in range(len(x))]
    else: X = x
    
    # comienza fun
    z = []
    
    s = np.size(X[0])
    rest1 = np.zeros(s)
    rest2 = np.zeros(s)
    for i,j in zip(X[0], X[1]):
        if i+j >= 1:
            z.append(i**2 + j**2)
        else:
            z.append(-10*(i+j)+20)
    
    a1 = 10000
    a2 = 10000
    
    if len(a) == 1:
        index = (-X[0]-1>= 0)
        rest1[index] = -X[0][index]-1
        index = (-X[1]-1>= 0)
        rest2[index] = -X[1][index]-1
    
    
        
    return z + a1 * rest1 + a2 * rest2
    
    
    
    

# Función y = x^1
def func_1v(x):
    # determine if x is data for gen.algo (==2) or gen.pso
    a = np.shape(x[0])
    if len(a) > 1: X = [x[i][0] for i in range(len(x))]
    else: X = x
    # evaluate and return 
    return X[0] ** 2

# Función y = x1^2 + x2^2 
def func_2v(x):
    # determine if x is data for gen.algo (==2) or gen.pso
    a = np.shape(x[0])
    if len(a) > 1: X = [x[i][0] for i in range(len(x))]
    else: X = x
    # evaluate and return 
    return X[0] ** 2 + X[1] ** 2
    
# Función  y = -x1*(10+100*x1) - x2*(5+40*x1) - x3*(5+50*x3)
def func1_3v(x):
    # determine if x is data for gen.algo (==2) or gen.pso
    a = np.shape(x[0])
    if len(a) > 1: X = [x[i][0] for i in range(len(x))]
    else: X = x
    # evaluate and return
    return -X[0]*(10+100*X[0]) - X[1]*(5+40*X[1]) - X[2]*(5+50*X[2]);
    
# Función y = ...
def func2_3v(x):
    # determine if x is data for gen.algo (==2) or gen.pso
    a = np.shape(x[0])
    if len(a) > 1: X = [x[i][0] for i in range(len(x))]
    else: X = x
    # evaluate and return 
    u = X[0]**2-2*X[0]+1-10*np.cos(X[0]-1)+X[1]**2+X[1]+\
    0.25-10*np.cos(X[0]+0.5)+X[2]**2-10*np.cos(X[2])
    return u

# Función de Rast...
def func_rast(x):
    # determine if x is data for gen.algo (==2) or gen.pso
    a = np.shape(x[0])
    if len(a) > 1: X = [x[i][0] for i in range(len(x))]
    else: X = x
    # evaluate and return 
    for i in range(len(X)):
        u = 10+ X[i]**2-10*np.cos(5*X[i])
    return u

# Función de Ackley
def func_ackley(x):
    # determine if x is data for gen.algo (==2) or gen.pso
    a = np.shape(x[0])
    if len(a) > 1: X = [x[i][0] for i in range(len(x))]
    else: X = x
    # evaluate and return 
    for i in range(len(X)):
        u = -10*np.exp(-np.sqrt(X[i]**2)) - np.exp(np.cos(5*X[i]))
    return u

# Función de Rosen
def func_rosen(x):
    # determine if x is data for gen.algo (==2) or gen.pso
    a = np.shape(x[0])
    if len(a) > 1: X = [x[i][0] for i in range(len(x))]
    else: X = x
    # evaluate and return 
    for i in range(len(X)-1):
        u = 100 * (X[i+1]-X[i])**2+(1-X[i+1])**2
    return u

# Markowitz function
def markowitz_1(X):
    '''
    function that applies the markowitz portfolio simulation
    '''

    # download data and calc. parameters
    import findata as fd
    n_acc, temp = np.shape(X)
    acc = ["GFNORTEO.MX","LIVEPOLC-1.MX","HERDEZ.MX", "BIMBOA.MX", "SANMEXB.MX", "ALSEA.MX"]
    acc = acc[:n_acc]
    price, returns = fd.download(acc)
    pa, rst, cst = fd.parameters(price, returns)
    
    '''    
    # conditional for genetic algo... 
    aa = np.shape(X[0])
    if len(aa) > 1:
        aux = X[0]
        for i in range(1,len(X)):
            aux = np.hstack([aux,X[i]])
        X = np.transpose(aux)
    '''
        
    # use part as matrix
    X = np.transpose(np.matrix(X))
    
    # determines the returns...  
    rp = X * np.transpose(rst)
    
    # determines the risk (var)
    riskp = X*cst*np.transpose(X)
    n,t = np.shape(X)
    te = np.arange(0,n)
    riskp = np.transpose(riskp[te,te])
    

    return np.array(rp), np.array(riskp), np.array(X.T), pa

def markowitz_2(rp, riskp, X):
    '''
    This function takes the two results of markowitz and...
    
    import matplotlib.pyplot as plt
    
    plt.plot(riskp, rp, 'b.')
    plt.title('Markowitz simulation')
    plt.xlabel('Risk (sd)')
    plt.ylabel('Returns')
    plt.show()
    '''
    
    b1 = 10000
    b2 = 100000
    alpha1 = 1000
    alpha2 = 50
    z = -b1*rp + b2*riskp
    
    i,j = X.shape
    rest = np.zeros(j)
    
    
    for act in X:
        # x > 0
        rest = rest + alpha1 * np.abs(act) * (act < 0)
        # x < 1
        rest = rest + alpha1 * np.abs(act) * (act > 1)
    
    
    X2 = np.transpose(np.matrix(X))
    X2 = np.array(X2)
    
    aux = []
    for act2 in X2:
        # sum x == 1
        aux.append(alpha2 * np.abs(np.sum(act2) - 1))
    rest = rest + np.array(aux)
    
    z = z.T + rest
    
    return z[0]

def markowitz(x):
    # determine if x is data for gen.algo (==2) or gen.pso
    '''
    a = np.shape(x[0])
    if len(a) > 1: X = [x[i][0] for i in range(len(x))]
    else: X = x
    '''
    X = x
    rp, riskp, X, pa = markowitz_1(X)
    z = markowitz_2(rp, riskp, X)
    
    return z
    
def simtrading_prom(data,nmovil,cash0,accion0,com):
    #nmovil = 10; #numero de dias del promedio movil
    #cash0 = 10000; #dinero disponible para comprar acciones
    #accion0 = 0; #numero de acciones disponibles para vender
    #com = 0.0029; # porcentaje de cobro de comision por operacion
    ndata,temp = np.shape(data); #numero de datos disponibles para simular
    promovil = np.zeros((ndata,1)); # iniciar el vector donde se guardara el promedio movil
    numaccion = np.ones((ndata+1,1))*accion0; # iniciar el vector donde se guardara el numero de acciones
    cash = np.ones((ndata+1,1))*cash0; # iniciar el vector donde se guardara el numero de acciones
    balance = np.ones((ndata+1,1))*(cash+accion0*data[nmovil-1,1]);
    
    for k in range(np.int(nmovil),np.int(ndata)):
        #calculo del promedio movil
        promovil[k,0] = np.mean(data[k-nmovil:k,1]);
        
        # simulacion de compra y venta
        if data[k,1]>=promovil[k,0]:
            #compra
            temp = np.floor(cash[k,0]/(data[k,1]*(1+com))); #acciones para que me alcanzan
            numaccion[k+1,0] = numaccion[k,0]+temp; # actualizo el numero de acciones
            cash[k+1,0] = cash[k,0]-temp*data[k,1]*(1+com); #actualizo el cash
            balance[k+1,0] = cash[k+1,0]+numaccion[k+1,0]*data[k,1];
        else:
            #vende
            numaccion[k+1,0] = 0;
            cash[k+1,0] = cash[k,0]+numaccion[k,0]*data[k,1]*(1-com);
            balance[k+1,0] = cash[k+1,0]+numaccion[k+1,0]*data[k,1];
    
    # La funcion regresa el promedio movil, el balance de la cuenta simulada,
    # el comportamiento del cash de la cuenta y el comportamiento de las acciones
    return promovil,balance,cash,numaccion

def prom_mov(x,f,price, returns, re):
    '''
    re = {0, 1}, apply resstrictions (1), or not (0)
    '''
    X = x
    cash_tot = 1000000    
    
    pa, rst, cst = fd.parameters(price, returns)
    
    X = np.transpose(np.matrix(x))
    
    #Uso de la funcion
    #nmovil = v; #numero de dias del promedio movil
    accion0 = 0; #numero de acciones disponibles para vender
    com = 0.0029; # porcentaje de cobro de comision por operacion
    rport = np.array([])
    sigma_port = np.array([])
    for i in range(np.shape(x)[1]):
        r_ind = []
        sigma_ind = []
        
        for j in range(np.shape(x)[0]):
            nmovil = f[j]
            pricegrum = price.values[:,j];
            ndata = np.size(pricegrum);
            data = np.reshape(pricegrum,(ndata,1));
            data = np.append(np.reshape(np.arange(0,ndata),(ndata,1)),data,axis=1);
        

            cash0 = cash_tot * x[j][i] #dinero disponible para comprar acciones
            promovil,balance,cash,numaccion = simtrading_prom(data,nmovil,cash0,accion0,com);
            
            #calculo del rendimiento promedio del balance final
            rend = (balance[nmovil+1:ndata]/balance[nmovil:ndata-1])-1;
            r_ind.append(np.mean(rend))
            sigma_ind.append(np.std(rend))
        
        x_reng = X[i,:]
        r_reng = np.matrix(r_ind).T
        sigma_reng = np.matrix(sigma_ind).T
        sigma_reng = np.power(sigma_reng,2)
        x_reng2 = np.power(x_reng,2)
        
        rport = np.append(rport, np.array(x_reng.dot(r_reng)))
        sigma_port = np.append(sigma_port, np.array(x_reng2.dot(sigma_reng)))
        
    X = np.array(X.T)
        
    # reestric
    if re == 1:
        
        b1 = 10000
        b2 = 100000
        alpha1 = 1000
        alpha2 = 50
        
        
        z = -b1*rport + b2*sigma_port
        
        i,j = X.shape
        rest = np.zeros(j)
        
        
        for act in X:
            # x > 0
            rest = rest + alpha1 * np.abs(act) * (act < 0)
            # x < 1
            rest = rest + alpha1 * np.abs(act) * (act > 1)
        
        
        X2 = np.transpose(np.matrix(X))
        X2 = np.array(X2)
        
        aux = []
        for act2 in X2:
            # sum x == 1
            aux.append(alpha2 * np.abs(np.sum(act2) - 1))
        rest = rest + np.array(aux)
        
        z = z.T + rest
    else:
        z = [rport, sigma_port]
        
    return z
    
def port_val(x,f, price, returns):
    
    cash_tot = 1000000        
    pa, rst, cst = fd.parameters(price, returns)        
    #Uso de la funcion
    #nmovil = v; #numero de dias del promedio movil
    accion0 = 0; #numero de acciones disponibles para vender
    com = 0.0029; # porcentaje de cobro de comision por operacion
    rport = np.array([])
    sigma_port = np.array([])
    r_ind = []
    sigma_ind = []
    for i in range(len(f)):
        nmovil = f[i]
        pricegrum = price.values[:,i];
        ndata = np.size(pricegrum);
        data = np.reshape(pricegrum,(ndata,1));
        data = np.append(np.reshape(np.arange(0,ndata),(ndata,1)),data,axis=1);
        

        cash0 = cash_tot * x[i] #dinero disponible para comprar acciones
        promovil,balance,cash,numaccion = simtrading_prom(data,nmovil,cash0,accion0,com);
            
        #calculo del rendimiento promedio del balance final
        rend = (balance[nmovil+1:ndata]/balance[nmovil:ndata-1])-1;
        r_ind.append(np.mean(rend))
        sigma_ind.append(np.std(rend))
        
    x_reng = np.asmatrix(x)
    r_reng = np.matrix(r_ind).T
    sigma_reng = np.matrix(sigma_ind).T
    sigma_reng = np.power(sigma_reng,2)
    x_reng2 = np.power(x_reng,2)
        
    rport = np.append(rport, np.array(x_reng.dot(r_reng)))
    sigma_port = np.append(sigma_port, np.array(x_reng2.dot(sigma_reng)))
        

    z = [rport, sigma_port]
        
    return z

def prom_mov2(x,f,price, returns, re):
    '''
    re = {0, 1}, apply resstrictions (1), or not (0)
    '''
    X = x
    cash_tot = 1000000    
    
    pa, rst, cst = fd.parameters(price, returns)
    
    #X = np.transpose(np.matrix(x))
    
    #Uso de la funcion
    #nmovil = v; #numero de dias del promedio movil
    accion0 = 0; #numero de acciones disponibles para vender
    com = 0.0029; # porcentaje de cobro de comision por operacion
    rport = np.array([])
    sigma_port = np.array([])
    for i in range(np.shape(x)[0]):
        r_ind = []
        sigma_ind = []
        
        for j in range(np.shape(x)[1]):
            nmovil = np.int(f[i,j])
            pricegrum = price.values[:,j];
            ndata = np.size(pricegrum);
            data = np.reshape(pricegrum,(ndata,1));
            data = np.append(np.reshape(np.arange(0,ndata),(ndata,1)),data,axis=1);
        

            cash0 = cash_tot * x[i,j] #dinero disponible para comprar acciones
            promovil,balance,cash,numaccion = simtrading_prom(data,nmovil,cash0,accion0,com);
            
            #calculo del rendimiento promedio del balance final
            rend = (balance[nmovil+1:ndata]/balance[nmovil:ndata-1])-1;
            r_ind.append(np.mean(rend))
            sigma_ind.append(np.std(rend))
        
        x_reng = X[i,:]
        r_reng = np.matrix(r_ind).T
        sigma_reng = np.matrix(sigma_ind).T
        sigma_reng = np.power(sigma_reng,2)
        x_reng2 = np.power(x_reng,2)
        
        rport = np.append(rport, np.array(x_reng.dot(r_reng)))
        sigma_port = np.append(sigma_port, np.array(x_reng2.dot(sigma_reng)))
        
    X = np.array(X.T)
        
    # reestric
    if re == 1:
        
        b1 = 10000
        b2 = 100000
        alpha1 = 1000
        alpha2 = 1000
        
        
        z = -b1*rport + b2*sigma_port
        
        i,j = X.shape
        rest = np.zeros(j)
        
        
        for act in X:
            # x > 0
            rest = rest + alpha1 * np.abs(act) * (act < 0)
            # x < 1
            rest = rest + alpha1 * np.abs(act) * (act > 1)
        
        
        X2 = np.transpose(np.matrix(X))
        X2 = np.array(X2)
        
        aux = []
        for act2 in X2:
            # sum x == 1
            aux.append(alpha2 * np.abs(np.sum(act2) - 1))
        rest = rest + np.array(aux)
        
        z = z.T + rest
    else:
        z = [rport, sigma_port]
        
    return z
    
    
def prom_mov3g(x,f,price, returns, re):
    '''
    re = {0, 1}, apply resstrictions (1), or not (0)
    '''
    X = x
    cash_tot = 1000000    
    
    pa, rst, cst = fd.parameters(price, returns)
    
    #X = np.transpose(np.matrix(x))
    
    #Uso de la funcion
    #nmovil = v; #numero de dias del promedio movil
    accion0 = 0; #numero de acciones disponibles para vender
    com = 0.0029; # porcentaje de cobro de comision por operacion
    rport = np.array([])
    sigma_port = np.array([])
    for i in range(np.shape(x)[0]):
        r_ind = []
        sigma_ind = []
        
        for j in range(np.shape(x)[1]):
            nmovil = np.int(f[i,j])
            pricegrum = price.values[:,j];
            ndata = np.size(pricegrum);
            data = np.reshape(pricegrum,(ndata,1));
            data = np.append(np.reshape(np.arange(0,ndata),(ndata,1)),data,axis=1);
        

            cash0 = cash_tot * x[i,j] #dinero disponible para comprar acciones
            promovil,balance,cash,numaccion = simtrading_prom(data,nmovil,cash0,accion0,com);
            rend = (balance[nmovil+1:ndata]/balance[nmovil:ndata-1])-1;
            rendm = np.mean(rend);
            riskm = np.std(rend);
            rendf = (balance[-1,0]/cash0)-1;
            print('Rendm = %.4f, Riskm = %.4f, Rendf = %.4f' % (rendm*100,riskm*100,rendf*100));
            
            # graf
            ndata,temp = np.shape(data);
            t = np.reshape(np.arange(0,ndata),(ndata,1));
            t1 = np.reshape(np.arange(0,ndata+1),(ndata+1,1));
            import matplotlib.pyplot as plt
            plt.figure(1);
            plt.subplot(3,1,1);
            plt.plot(data[nmovil:,0],data[nmovil:,1],'b-',t[nmovil:,0],promovil[nmovil:,0],'r-');
            plt.ylabel('precio');
            plt.grid(color='k', linestyle='--');
            plt.subplot(3,1,2);
            plt.plot(t1[nmovil:,0],numaccion[nmovil:,0],'b-');
            plt.ylabel('acciones');
            plt.grid(color='k', linestyle='--');
            plt.subplot(3,1,3);
            plt.plot(t1[nmovil:,0],balance[nmovil:,0],'b-');
            plt.ylabel('balance');
            plt.xlabel('día');
            plt.grid(color='k', linestyle='--');
            plt.show();
            

            #calculo del rendimiento promedio del balance final
            r_ind.append(np.mean(rend))
            sigma_ind.append(np.std(rend))
        
        x_reng = X[i,:]
        r_reng = np.matrix(r_ind).T
        sigma_reng = np.matrix(sigma_ind).T
        sigma_reng = np.power(sigma_reng,2)
        x_reng2 = np.power(x_reng,2)
        
        rport = np.append(rport, np.array(x_reng.dot(r_reng)))
        sigma_port = np.append(sigma_port, np.array(x_reng2.dot(sigma_reng)))
        
    X = np.array(X.T)
        
    # reestric
    if re == 1:
        
        b1 = 10000
        b2 = 100000
        alpha1 = 1000
        alpha2 = 1000
        
        
        z = -b1*rport + b2*sigma_port
        
        i,j = X.shape
        rest = np.zeros(j)
        
        
        for act in X:
            # x > 0
            rest = rest + alpha1 * np.abs(act) * (act < 0)
            # x < 1
            rest = rest + alpha1 * np.abs(act) * (act > 1)
        
        
        X2 = np.transpose(np.matrix(X))
        X2 = np.array(X2)
        
        aux = []
        for act2 in X2:
            # sum x == 1
            aux.append(alpha2 * np.abs(np.sum(act2) - 1))
        rest = rest + np.array(aux)
        
        z = z.T + rest
    else:
        z = [rport, sigma_port]
        
    return z
    
    
def prom_mov3(x,f,price, returns, re):
    '''
    re = {0, 1}, apply resstrictions (1), or not (0)
    '''
    X = x
    cash_tot = 1000000    
    
    pa, rst, cst = fd.parameters(price, returns)
    
    X = np.transpose(np.matrix(x))
    
    #Uso de la funcion
    #nmovil = v; #numero de dias del promedio movil
    accion0 = 0; #numero de acciones disponibles para vender
    com = 0.0029; # porcentaje de cobro de comision por operacion
    rport = np.array([])
    sigma_port = np.array([])
    z = []
    for i in range(np.shape(x)[1]):
        r_ind = []
        sigma_ind = []
        
        for j in range(np.shape(x)[0]):
            nmovil = f[j]
            pricegrum = price.values[:,j];
            ndata = np.size(pricegrum);
            data = np.reshape(pricegrum,(ndata,1));
            data = np.append(np.reshape(np.arange(0,ndata),(ndata,1)),data,axis=1);
        

            cash0 = cash_tot * x[j][i] #dinero disponible para comprar acciones
            promovil,balance,cash,numaccion = simtrading_prom(data,nmovil,cash0,accion0,com);
            
            #calculo del rendimiento promedio del balance final
            rend = (balance[nmovil+1:ndata]/balance[nmovil:ndata-1])-1;
            r_ind.append(np.mean(rend))
            sigma_ind.append(np.std(rend))
        
        x_reng = X[i,:]
        z.append(r_ind)
        r_reng = np.matrix(r_ind).T
        sigma_reng = np.matrix(sigma_ind).T
        sigma_reng = np.power(sigma_reng,2)
        x_reng2 = np.power(x_reng,2)
        
        rport = np.append(rport, np.array(x_reng.dot(r_reng)))
        sigma_port = np.append(sigma_port, np.array(x_reng2.dot(sigma_reng)))
    '''
        
    X = np.array(X.T)
        
    # reestric
    if re == 1:
        
        b1 = 10000
        b2 = 100000
        alpha1 = 1000
        alpha2 = 50
        
        
        z = -b1*rport + b2*sigma_port
        
        i,j = X.shape
        rest = np.zeros(j)
        
        
        for act in X:
            # x > 0
            rest = rest + alpha1 * np.abs(act) * (act < 0)
            # x < 1
            rest = rest + alpha1 * np.abs(act) * (act > 1)
        
        
        X2 = np.transpose(np.matrix(X))
        X2 = np.array(X2)
        
        aux = []
        for act2 in X2:
            # sum x == 1
            aux.append(alpha2 * np.abs(np.sum(act2) - 1))
        rest = rest + np.array(aux)
        
        z = z.T + rest
    else:
        z = [rport, sigma_port]
        '''
    return z
