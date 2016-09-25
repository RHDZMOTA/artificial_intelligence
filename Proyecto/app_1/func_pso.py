# -*- coding: utf-8 -*-

'''

Optimización por Enjambre de Partículas

Particle Swatm Optimization (PSO)



@author = Rodrigo Hernández Mota

'''

import numpy as np


def algo_pso(npart, c1, c2, func, nvar, desc):

    ini = float("inf")
    desc = (-1) * desc

    # numero de interaciones de búsqueda
    niter = 3000

    # inicialización del enjambre
    prtl = []    
    for i in range(nvar):
        prtl.append(np.random.rand(npart))

    # Mejores locales iniciales y desempeño
    prtl_ml = prtl
    fpl     = ini * np.ones(npart)


    # Mejor global inicial y desempeño
    prtl_mg = [0] * nvar
    fpg = ini;
    lfpg = []

    # velocidad
    prtl_v = []
    for i in range(nvar):
        prtl_v.append(np.zeros(npart))


    for k in range(niter):

        # función para calcular desempeño del enjambre
        fp = desc * func(prtl)

        # Encontrar mejor global (mínimo)
        index = np.argmin(fp)

        # comparación para determinar mínimo global
        if fp[index] < fpg:
            fpg = fp[index]
            for j in range(nvar):
                prtl_mg[j] = prtl[j][index]
            
        # encontrar mejores locales 
        for ind in range(npart):
            if fp[ind] < fpl[ind]:
                fpl[ind] = fp[ind]
                for j in range(nvar):
                    prtl_ml[j][ind] = prtl[j][ind]                

        # movimiento de partículas
        for j in range(nvar):
            prtl_v[j] = prtl_v[j] + c1 * np.random.rand(npart) * (prtl_mg[j] - prtl[j]) + c2 * np.random.rand(npart) * (prtl_ml[j] - prtl[j])
            prtl[j]   = prtl[j] + prtl_v[j]
        lfpg.append(fpg)
        
        if ind%(niter/20) == 0:
            print('Evaluation: {} %'.format(100*(k+1)/niter))

    

    fp = func(prtl)   
    '''
    import matplotlib.pyplot as plt
    plt.plot(lfpg)
    '''
    
    return prtl_mg, desc * fpg, fp, prtl

    
