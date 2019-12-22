# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 20:56:34 2018

@author: lele
"""
import numpy as np
from matplotlib import pyplot as plt

def integrator(func_to_int,ome,init_cond,tf,nstep): # 
    
    res = np.array([init_cond])
    delt = tf/nstep
    
    for i in range(0,nstep):
        tempo = i*delt
        omega = ome(tempo)
        new_res_p = func_to_int(res[-1],omega)
        new_res = res[-1]+new_res_p*delt
        res = np.append(res, [new_res], axis=0)
        
    plt.plot(res)
    plt.show()
    
def integratorMRP(func_to_int,ome,init_cond,tf,nstep): # 
    
    res = np.array([init_cond])
    delt = tf/nstep
    
    for i in range(0,nstep):
        tempo = i*delt
        omega = ome(tempo)
        new_res_p = func_to_int(res[-1],omega)
        new_res = res[-1]+new_res_p*delt
        if np.linalg.norm(new_res)>1:
            new_res = -new_res/np.linalg.norm(new_res)**2
        res = np.append(res, [new_res], axis=0)
        
    plt.plot(res)
    plt.show()
    