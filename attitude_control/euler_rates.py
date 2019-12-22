# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 22:21:33 2018

@author: lele
"""
import numpy as np

from numpy import deg2rad as r
from numpy import rad2deg as d
from numpy import cos as c
from numpy import sin as s

def matTrans(thetArr):
    t1 = thetArr[0]
    t2 = thetArr[1]
    t3 = thetArr[2]
    matRes = np.array([[0,s(t3),c(t3)],[0,c(t3)*c(t2),-s(t3)*c(t2)],[c(t2),s(t2)*s(t3),c(t3)*s(t2)]])
    return matRes

thet = np.array([[r(40),r(30),r(80)]])


for i in range(0,600):
    tempo = i/10
    omega = np.array([r(20*s(0.1*tempo)),r(20*0.01),r(20*c(0.1*tempo))])
    matt = matTrans(thet[-1])
    new_thet_p = np.dot(matt, omega)
    new_thet = thet[-1]+new_thet_p*0.1
    thet = np.append(thet, [new_thet], axis=0)

