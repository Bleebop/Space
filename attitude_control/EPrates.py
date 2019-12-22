# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 21:22:19 2018

@author: lele
"""

import numpy as np

from numpy import deg2rad as r
from numpy import rad2deg as d
from numpy import cos as c
from numpy import sin as s

from DCMtoEP import EPrates

bet = np.array([[0.408248, 0,0.408248,0.816497]])

for i in range(0,6000):
    tempo = i/100
    omega = np.array([r(20*s(0.1*tempo)),r(20*0.01),r(20*c(0.1*tempo))])
    new_bet_p = EPrates(bet[-1],omega)
    new_bet = bet[-1]+new_bet_p*0.01
    bet = np.append(bet, [new_bet], axis=0)