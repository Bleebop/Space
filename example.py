import sys
from random import random as rd
import random
from copy import deepcopy

import math
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from numpy import cos as cos
from numpy import sin as sin
from numpy import pi as PI
import matplotlib.pyplot as plt

import KepEq
import LambSolve
import Optimiser
import Bodies
import trajViz
import config
        
solarsyst = Bodies.SolarSystem()
#planet(name, code, apo, per, INC, LPE, LAN, MNA, mu, SOI_rad, alt_lim, SOI)
#sat(name, SMA, ECC, INC, LAN, LPE, EPH, MNA, SOI)
avoidcases = Bodies.Gal('avoidcases')
avoidcases2 = Bodies.Gal('avoidcases2')
avoidcases.SOI = avoidcases2

Kerbol = Bodies.Planet('Kerbol', 0, 0, 0, 0, 0, 0, 0, 1.1723328e18, 4e11, 2.63e8, avoidcases)

Moho = Bodies.Planet('Moho', 4, 6315765980, 4210510628, 7, 15, 70, 3.14, 1.6860938e11, 9646663, 257000, Kerbol)

Eve = Bodies.Planet('Eve', 5, 9931011387, 9734357701, 2.1, 0, 15, 3.14, 8.1717302e12, 85109365, 790000, Kerbol)
Gilly = Bodies.Planet('Gilly', 13, 48825000, 14175000, 12, 10, 80, 0.9, 8289449.8, 126123.27, 20000, Eve)

Kerbin = Bodies.Planet('Kerbin', 1, 13599840256, 13599840256, 0, 0, 0, 3.14, 3.5315984e12, 84159286, 670000, Kerbol)
Mun = Bodies.Planet('Mun', 2, 12000000, 12000000, 0, 0, 0, 1.7, 6.5138398e10, 2429559.6, 207100, Kerbin)
Minmus = Bodies.Planet('Minmus', 3, 47000000, 47000000, 6, 38, 78, 0.9, 1.7658000e9, 2247428.8, 66000, Kerbin)

Duna = Bodies.Planet('Duna', 6, 21783189163, 19669121365, 0.06, 0, 135.5, 3.14, 3.0136321e11, 47921949, 370000, Kerbol)
Ike = Bodies.Planet('Ike', 7, 3296000, 3104000, 0.2, 0, 0, 1.7, 1.8568369e10, 1049598.9, 143000, Duna)

Dres = Bodies.Planet('Dres', 15, 46761053692, 34917642714, 5, 90, 280, 3.14, 2.1484489e10, 32832840, 144000, Kerbol)

Jool = Bodies.Planet('Jool', 8, 72212238387, 65334882253, 1.304, 0, 52, 0.1, 2.8252800e14, 2.4559852e9, 6200000, Kerbol)
Laythe = Bodies.Planet('Laythe', 9, 27184000, 27184000, 0, 0, 0, 3.14, 1.9620000e12, 3723645.8, 550000, Jool)
Val = Bodies.Planet('Val', 10, 43152000, 43152000, 0, 0, 0, 0.9, 2.0748150e11, 2406401.4, 308000, Jool)
Tylo = Bodies.Planet('Tylo', 12, 68500000, 68500000, 0.25, 0, 0, 3.14, 2.8252800e12, 10856518, 612000, Jool)
Bop = Bodies.Planet('Bop', 11, 158697500, 98302500, 15, 25, 10, 0.9, 2.4868349e9, 1221060.9, 87000, Jool)
Pol = Bodies.Planet('Pol', 14, 210624207, 149155794, 4.25, 15, 2, 0.9, 7.2170208e8, 1042138.9, 45000, Jool)

Eeloo = Bodies.Planet('Eeloo', 16, 113549713200, 66687926800, 6.15, 260, 50, 3.14, 7.4410815e10, 1.1908294e8, 214000, Kerbol)

year = Kerbin.T
day = 6*3600

solarsyst.AddSun(Kerbol,0)
solarsyst.AddPlanet(Kerbin,1)
solarsyst.AddPlanet(Mun,2)
solarsyst.AddPlanet(Minmus,3)
solarsyst.AddPlanet(Moho,4)
solarsyst.AddPlanet(Eve,5)
solarsyst.AddPlanet(Duna,6)
solarsyst.AddPlanet(Ike,7)
solarsyst.AddPlanet(Jool,8)
solarsyst.AddPlanet(Laythe,9)
solarsyst.AddPlanet(Val,10)
solarsyst.AddPlanet(Bop,11)
solarsyst.AddPlanet(Tylo,12)
solarsyst.AddPlanet(Gilly,13)
solarsyst.AddPlanet(Pol,14)
solarsyst.AddPlanet(Dres,15)
solarsyst.AddPlanet(Eeloo,16)

plan_name_dict = solarsyst.plan_names

t_lim = 10*year
s_sit = 'circ_15r'
e_sit = 'circ_15r'

bou_rb = [[1.05,12],[-PI,PI]]
bou_eta = [[0.01,0.9]]

plan_list = [Kerbin, Kerbin, Jool]
#plan_list = [Kerbin, Eve, Moho]
## best (?) 3873
#plan_list = [Kerbin,Moho]
## best (?) 4054
#plan_list = [Kerbin, Duna, Jool, Eeloo]
## current best around 3000m/s
#plan_list = [Kerbin, Eeloo]
## probable best fit for direct transfer at 3489m/s
bounds = [[0,t_lim],[50,4000]]+Optimiser.build_bounds(plan_list, bou_eta, bou_rb)

pop = 20
Feval = 40000
F = 0.7
CR = 0.7
resu, fit_l = Optimiser.GeneticAlgo2(plan_list, bounds, t_lim, s_sit, e_sit, Feval, pop, F, CR)

for i, fit in enumerate(fit_l):
    if np.isnan(fit):
        fit_l[i] = float('inf')
cand_ind = np.argmin(fit_l)
best_cand = resu[cand_ind]
print("best fit GA:")
print(fit_l[cand_ind])

# Ti = 20000
# Tf = 100
# nfann = 2000
# Feval = 20000
# parameters to explore...
# best_cand, fit, rat_hist = Optimiser.SA_AN(plan_list, bounds, t_lim, s_sit, e_sit, Ti, Tf, nfann, Feval)
# print(fit)

cand_local, fit_local, N_eval = Optimiser.local_search(best_cand,0.0001,0.01,plan_list,bounds,t_lim,s_sit,e_sit,False)
print(fit_local)
print(N_eval)

startpos = ['testsat', 13599840256, 0, 0,0,0,0,0,Kerbol]
testsat = Bodies.Sat(*startpos)

(DV_DSM, pos_f, pos_tr, orb_tr) = testsat.MoveToEnd2(plan_list, cand_local, True)

trajViz.Visualize(pos_tr, orb_tr, plan_name_dict, True)









