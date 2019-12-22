import random
from random import random as rd
import numpy as np

import Bodies
import KepEq
import config

#Genetic algorithm for MGA-1DSM problem (about MGA-1DSM, see https://pdfs.semanticscholar.org/5ca7/dec2d84dc269921fa19d357b07af7f341f30.pdf)
def GeneticAlgo2(plan_list, bounds, t_lim, s_sit, e_sit, gene, pop, F, CR):
    choice = range(pop)
    var_n = (len(plan_list)-1)*4 + 2
    if t_lim == float('inf'):
        lim_time = False
    else:
        lim_time = True
    
    startpos = ['satstart',1e10,0,0,0,0,0,0,plan_list[0].SOI]
    satstart = Bodies.Sat(*startpos)
    cans = [np.array([rd()*(bounds[i][1]-bounds[i][0])+bounds[i][0] for i in range(var_n)]) for _ in range(pop)]
    fit_l = [0]*pop
        
    for i in range(pop):
        if lim_time: 	# todo faire une fonction commune "scale_time"
            t_tot = 0
            for j in range(len(plan_list)-1):
                t_tot += cans[i][j*4+5]
            if t_tot > t_lim:
                rap = t_lim/t_tot
                for j in range(len(plan_list)-1):
                    cans[i][j*4+5] = cans[i][j*4+5]*rap
        
        fit_l[i] = obj_MGA_1DSM(plan_list, cans[i], satstart, s_sit, e_sit)
    
    for g in range(gene):
        if g%100==0:
            print(g)
            converg = True
            vfit1 = fit_l[0]
            for fit in fit_l[1:]:
                if abs(fit-vfit1)>0.000001:
                    converg = False
                    break
            if converg:
                break
        for p in range(pop):
            [x, a, b, c] = random.sample(choice,4)
            elemx = cans[x]
            elema = cans[a]
            elemb = cans[b]
            elemc = cans[c]
            vec_crois = elema + F*(elemb-elemc)
            cand = np.copy(elemx)
            
            R = random.randrange(var_n)
            for v in range(var_n):
                if rd()<CR or v==R:
                    if vec_crois[v]<bounds[v][0]:
                        diff = bounds[v][0]-vec_crois[v]
                        vec_crois[v] += 1.5*diff
                    elif bounds[v][1]<vec_crois[v]:
                        diff = vec_crois[v]-bounds[v][1]
                        vec_crois[v] -= 1.5*diff
                    cand[v] = vec_crois[v]
            
            if lim_time:		# todo faire une fonction commune "scale_time"
                t_tot = 0
                for j in range(len(plan_list)-1):
                    t_tot += cand[j*4+5]
                if t_tot > t_lim:
                    rap = t_lim/t_tot
                    for j in range(len(plan_list)-1):
                        cand[j*4+5] = cand[j*4+5]*rap
            
            fit_can = obj_MGA_1DSM(plan_list, cand, satstart, s_sit, e_sit)
            if fit_can<fit_l[x] and not np.isnan(fit_can):
                cans[x] = cand
                fit_l[x] = fit_can
    return cans, fit_l

#Fitness function (ie Delta-V)
def obj_MGA_1DSM(plan_list, dec_vec, testsat, s_sit, e_sit):
    startpos = plan_list[0].retOrbDeg()
    testsat.reset(*startpos)
    DV_DSM, pos_f, _, _  = testsat.MoveToEnd2(plan_list, dec_vec, False)
    DV_tot = DV_DSM
    if s_sit == 'circ_15r':
        sq_v_circ = plan_list[0].mu/(plan_list[0].alt_lim*1.5)
        sq_v_e = 2*sq_v_circ
        v_pb = np.sqrt(dec_vec[1]*dec_vec[1]+sq_v_e)
        DV_tot += (v_pb-np.sqrt(sq_v_circ))
    if s_sit == 'circ_15r':
        sq_v_circ = plan_list[-1].mu/(plan_list[-1].alt_lim*1.5)
        sq_v_e = 2*sq_v_circ
        v_pb = np.sqrt(np.linalg.norm(pos_f[1])**2+sq_v_e)
        DV_tot += (v_pb-np.sqrt(sq_v_circ))
    return DV_tot

def build_bounds(plan_list, eta_b, bou_rb):
    bounds = [[0,1],[0,1]]
    for ind, plan in enumerate(plan_list[:-2]):
        sumT = plan.T+plan_list[ind+1].T
        bounds += eta_b + [[sumT*0.1, sumT*2]] + bou_rb
    sumT = plan_list[-2].T + plan_list[-1].T
    bounds += eta_b + [[sumT*0.1, sumT*2]]
    return bounds

