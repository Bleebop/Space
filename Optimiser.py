import random
import math
from random import random as rd
import numpy as np

import Bodies
import KepEq
import config

#Genetic algorithm for MGA-1DSM problem (about MGA-1DSM, see https://pdfs.semanticscholar.org/5ca7/dec2d84dc269921fa19d357b07af7f341f30.pdf)
def GeneticAlgo2(plan_list, bounds, t_lim, s_sit, e_sit, Feval, pop, F, CR):
    gene = math.floor(Feval/pop)
    choice = range(pop)
    var_n = (len(plan_list)-1)*4 + 2
    if t_lim == float('inf'):
        lim_time = False
    else:
        lim_time = True
    
    startpos = ['satstart',1e10,0,0,0,0,0,0,plan_list[0].SOI]
    satstart = Bodies.Sat(*startpos)
    cans = [scale_time(np.array([rd()*(bounds[i][1]-bounds[i][0])+bounds[i][0] for i in range(var_n)]),t_lim,plan_list) for _ in range(pop)]
    fit_l = [0]*pop
    best_fit = float('inf')
        
    for i in range(pop):
        if lim_time:
            cans[i] = scale_time(cans[i],t_lim,plan_list)
        
        fit_l[i] = obj_MGA_1DSM(plan_list, cans[i], satstart, s_sit, e_sit)
        if fit_l[i]<best_fit:
            best_fit = fit_l[i]
    
    for g in range(gene):
        if g%100==0:
            print(g)
            print(best_fit)
            converg = True
            vfit1 = fit_l[0]
            for fit in fit_l[1:]:
                if abs(fit-vfit1)>0.01:
                    converg = False
                    break
            if converg:
                print('Convergence GA')
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
            
            if lim_time:
                cand = scale_time(cand,t_lim,plan_list)
            
            fit_can = obj_MGA_1DSM(plan_list, cand, satstart, s_sit, e_sit)
            if (fit_can<fit_l[x] or np.isnan(fit_l[x])) and not np.isnan(fit_can):
                cans[x] = cand
                fit_l[x] = fit_can
                if fit_l[i]<best_fit:
                    best_fit = fit_l[i]
    return cans, fit_l
    
#Simulated annealing for MGA-1DSM problem (about MGA-1DSM, see https://pdfs.semanticscholar.org/5ca7/dec2d84dc269921fa19d357b07af7f341f30.pdf)
def SA_AN(plan_list, bounds, t_lim, s_sit, e_sit, Ti, Tf, nfann, Feval):
    if t_lim == float('inf'):
        lim_time = False
    else:
        lim_time = True
    c=2
    var_n = (len(plan_list)-1)*4 + 2
    neval = 1
    nt = 1
    nr = 20
    n0 = math.floor(nfann/(nt*nr*var_n))
    alpha = (Tf/Ti)**(1/n0)
    
    startpos = ['satstart',1e10,0,0,0,0,0,0,plan_list[0].SOI]
    satstart = Bodies.Sat(*startpos)
    can = scale_time(np.array([rd()*(bounds[k][1]-bounds[k][0])+bounds[k][0] for k in range(var_n)]),t_lim,plan_list)
    fit = obj_MGA_1DSM(plan_list, can, satstart, s_sit, e_sit)
    best_can = np.copy(can)
    best_fit = fit
    
    rat_hist = [[]]*14
    SA_cycle = 0
    
    while neval < Feval:
        print(neval)
        print(best_fit)
        T = Ti
        neigh = np.array([bounds[k][1]-bounds[k][0] for k in range(var_n)])
        for i in range(n0):
            for z in range(nt):
                nacc = [0]*14
                for j in range(nr):
                    for k in range(var_n):
                        new_can = np.copy(can)
                        UB = min(new_can[k]+neigh[k],bounds[k][1])
                        LB = max(new_can[k]-neigh[k],bounds[k][0])
                        new_can[k] = rd()*(UB-LB)+LB
                        if lim_time and k%4==1:
                            new_can = scale_time(new_can,t_lim,plan_list)
                        new_fit = obj_MGA_1DSM(plan_list, new_can, satstart, s_sit, e_sit)
                        neval = neval + 1
                        if new_fit<=fit:
                            nacc[k] = nacc[k] + 1
                            can = new_can
                            fit = new_fit
                            if new_fit<best_fit:
                                best_can = new_can
                                best_fit = fit
                        elif rd()<math.exp((fit-new_fit)/T):
                            nacc[k] = nacc[k] + 1
                            can = new_can
                            fit = new_fit
                for k in range(var_n):
                    rat = nacc[k]/nr
                    rat_hist[k] = rat_hist[k] + [rat]
                    if rat>0.6:
                        fac = 1+c*(rat-0.6)/0.4
                    elif rat<0.4:
                        fac = 1/(1+c*(0.4-rat)/0.4)
                    else:
                        fac = 1
                    neigh[k] = neigh[k]*fac
            T = alpha*T
        SA_cycle += 1
    return best_can, best_fit, rat_hist
            
#Greedy local search for MGA-1DSM
def local_search(can, Fmin, Fmax, plan_list, bounds, t_lim, s_sit, e_sit, check_min):
    can_list = [can]
    n_var = len(can)
    startpos = ['satstart',1e10,0,0,0,0,0,0,plan_list[0].SOI]
    satstart = Bodies.Sat(*startpos)
    fit = obj_MGA_1DSM(plan_list, can, satstart, s_sit, e_sit)
    var_order = list(range(n_var*2))
    random.shuffle(var_order)
    var_ind = 0
    prev_var = []
    prev_ind = -1
    Fmin_array = np.array([(bounds[math.floor(i/2)][1]-bounds[math.floor(i/2)][0])*Fmin for i in range(n_var*2)])
    Fmax_array = np.array([(bounds[math.floor(i/2)][1]-bounds[math.floor(i/2)][0])*Fmax for i in range(n_var*2)])
    F_fac = [1]*(n_var*2)
    F = Fmin_array.copy()
    hist_tried = []
    hist_acc = []
    N_eval = 0
    local_min_reached = False
    while not local_min_reached:
        if N_eval%200==0:
            print(fit)
            print(F_fac)
        if prev_ind>-1:
            if prev_ind == len(prev_var)-1 and 2*F[prev_var[prev_ind]]<Fmax_array[prev_var[prev_ind]]:
                F[prev_var[prev_ind]] = 2*F[prev_var[prev_ind]]
                ### diag
                F_fac[prev_var[prev_ind]] = 2*F_fac[prev_var[prev_ind]]
                ###
            new_can, new_fit = try_var(can,prev_var[prev_ind],F,bounds,plan_list,satstart,s_sit,e_sit)
            if new_fit<fit:
                fit = new_fit
                can = new_can 
                can_list.append(new_can)
                var_used = prev_var.pop(prev_ind)
                prev_var += [var_used]
                prev_ind = len(prev_var)-1
            else:
                prev_ind -= 1
                F[prev_var[prev_ind]] = max(F[prev_var[prev_ind]]/4,Fmin_array[prev_var[prev_ind]])
                ### diag
                F_fac[prev_var[prev_ind]] = max(F_fac[prev_var[prev_ind]]/4,1)
                ###
        elif len(var_order)==0:
            if check_min and F != Fmin_array:
                F = Fmin_array
                ### diag
                F_fac = [1]*(n_var*2)
                ###
                prev_ind = len(prev_var)-1
            else:
                local_min_reached = True
        else:
            new_can, new_fit = try_var(can,var_order[var_ind],F,bounds,plan_list,satstart,s_sit,e_sit)
            if new_fit<fit:
                fit = new_fit
                can = new_can
                can_list.append(new_can)
                var_used = var_order.pop(var_ind)
                prev_var += [var_used]
                prev_ind = len(prev_var)-1
                var_ind = 0
            else:
                var_ind += 1
                if var_ind == len(var_order):
                    if check_min and F != Fmin_array:
                        F = Fmin_array
                        ### diag
                        F_fac = [1]*(n_var*2)
                        ###
                        prev_ind = len(prev_var)-1
                        var_ind = 0
                    else:
                        local_min_reached = True
        N_eval += 1
    return can, fit, N_eval
            
#Time scaling to avoid going out of bounds
def scale_time(can, t_lim, plan_list):
    t_tot = can[0]
    for j in range(len(plan_list)-1):
        t_tot += can[j*4+5]
    if t_tot > t_lim:
        rap = t_lim/t_tot
        can[0] = can[0]*rap
        for j in range(len(plan_list)-1):
            can[j*4+5] = can[j*4+5]*rap
    return can

def try_var(can,var,F,bounds,plan_list,testsat,s_sit,e_sit):
    if var%2==0:
        var_to_change = int(var/2)
        e = 1
    else:
        var_to_change = int((var-1)/2)
        e = -1
    new_can = can.copy()
    new_can[var_to_change] = max(min(can[var_to_change]+e*F[var],bounds[var_to_change][1]),bounds[var_to_change][0])
    #TODO add some kind of time scaling
    return new_can, obj_MGA_1DSM(plan_list, new_can, testsat, s_sit, e_sit)

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

