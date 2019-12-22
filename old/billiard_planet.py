import sys
from random import random as rd
import random
from copy import deepcopy

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from numpy import cos as cos
from numpy import sin as sin
import matplotlib.pyplot as plt

def distAng(ang1, ang2):
    if abs(ang1-ang2)<np.pi:
        return abs(ang1-ang2)
    else:
        return 2*np.pi-abs(ang1-ang2)

def NewtRaphELL(x,e,M):
    bla = x - (x - e*sin(x) - M)/(1-e*cos(x))
    return bla

def NewtStartELL(e, M):
    t34 = e*e
    t35 = e*t34
    t33 = cos(M)
    return M+(0.5*t35+e+(t34+3*t33*t35/2)*t33)*sin(M)

def NewtRaphHYP(x,e,N):
    bla = x - (e*np.sinh(x) - N - x)/(e*np.cosh(x) - 1)
    return bla
    
def NewtStartHYP(e, N):
    if N/e>3:
        H = np.log(N/e)+0.85
    else:
        H = N/(e-1)
        if H*H>6*(e-1):
            H = np.exp(np.log(6*N)/3)
    return H
    
def getE(ECC, M):
    global n_it
    E = NewtStartELL(ECC,M)
    diff = 100
    it = 0
    while diff > 1e-10:
        it += 1
        E_bis = NewtRaphELL(E, ECC, M)
        diff = abs(E - E_bis)
        E = E_bis
        if it > n_it:
            print('more than '+str(n_it)+' iterations for E ; ECC = '+str(ECC)+' ; M = '+str(M))
            break
#    print('it Ell : '+str(it))
    return E
    
def getH(ECC, N):
    global n_it
    if N<0:
        N = -N
        is_neg = True
    else:
        is_neg = False
    H = NewtStartHYP(ECC,N)
    diff = 100
    it = 0
    while diff > 1e-10:
        it += 1
        H_bis = NewtRaphHYP(H, ECC, N)
        diff = abs(H - H_bis)
        H = H_bis
        if it > n_it:
            print('more than '+str(n_it)+' iterations for H')
            break
#    print('it Hyp : '+str(it))
    if is_neg:
        H = -H
    return H

def findClosestApp(nt, body_c, body_p, t_lim, D, UT, tarplan, parplan, closest_app):
    global n_it
    resu = [0,'']
    resumod = False
    approach = [0,0,'']
    appmod = False
    R = 1000
    dR = 1
    sq_dist_cp = float('inf')

    it=0
    while abs(R/dR)>10:
        it+=1
        sq_dist_cp, R, dR = NewtRaphTimeClos(body_c, body_p, body_c.SOI, nt)
        nt = nt - R/dR
        if it > n_it:
            break
#    print(it)
#    print("dist : "+str(np.sqrt(sq_dist_cp)))
    
    if sq_dist_cp<D*D and nt>UT:
        nt1 = nt - body_p.T*D/(2*np.pi*body_p.SMA)
        
        H = 1000
        R = 1
        
        it=0
        while abs(H/(2*R))>1:
            it+=1
            H, R = NewtRaphTimeInter(body_c, body_p, body_c.SOI, nt1, D)
            nt1 = nt1 - H/(2*R)
            if it > n_it:
                break
        if nt1<t_lim and nt1>UT:
            resu = [nt1, body_p]
            resumod = True
            t_lim = nt1
            if tarplan or parplan:
                approach = [3,0,body_p]
                appmod = True
        else:
            [rc,drc] = body_c.GetCartFromOrb(t_lim)
            [rp,drp] = body_p.GetCartFromOrb(t_lim)
            dist_cp = np.sqrt(np.dot(rc-rp,rc-rp))
            if dist_cp<closest_app:
                approach = [3,dist_cp, body_p]
                appmod = True
    elif nt>UT and (tarplan or parplan) and nt<t_lim:
        dist_cp = np.sqrt(sq_dist_cp)
        if dist_cp<closest_app:
            approach = [3,dist_cp, body_p]
            appmod = True
    return (resumod, resu, appmod, approach)
    
def NewtRaphTimeClos(body_c, body_p, SOI, nt):
    global pos_vel_list
    [rc,drc] = body_c.GetCartFromOrb(nt)
    nrc = np.linalg.norm(rc)
#    ndrc = np.linalg.norm(drc)
    ddrc = -SOI.mu*rc/nrc**3
#    nddrc = np.linalg.norm(ddrc)
    [rp,drp] = body_p.GetCartFromOrb(nt)
    nrp = np.linalg.norm(rp)
#    ndrp = np.linalg.norm(drp)
    ddrp = -SOI.mu*rp/nrp**3
#    nddrp = np.linalg.norm(ddrp)
    sq_dist_cp = np.dot(rc-rp,rc-rp)
    R = np.dot(drc-drp,rc-rp)
    dR = np.dot(ddrc-ddrp,rc-rp) + np.dot(drc-drp,drc-drp)
    
#    sq_dist_cp = nrc**2 + nrp**2 - 2*np.dot(rc,rp)
#    R = nrc*ndrc + nrp*ndrp - np.dot(drc,rp) - np.dot(rc,drp)
#    dR = ndrc**2 + nrc*nddrc + ndrp**2 + nrp*nddrp - np.dot(ddrc,rp) - 2*np.dot(drc,drp) - np.dot(rc,ddrp)
    
    return (sq_dist_cp, R, dR)
    
def NewtRaphTimeInter(body_c, body_p, SOI, nt, D):
    [rc,drc] = body_c.GetCartFromOrb(nt)
#    nrc = np.linalg.norm(rc)
#    ndrc = np.linalg.norm(drc)
    [rp,drp] = body_p.GetCartFromOrb(nt)
#    nrp = np.linalg.norm(rp)
#    ndrp = np.linalg.norm(drp)
    
    H = np.dot(rc-rp,rc-rp) - D*D
    R = np.dot(drc-drp,rc-rp)
    
#    H = nrc**2 + nrp**2 - 2*np.dot(rc,rp) - D**2
#    R = nrc*ndrc + nrp*ndrp - np.dot(drc,rp) - np.dot(rc,drp)
    return (H, R)

def inter_intersec(l1,l2):
    global n_it
    ll1 = len(l1)
    ll2 = len(l2)
    i=0
    j=0
    l1v = False
    l2v = False
    l_inter = []
    while i<ll1 and j<ll2:
        if l1[i]<l2[j]:
            l1v = not l1v
            if l2v:
                l_inter += [l1[i]]
            i+=1
        elif l1[i]==l2[j]:
            l1v = not l1v
            l2v = not l2v
            if l1v == l2v:
              l_inter += [l1[i]]
            i+=1
            j+=1
        else:
            l2v = not l2v
            if l1v:
                l_inter += [l2[j]]
            j+=1
    return l_inter

def obj_func(target, resu, end_pos, end_orb, mindist_w_targ, mindist_w_targ_par, DV_tot, endt, int_s, opti_DV):
    if not resu:
        fit = float('inf')
    elif target[1] == 'PosCirc':
        if end_orb[7].name == target[0].name:
            per = end_orb[0]*(1-end_orb[1])
            fit = abs(2*target[0].alt_lim-per)
        else:
            if mindist_w_targ[0]==1:
                fit = 2*target[0].SMA + mindist_w_targ[1]
            elif mindist_w_targ[0]==3:
                fit = mindist_w_targ[1]
            else:
                fit = float('inf')
    else:
        fit = float('inf')
    fit = (5*fit/target[0].clos_rad)**2
    DV_score = 3*np.arctan(DV_tot/20000)
    fit += DV_score
    return fit
    
def obj_func_ej(target, resu, end_pos, end_orb, mindist_w_targ, mindist_w_targ_par, DV_tot, endt, int_s, opti_DV):
    if not resu:
        fit = 0
    else:
        fit = 0
        for interc in int_s:
            if interc[0]!=Kerbin.alt_lim:
                sc_per = 200/(interc[1]/(interc[0]*3) + (interc[0]*3)/interc[1])
                fit += 1000 + sc_per
        if DV_tot<2000:
            DV_score = 2000
        else:
            DV_score = 2000*(2000/DV_tot)**8 
        fit += DV_score
    return -fit
    
def obj_func_circ(target, resu, end_pos, end_orb, mindist_w_targ, mindist_w_targ_par, DV_tot, endt, int_s, opti_DV):
    if not resu:
        fit = float('inf')
    elif end_orb[7].name == target[0].name:
        apo = (1+end_orb[1])*end_orb[0]
        if apo>0:
            score_apo = 2*np.arctan(abs(apo-target[2])/target[2])/np.pi
        else:
            score_apo = 1 + 2*np.arctan(-target[2]/apo)/np.pi
        per = (1-end_orb[1])*end_orb[0]
        score_per = 2*np.arctan(abs(per-target[2])/target[2])/np.pi
        score_inc = end_orb[2]
        DV_score = 2*np.arctan(DV_tot/10000)/np.pi
        fit = DV_score + score_per + score_apo + score_inc
    else:
        if mindist_w_targ[0]==1:
            fit = 56 + 2*np.arctan(2*target[0].SMA + mindist_w_targ[1])/np.pi
        elif mindist_w_targ[0]==3:
            fit = 55 + 2*np.arctan(mindist_w_targ[1])/np.pi
        else:
            fit = float('inf')
    return fit
        
    
#        if end_orb[7].name == 'Kerbol':
#            apo = (1+end_orb[1])*end_orb[0]
#            fit = 1e11/apo
#        else:
#            apo = end_orb[7].apo
#            fit = 1e11/apo
#        if DV_tot<1500:
#            DV_score = 0
#        else:
#            DV_score = (DV_tot/1500)**4
#        fit += DV_score
#    return fit
            
#        if end_orb[7].name == target[0].name:
#            dist_pos_eq = np.linalg.norm(end_pos[0][0:2])
#            fit = abs(dist_pos_eq-target[2])+abs(end_pos[0][2])
#        else:
#            end_pos_abs_c = end_pos[0]+end_orb[7].GetAbsPos(endt)[0]
#            end_pos_abs_p = target[0].GetAbsPos(endt)[0]
#            dist_pos_eq = np.linalg.norm(end_pos_abs_c[0:2]-end_pos_abs_p[0:2])
#            fit = abs(dist_pos_eq-target[2])+abs(end_pos_abs_c[2]-end_pos_abs_p[2])


#target (cart) = [SOI, 'Circ', radius ]              
#target (cart) = [SOI, 'Cart', [np.array([x,y,z]),np.array([vx,vy,vz])] ]
#target (orb) = [SOI, 'Orb', [SMA, ECC, INC, LAN, LPE] ]
#(resu, end_pos, end_orb, pos_vel_dict, orb_dict, mindist_w_targ, mindist_w_targ_par)
#end_orb = [self.SMA, self.ECC, self.INC, self.LAN, self.LPE, self.EPH, self.MNA, self.SOI]
    
#    fit = float('inf')
#    if not resu:
#        fit = float('inf')
#    elif end_orb[7].name != target[0].name:
#        if mindist_w_targ[0]==0:
#            if target[0].SOI.code == 0:
#                if end_orb[7].SOI.code == 0:
#                    fit = 800000 + 100000*(1-np.clip(end_orb[1],None,2)/2)
#                else:
#                    fit = 1000000 + 100000*(1-np.clip(end_orb[1],None,2)/2)
#            else:
#                if mindist_w_targ_par[0]==0:
#                    if end_orb[7].SOI.code == 0:
#                        fit = 1800000 + 100000*(1-np.clip(end_orb[1],None,2)/2)
#                    else:
#                        fit = 2000000 + 100000*(1-np.clip(end_orb[1],None,2)/2)
#                elif mindist_w_targ_par[0]==1:
#                    fit = 1600000 + 100000*mindist_w_targ_par[1]/target[0].SOI.SOI.SOI_rad
#                elif mindist_w_targ_par[0]==2:
#                    fit = 1400000 + 100000*mindist_w_targ_par[1]/target[0].SOI.SOI.SOI_rad
#                elif mindist_w_targ_par[0]==3:
#                    fit = 1200000 + 10000*mindist_w_targ_par[1]/target[0].SOI.SOI_rad
#                else:
#                    fit = 2.2e6
#        elif mindist_w_targ[0]==1:
#            fit = 600000 + 100000*mindist_w_targ[1]/target[0].SOI.SOI_rad
#        elif mindist_w_targ[0]==2:
#            fit = 400000 + 100000*mindist_w_targ[1]/target[0].SOI.SOI_rad
#        elif mindist_w_targ[0]==3:
#            fit = 200000 + 10000*mindist_w_targ[1]/target[0].SOI_rad
#        else:
#            fit = 2.2e6
#    else:
#        if target[1] == 'Cart':
#            dist_pos = np.linalg.norm(target[2][0]-end_pos[0])
#            if type(target[2][1]) is np.ndarray:
#                dist_vel = np.linalg.norm(target[2][1]-end_pos[1])
#            else:
#                dist_vel = 0
#            fit = 50000*dist_pos/target[0].SOI_rad/2 + 100000*np.arctan(dist_vel/target[0].vel_carac)/np.pi
#        else:
#            score_SMA = 20000*abs(end_orb[0]-target[2][0])/target[2][0]
#            if target[2][1] == 0:
#                score_ECC = 40000*np.clip(end_orb[1],None,4)/4
#            else:
#                score_ECC = 20000*np.clip(abs(end_orb[1]-target[2][1]),None,4)/4 + 20000*distAng(end_orb[4],target[2][4]*np.pi/180)/np.pi
#            if target[2][2] == 0:
#                score_INC = 40000*end_orb[2]/np.pi
#            else:
#                score_INC = 20000*abs(end_orb[2]-target[2][2])/np.pi + 20000*distAng(end_orb[3],target[2][3]*np.pi/180)/np.pi
#            fit = score_SMA + score_ECC + score_INC
#    score_DV = 20000*np.arctan(DV_tot/10000)/np.pi
#    fit += score_DV
#    return fit



# 2) Sat
# -------

class Sat:
    def __init__(self, name, SMA, ECC, INC, LAN, LPE, EPH, MNA, SOI, solar_syst):
        self.name = name
        self.SOI = SOI
        self.solar_syst = solar_syst
        self.SMA = SMA
        self.ECC = ECC
        self.INC = INC*2*np.pi/360
        self.LAN = LAN*2*np.pi/360
        self.LPE = LPE*2*np.pi/360
        self.EPH = EPH
        self.MNA = MNA
        self.Mat_Rot = np.transpose(np.array([[cos(self.LPE)*cos(self.LAN)-sin(self.LPE)*cos(self.INC)*sin(self.LAN),
                                               -sin(self.LPE)*cos(self.LAN)-cos(self.LPE)*cos(self.INC)*sin(self.LAN),
                                                sin(self.INC)*sin(self.LAN)],
                   [cos(self.LPE)*sin(self.LAN)+sin(self.LPE)*cos(self.INC)*cos(self.LAN),
                    cos(self.LPE)*cos(self.LAN)*cos(self.INC)-sin(self.LPE)*sin(self.LAN),
                    -sin(self.INC)*cos(self.LAN)],
                   [sin(self.LPE)*sin(self.INC), cos(self.LPE)*sin(self.INC), cos(self.INC)]]))
        if ECC >= 1:
            self.T = float("inf")
        else:
            self.T = 2*np.pi*np.sqrt(SMA**3/SOI.mu)
        self.GetCartFromOrb(EPH)
        
    def retOrbDeg(self):
        return [self.SMA, self.ECC, self.INC*180/np.pi, self.LAN*180/np.pi, self.LPE*180/np.pi, self.EPH, self.MNA, self.SOI.name]
    
    def retOrbRad(self):
        return [self.SMA, self.ECC, self.INC, self.LAN, self.LPE, self.EPH, self.MNA, self.SOI.name]
            
    def reset(self, SMA, ECC, INC, LAN, LPE, EPH, MNA, SOI):
        self.SOI = SOI
        self.SMA = SMA
        self.ECC = ECC
        self.INC = INC*2*np.pi/360
        self.LAN = LAN*2*np.pi/360
        self.LPE = LPE*2*np.pi/360
        self.EPH = EPH
        self.MNA = MNA
        self.Mat_Rot = np.transpose(np.array([[cos(self.LPE)*cos(self.LAN)-sin(self.LPE)*cos(self.INC)*sin(self.LAN),
                                               -sin(self.LPE)*cos(self.LAN)-cos(self.LPE)*cos(self.INC)*sin(self.LAN),
                                                sin(self.INC)*sin(self.LAN)],
                   [cos(self.LPE)*sin(self.LAN)+sin(self.LPE)*cos(self.INC)*cos(self.LAN),
                    cos(self.LPE)*cos(self.LAN)*cos(self.INC)-sin(self.LPE)*sin(self.LAN),
                    -sin(self.INC)*cos(self.LAN)],
                   [sin(self.LPE)*sin(self.INC), cos(self.LPE)*sin(self.INC), cos(self.INC)]]))
        if ECC >= 1:
            self.T = float("inf")
        else:
            self.T = 2*np.pi*np.sqrt(SMA**3/SOI.mu)
        self.GetCartFromOrb(EPH)
            
    def updT(self):
        if self.ECC >= 1:
            self.T = float("inf")
        else:
            self.T = 2*np.pi*np.sqrt(self.SMA**3/self.SOI.mu)
            
    def MoveToEnd(self, startt, endt, burns, target, trace, sca_SOI):
        global n_it
        pos_vel_dict = {}
        orb_dict = {}
        if trace:
            pos_vel_dict[self.SOI.name] = [[startt,self.GetCartFromOrb(startt)]]
            orb_dict[self.SOI.name] = [[self.SMA, self.ECC, self.INC, self.LAN, self.LPE, self.SOI]]
        
        mindist_w_targ = [0,float('inf')]
        mindist_w_targ_par = [0,float('inf')]
        r_burns = [burn for burn in burns if burn[0]>startt and burn[0]<endt]
        s_burns = sorted(r_burns)
        DV_tot = 0
        for burn in burns:
            DV_tot += np.linalg.norm(burn[1])
        i_b = 0
        next_op = [endt,'','NA']
        currt = startt
        fin = False
        inter_score = []
        it = 0
        while not fin:
            it+=1
            next_op = [endt,'end','NA']
            if i_b<len(s_burns):
                if s_burns[i_b][0]<next_op[0]:
                    next_op = [s_burns[i_b][0],'burn',s_burns[i_b][1]]
            ttoleave = self.TimeToLeaveSOI(currt, sca_SOI) + currt
            if ttoleave<next_op[0]:
                next_op = [ttoleave,'leaveSOI','NA']
            ttocrash = self.TimeToCrash(currt) + currt
            if ttocrash<next_op[0]:
                next_op = [ttocrash,'crash','NA']
            [ttoenter, nextSOI, approach] = self.TimeToEnterSOI(currt, next_op[0], target[0], sca_SOI)
            if approach[2] != 'NA':
                if approach[2].name == target[0].name:
                    if approach[0]>mindist_w_targ[0]:
                        mindist_w_targ = [approach[0], approach[1]]
                    elif approach[1]<mindist_w_targ[1] and approach[0]==mindist_w_targ[0]:
                        mindist_w_targ[1] = approach[1]
                elif approach[2].name == target[0].SOI.name:
                    if approach[0]>mindist_w_targ_par[0]:
                        mindist_w_targ_par = [approach[0], approach[1]]
                    elif approach[1]<mindist_w_targ_par[1] and approach[0]==mindist_w_targ_par[0]:
                        mindist_w_targ_par[1] = approach[1]
            if ttoenter<next_op[0]:
                next_op = [ttoenter,'enterSOI',nextSOI]
            
            if trace:
                if self.SOI.name in pos_vel_dict:
                    pos_vel_dict[self.SOI.name] += [[currt,self.GetCartFromOrb(currt)]]
                    pos_vel_dict[self.SOI.name] += [[next_op[0],self.GetCartFromOrb(next_op[0])]]
                else:
                    pos_vel_dict[self.SOI.name] = [[currt,self.GetCartFromOrb(currt)]]
                    pos_vel_dict[self.SOI.name] += [[next_op[0],self.GetCartFromOrb(next_op[0])]]
                if self.SOI.name in orb_dict:
                    orb_dict[self.SOI.name] += [[self.SMA, self.ECC, self.INC, self.LAN, self.LPE]]
                else:
                    orb_dict[self.SOI.name] = [[self.SMA, self.ECC, self.INC, self.LAN, self.LPE]]
                
            if next_op[1]=='enterSOI':
                self.EnterSOI(next_op[2],next_op[0])
                if next_op[2].SOI.code == 0:
                    per = (1-self.ECC)*self.SMA
                    inter_score += [[next_op[2].alt_lim,per]]
                if next_op[2].name == target[0].name:
                    end_pos = self.GetCartFromOrb(next_op[0])
                    end_orb = [self.SMA, self.ECC, self.INC, self.LAN, self.LPE, self.EPH, self.MNA, self.SOI]
                    resu = True
                    fin = True
                    if trace:
                        if self.SOI.name in pos_vel_dict:
                            pos_vel_dict[self.SOI.name] += [[next_op[0],self.GetCartFromOrb(next_op[0])]]
                        else:
                            pos_vel_dict[self.SOI.name] = [[next_op[0],self.GetCartFromOrb(next_op[0])]]
                        if self.SOI.name in orb_dict:
                            orb_dict[self.SOI.name] += [[self.SMA, self.ECC, self.INC, self.LAN, self.LPE]]
                        else:
                            orb_dict[self.SOI.name] = [[self.SMA, self.ECC, self.INC, self.LAN, self.LPE]]
            elif next_op[1]=='leaveSOI':
                self.LeaveSOI(next_op[0])
            elif next_op[1]=='burn':
                self.Burn(False,next_op[2],next_op[0])
                i_b += 1
            elif next_op[1]=='end':
                end_pos = self.GetCartFromOrb(next_op[0])
                end_orb = [self.SMA, self.ECC, self.INC, self.LAN, self.LPE, self.EPH, self.MNA, self.SOI]
                resu = True
                fin = True
            elif next_op[1]=='crash':
                end_pos = self.GetCartFromOrb(next_op[0])
                end_orb = [self.SMA, self.ECC, self.INC, self.LAN, self.LPE, self.EPH, self.MNA, self.SOI]
                resu = False
                fin = True
            
            currt = next_op[0]
            if it > n_it*10:
                print(next_op)
                if next_op[2]!='NA':
                    print(next_op[2].name)
                print(self.ECC, self.INC, self.LAN, self.LPE, self.MNA, self.SMA, self.SOI.name)
                #print('more than '+str(n_it*10)+' iterations for move to end')
                if it > n_it*10.2:
                    print(burns)
                    break
                
        return (resu, end_pos, end_orb, pos_vel_dict, orb_dict, mindist_w_targ, mindist_w_targ_par, DV_tot, currt, inter_score)
            
    
    def GetAbsPos(self,UT):
        global n_it
        [X_abs, V_abs] = self.GetCartFromOrb(UT)
        plan = self.SOI
        while plan.code != 0:
            [X_plan, V_plan] = plan.GetCartFromOrb(UT)
            X_abs += X_plan
            V_abs += V_plan
            plan = plan.SOI
        return [X_abs, V_abs]
    
    def GetCartFromOrb(self, UT):
        SMA = self.SMA
        ECC = self.ECC
        EPH = self.EPH
        MNA = self.MNA
        mu = self.SOI.mu
        if ECC==1:
            ECC=0.99999999
        if ECC<1:
            M = (MNA + np.sqrt(mu/SMA**3)*(UT-EPH))%(2*np.pi)
            E = getE(ECC, M)
            nu = (2*np.arctan2(np.sqrt(1+ECC)*sin(E/2),np.sqrt(1-ECC)*cos(E/2)))%(2*np.pi)
            r = SMA*(1-ECC*cos(E))
            O = np.array([r*cos(nu), r*sin(nu), 0])
            dO = np.array([-sin(E)*np.sqrt(mu*SMA)/r, np.sqrt(1-ECC**2)*cos(E)*np.sqrt(mu*SMA)/r, 0])
            X = O@self.Mat_Rot
            Vc = dO@self.Mat_Rot
        else:
            N = MNA + np.sqrt(mu/abs(SMA)**3)*(UT-EPH)
            H = getH(ECC,N)
            f = (2*np.arctan2(np.sqrt(1+ECC)*np.sinh(H/2),np.sqrt(ECC-1)*np.cosh(H/2)))%(2*np.pi)
            r = SMA*(1-ECC*np.cosh(H))
            O = np.array([r*cos(f), r*sin(f), 0])
            vel = np.sqrt(mu*(2/r-1/SMA))
            phi = np.arctan2(ECC*sin(f),1+ECC*cos(f))%(2*np.pi)
            dO = np.array([-vel*sin(f-phi), vel*cos(f-phi), 0])
            X = O@self.Mat_Rot
            Vc = dO@self.Mat_Rot
        self.X = X
        self.Vc = Vc
        return [X, Vc]
        
    def GetOrbFromCart(self, UT):
        mu = self.SOI.mu
        h = np.cross(self.X, self.Vc)
        hnorm = np.linalg.norm(h)
        r = np.linalg.norm(self.X)
        vel = np.linalg.norm(self.Vc)
        
        sE = vel**2/2 - mu/r
        #if sE == 0:
            #para todo (oupas)
        self.SMA = - mu/(2*sE)
        self.ECC = np.sqrt(1-hnorm**2/(self.SMA*mu))
        self.INC = np.arccos(h[2]/hnorm)
        if self.INC == 0.0:
            self.LAN = 0.0
            arglat = np.arctan2(self.X[1],self.X[0])%(2*np.pi)
        elif self.INC == 90.0:
            self.LAN = np.arctan2(h[0],-h[1])%(2*np.pi)
            arglat = np.arctan2(self.X[2], np.sqrt(self.X[0]**2+self.X[1]**2))%(2*np.pi)
        else:
            self.LAN = np.arctan2(h[0],-h[1])%(2*np.pi)
            sinbet = self.X[2]/r
            lamb = np.arctan2(self.X[1],self.X[0])%(2*np.pi)
            if (lamb-self.LAN)%(np.pi) == np.pi/2:
                arglat = (lamb-self.LAN)%(2*np.pi)
            else:
                tan_arglat = np.tan(lamb-self.LAN)/cos(self.INC)
                if sinbet>0 and tan_arglat>0:
                    arglat = np.arcsin(sinbet/sin(self.INC))
                elif sinbet<0 and tan_arglat>0:
                    arglat = np.pi - np.arcsin(sinbet/sin(self.INC))
                elif sinbet<0 and tan_arglat<0:
                    arglat = 2*np.pi + np.arcsin(sinbet/sin(self.INC))
                elif sinbet>0 and tan_arglat<0:
                    arglat = np.pi - np.arcsin(sinbet/sin(self.INC))
                else:
                    if self.Vc[2]>0:
                        arglat = 0
                    else:
                        arglat = np.pi
        if self.ECC>0.0000001:
            cosnu = (self.SMA*(1-self.ECC**2)-r)/(self.ECC*r)
            if cosnu>1:
                cosnu = 1
            elif cosnu<-1:
                cosnu = -1
            if np.dot(self.X,self.Vc)>0:
                nu = np.arccos(cosnu)
            else:
                nu = 2*np.pi - np.arccos(cosnu)
            self.LPE = (arglat - nu)%(2*np.pi)
            if sE<0:
                E = (2*np.arctan2(np.sqrt(1-self.ECC)*sin(nu/2),np.sqrt(1+self.ECC)*cos(nu/2)))%(2*np.pi)
                self.MNA = E - self.ECC*sin(E)
            else:
                E = (2*np.arctanh(np.sqrt((self.ECC-1)/(1+self.ECC))*np.tan(nu/2)))
                self.MNA = self.ECC*np.sinh(E)-E
        else:
            n = np.cross(np.array([0,0,1]),h)
            if np.linalg.norm(n)==0:
                r_un = self.X/r
                cosnu = np.dot(np.array([1,0,0]),r_un)
                self.LPE = 0
                if self.X[1]>0:
                    self.MNA = np.arccos(cosnu)
                else:
                    self.MNA = 2*np.pi - np.arccos(cosnu)
            else:
                n_un = n/np.linalg.norm(n)
                r_un = self.X/r
                cosnu = np.dot(n_un,r_un)
                self.LPE = 0
                if self.X[2]>0:
                    self.MNA = np.arccos(cosnu)
                else:
                    self.MNA = 2*np.pi - np.arccos(cosnu)
        self.EPH = UT
        LAN = self.LAN
        LPE = self.LPE
        INC = self.INC
        self.Mat_Rot = np.transpose(np.array([[cos(LPE)*cos(LAN)-sin(LPE)*cos(INC)*sin(LAN), -sin(LPE)*cos(LAN)-cos(LPE)*cos(INC)*sin(LAN), sin(INC)*sin(LAN)],
                   [cos(LPE)*sin(LAN)+sin(LPE)*cos(INC)*cos(LAN), cos(LPE)*cos(LAN)*cos(INC)-sin(LPE)*sin(LAN), -sin(INC)*cos(LAN)],
                   [sin(LPE)*sin(INC), cos(LPE)*sin(INC), cos(INC)]]))
        self.updT()
        
    def Burn(self, cart, DV, UT):
        self.GetCartFromOrb(UT)
        if cart:
            self.Vc = self.Vc + DV
        else:
#            SMA = self.SMA
#            ECC = self.ECC
#            EPH = self.EPH
#            MNA = self.MNA
#            mu = self.SOI.mu
#            if ECC==1:
#                ECC=1.000001
#            if ECC<1:
#                M = (MNA + np.sqrt(mu/SMA**3)*(UT-EPH))%(2*np.pi)
#                E = getE(ECC, M)
##                nu = (2*np.arctan2(np.sqrt(1+ECC)*sin(E/2),np.sqrt(1-ECC)*cos(E/2)))%(2*np.pi)
#                r = SMA*(1-ECC*cos(E))
##                O = np.array([r*cos(nu), r*sin(nu), 0])
#                dO = np.array([-sin(E)*np.sqrt(mu*SMA)/r, np.sqrt(1-ECC**2)*cos(E)*np.sqrt(mu*SMA)/r, 0])
#            else:
#                N = MNA + np.sqrt(mu/abs(SMA)**3)*(UT-EPH)
#                H = getH(ECC, N)
#                f = (2*np.arctan2(np.sqrt(1+ECC)*np.sinh(H/2),np.sqrt(ECC-1)*np.cosh(H/2)))%(2*np.pi)
#                r = SMA*(1-ECC*np.cosh(H))
##                O = np.array([r*cos(f), r*sin(f), 0])
#                vel = np.sqrt(mu*(2/r-1/SMA))
#                phi = (np.arctan2(ECC*sin(f),1+ECC*cos(f)))%(2*np.pi)
#                dO = np.array([-vel*sin(f-phi), vel*cos(f-phi), 0])
            
            vec_tang = self.Vc/np.linalg.norm(self.Vc)
            vec_n = np.cross(self.X,vec_tang)
            vec_norm = vec_n/np.linalg.norm(vec_n)
            vec_rad = np.cross(vec_tang, vec_norm)
#            vec_norm = np.array([vec_tang[1],-vec_tang[0],0])
#            dVcO = vec_tang*DV[2] + vec_norm*DV[0] + np.array([0,0,DV[1]])
#            dVcC = dVcO@self.Mat_Rot
            dVcC = DV[2]*vec_tang + DV[0]*vec_rad + DV[1]*vec_norm
            self.Vc = self.Vc + dVcC
        self.GetOrbFromCart(UT)
        
    def LeaveSOI(self, UT):
        nextSOI = self.SOI.SOI
        [pos_c, vel_c] = self.GetCartFromOrb(UT)
        [pos_plan, vel_plan] = self.SOI.GetCartFromOrb(UT)
        nextPos = pos_c + pos_plan
        nextVel = vel_c + vel_plan
        self.X = nextPos
        self.Vc = nextVel
        self.SOI = nextSOI
        self.GetOrbFromCart(UT)
        
    def EnterSOI(self, nextSOI, UT):
        [pos_c, vel_c] = self.GetCartFromOrb(UT)
        [pos_plan, vel_plan] = nextSOI.GetCartFromOrb(UT)
        nextPos = pos_c - pos_plan
        nextVel = vel_c - vel_plan
        self.X = nextPos
        self.Vc = nextVel
        self.SOI = nextSOI
        self.GetOrbFromCart(UT)
    
    def TimeToCrash(self, UT):
        per = (1-self.ECC)*self.SMA
        if per > self.SOI.alt_lim:
            t = float('inf')
        else:
            r = self.SOI.alt_lim
            n = np.sqrt(self.SOI.mu/abs(self.SMA)**3)
            if self.SMA < 0:
                N = self.MNA + n*(UT - self.EPH)
                if N<0:
                    H = -np.arccosh((self.SMA-r)/(self.ECC*self.SMA))
                    N1 = self.ECC*np.sinh(H) - H
                    t = (N1-N)/n
                else:
                    t = float('inf')
            else:
                M = (self.MNA + n*(UT - self.EPH))%(2*np.pi)
                E2 = 2*np.pi - np.arccos((self.SMA-r)/(self.ECC*self.SMA))
                M2 = E2 - self.ECC*sin(E2)
                t = ((M2-M)%(2*np.pi))/n
        return t
    
    def TimeToLeaveSOI(self, UT, sca_SOI):
        if self.SOI.code == 0:
            t = float('inf')
        else:
            r = (self.SOI.SOI_rad*(1-sca_SOI) + self.SOI.clos_rad*sca_SOI)*1.00001                  # /!\ futur problemes ?
            n = np.sqrt(self.SOI.mu/abs(self.SMA)**3)
            apo = (1+self.ECC)*self.SMA
            per = (1-self.ECC)*self.SMA
            if r < per:
                t = 0
            else:
                if apo < 0:
                    N = self.MNA + n*(UT - self.EPH)
                    H = np.arccosh((self.SMA-r)/(self.ECC*self.SMA))
                    N1 = self.ECC*np.sinh(H) - H
                    if (abs(N)<N1):
                        t = (N1-N)/n
                    else:
                        t = 0
                elif apo > r:
                    M = (self.MNA + n*(UT - self.EPH))%(2*np.pi)
                    E1 = np.arccos((self.SMA-r)/(self.ECC*self.SMA))
                    E2 = 2*np.pi - E1
                    M1 = E1 - self.ECC*sin(E1)
                    M2 = E2 - self.ECC*sin(E2)
                    if (M>=M1) and (M<=M2):
                        t = 0
                    else:
                        t = ((M1-M)%(2*np.pi))/n
                else:
                    t = float("inf")
        return t

    def TimeToEnterSOI(self, UT, nextt, targ, sca_SOI):
        global n_it
        global pos_vel_list
        SOI = self.SOI
        SOI_poss = SOI.enfants
        SOI_poss_bis = []
        SOI_poss_ter = []
        n = np.sqrt(SOI.mu/abs(self.SMA)**3)
        approach = [0, float('inf'), 'NA']
        app_calc = [False, float('inf'),float('inf')]
        resu = [float('inf'), avoidcases]
        
#        print([pl.code for pl in SOI_poss])
        
        if self.ECC >= 1:
            apo = float('inf')
            Mc = self.MNA + n*(UT - self.EPH)
        else:
            apo = (1+self.ECC)*self.SMA
            Mc = (self.MNA + n*(UT - self.EPH))%(2*np.pi)
        tfp_c = Mc/n
        per = (1-self.ECC)*self.SMA
        Wc = np.array([sin(self.LAN)*sin(self.INC), cos(self.LAN)*sin(self.INC), cos(self.INC)])
        p_c = self.SMA*(1-self.ECC**2)
        for plan in SOI_poss:
            D = plan.SOI_rad*(1-sca_SOI) + plan.clos_rad*sca_SOI
            if not(min(apo, plan.apo) < (max(per, plan.per)-D)):
                SOI_poss_bis.append(plan)
            elif plan.name == targ.name or plan.name == targ.SOI.name:
                approach = [1,(max(per, plan.per)-D)-min(apo, plan.apo), plan]
        
#        print([pl.code for pl in SOI_poss_bis])
        
        for plan in SOI_poss_bis:
            D = plan.SOI_rad*(1-sca_SOI) + plan.clos_rad*sca_SOI
            p_p = plan.SMA*(1-plan.ECC**2)
            
            Wp = np.array([sin(plan.LAN)*sin(plan.INC), cos(plan.LAN)*sin(plan.INC), cos(plan.INC)])
            K = np.cross(Wp, Wc)
            cosIr = np.dot(Wp,Wc)
            sinIr = np.linalg.norm(K)
            
            if sinIr!=0:
                if (1/sinIr)*sin(self.INC)*sin(self.LAN-plan.LAN)>0:
                    Dp = np.arccos((1/sinIr)*(sin(self.INC)*cos(plan.INC)*cos(self.LAN-plan.LAN)-sin(plan.INC)*cos(self.INC)))
                else:
                    Dp = -np.arccos((1/sinIr)*(sin(self.INC)*cos(plan.INC)*cos(self.LAN-plan.LAN)-sin(plan.INC)*cos(self.INC)))
                if (1/sinIr)*sin(plan.INC)*sin(self.LAN-plan.LAN)>0:
                    Dc = np.arccos((1/sinIr)*(sin(self.INC)*cos(plan.INC)-sin(plan.INC)*cos(self.INC)*cos(self.LAN-plan.LAN)))
                else:
                    Dc = -np.arccos((1/sinIr)*(sin(self.INC)*cos(plan.INC)-sin(plan.INC)*cos(self.INC)*cos(self.LAN-plan.LAN)))
            else:
                Dp = plan.LPE
                Dc = self.LPE
            
            if apo-per < plan.apo+D-(plan.per-D):
                quasicirc=True
                fc_init1 = Dc-self.LPE
                fp_init1 = Dp-plan.LPE
            else:
                quasicirc=False
                #mean_r_p = min([max([(plan.per+plan.apo)/2, per-D]), apo-D])
                mean_r_p = (max([per, plan.per-D])+min(apo, plan.apo+D))/2
                fc_init1 = np.arccos((p_c-mean_r_p)/(self.ECC*mean_r_p))
                Oc_init1 = np.array([mean_r_p*cos(fc_init1), mean_r_p*sin(fc_init1), 0])
                Xc_init1 = Oc_init1@self.Mat_Rot
                fp_init1 = (np.arctan2(Xc_init1[1],Xc_init1[0])-plan.LAN-plan.LPE)%(2*np.pi)
            
            ayp = plan.ECC*sin(plan.LPE-Dp)
            axp = plan.ECC*cos(plan.LPE-Dp)
            ayc = self.ECC*sin(self.LPE-Dc)
            axc = self.ECC*cos(self.LPE-Dc)
            
            (fpsol1, fcsol1, rpsol1, rcsol1, Dsol1) = self.NewtMinDist(plan, fc_init1, fp_init1, ayc, axc, ayp, axp, Dp, Dc, cosIr, D)
            
            if quasicirc:
                fc_init2 = fcsol1 + np.pi
                fp_init2 = fpsol1 + np.pi
            else:
                fc_init2 = 2*np.pi - fc_init1
                Oc_init2 = np.array([mean_r_p*cos(fc_init2), mean_r_p*sin(fc_init2), 0])
                Xc_init2 = Oc_init2@self.Mat_Rot
                fp_init2 = (np.arctan2(Xc_init2[1],Xc_init2[0])-plan.LAN-plan.LPE)%(2*np.pi)
            
            (fpsol2, fcsol2, rpsol2, rcsol2, Dsol2) = self.NewtMinDist(plan, fc_init2, fp_init2, ayc, axc, ayp, axp, Dp, Dc, cosIr, D)
            
#            print(Dsol1, Dsol2)
            if Dsol1<D or Dsol2<D:
                SOI_poss_ter.append([plan, [ayc, axc, D, sinIr, Dc, quasicirc, ayp, axp, p_p, Dp]])
            if plan.name == targ.name or plan.name == targ.SOI.name:
                #approach = [2,min(Dsol1, Dsol2), plan]
                app_calc = [True, fcsol1, fcsol2, plan]
                
#        print([pl[0].name for pl in SOI_poss_ter])
        
        t_lim = nextt
        for plan_and_var in SOI_poss_ter:
            plan = plan_and_var[0]
            [ayc, axc, D, sinIr, Dc, quasicirc, ayp, axp, p_p, Dp] = plan_and_var[1]
            tarplan = False
            parplan = False
            closest_app = float('inf')
            if plan.name == targ.name:
                tarplan = True
            elif plan.name == targ.SOI.name:
                parplan = True
            
            P_p = plan.T
            Mp = (plan.MNA + 2*np.pi*UT/P_p)%(2*np.pi)
            tfp_p = Mp*P_p/(2*np.pi)
        
            vuln_ang_c = self.getVulnWin(self, ayc, axc, D, sinIr, p_c, Dc, quasicirc, plan.apo, plan.per)
            vuln_ang_p = self.getVulnWin(plan, ayp, axp, D, sinIr, p_p, Dp, True, apo, per)
            
            if self.ECC>=1:
                vuln_t_c = []
                marq = True
                prevt = 0
                for f in vuln_ang_c:
                    E = (2*np.arctanh(np.sqrt((self.ECC-1)/(1+self.ECC))*np.tan(f/2)))
                    M = self.ECC*np.sinh(E)-E
                    t = M*np.sqrt(-self.SMA**3/SOI.mu)
                    if marq:
                        prevt = t
                    else:
                        vuln_t_c += [[(t+prevt)/2,abs(t-prevt)/2]]
                    marq = not marq
                if len(vuln_t_c)==2:
                    if vuln_t_c[1][0]<0:
                        win1 = vuln_t_c[1]
                        win2 = vuln_t_c[0]
                        vuln_t_c = [win1,win2]
            else:
                vuln_t_c = []
                marq = True
                prevt = 0
                for f in vuln_ang_c:
                    E = (2*np.arctan2(np.sqrt(1-self.ECC)*sin(f/2),np.sqrt(1+self.ECC)*cos(f/2)))%(2*np.pi)
                    if f>=2*np.pi:
                        E += 2*np.pi
                    M = E - self.ECC*sin(E)
                    t = M*self.T/(2*np.pi)
                    if marq:
                        prevt = t
                    else:
                        vuln_t_c += [[(t+prevt)/2,abs(t-prevt)/2]]
                    marq = not marq
            
            if vuln_ang_p==[0,2*np.pi]:
                vuln_t_p_b = [[P_p/2,P_p/2]]
                vuln_t_p = [[P_p/2,P_p/2]]
            else:
                vuln_t_p_b = []
                marq = True
                prevt = 0
                for f in vuln_ang_p:
                    E = (2*np.arctan2(np.sqrt(1-plan.ECC)*sin(f/2),np.sqrt(1+plan.ECC)*cos(f/2)))%(2*np.pi)
                    if f>=2*np.pi:
                        E += 2*np.pi
                    M = E - plan.ECC*sin(E)
                    t = M*P_p/(2*np.pi)
                    if marq:
                        prevt = t
                    else:
                        vuln_t_p_b += [[(t+prevt)/2,abs(t-prevt)/2]]
                    marq = not marq
                vuln_t_p = vuln_t_p_b[:]
                if vuln_t_p_b[-1][0]>=P_p/2:
                    vuln_t_p = [[vuln_t_p_b[-1][0]-P_p, vuln_t_p_b[-1][1]]] + vuln_t_p
                if vuln_t_p_b[0][0]<=P_p/2:
                    vuln_t_p = vuln_t_p + [[vuln_t_p_b[0][0]+P_p, vuln_t_p_b[0][1]]]
            
#            print("vuln_t_p :")
#            print(vuln_t_p)
#            print("vuln_t_c :")
#            print(vuln_t_c)
#            for windp in vuln_t_p:
#                pos_vel_list += [plan.GetCartFromOrb(windp[0]-tfp_p)]
#            for windc in vuln_t_c:
#                pos_vel_list += [self.GetCartFromOrb(windc[0]-tfp_c, SOI)]
            
#            print(tfp_p, tfp_c)
            rap_c = 0
            rap_p = 0
            for vuln_win_t_c in vuln_t_c:
                rap_c += 2*vuln_win_t_c[1]
            for vuln_win_t_p in vuln_t_p_b:
                rap_p += 2*vuln_win_t_p[1]
            
#            print(vuln_ang_c)
#            print(plan.name)
#            print(rap_c/self.T, rap_p/P_p)
            if (rap_c/self.T)*(rap_p/P_p)>0.5:
                per_check = min(P_p, self.T)/5
                dt = UT-per_check
                R = 0
                prevR = 0
                fl_first = True
                while dt<t_lim+2*per_check:
                    prevR = R
                    [rc,drc] = self.GetCartFromOrb(dt)
                    [rp,drp] = plan.GetCartFromOrb(dt)
                    R = np.dot(drc-drp,rc-rp)
                    if fl_first:
                        fl_first=False
                    else:
                        if np.sign(R)!=np.sign(prevR):
                            nt = dt-per_check + per_check*abs(prevR)/(abs(R)+abs(prevR))
                            
                            (resumod, resu1, appmod, approach1) = findClosestApp(nt, self, plan, t_lim, D, UT, tarplan, parplan, closest_app)
                            if resumod:
                                resu = resu1
                                t_lim = resu[0]
                            if appmod:
                                approach = approach1
                                closest_app = approach[1]
                    dt += per_check
                            
            elif self.ECC>=1:
                for v_c in vuln_t_c:
                    if v_c[0]-tfp_c-v_c[1]<t_lim:
                        tc_in_p = (tfp_p+v_c[0]-tfp_c)%P_p
    #                    print("tc_in_p : "+str(tc_in_p))
                        for v_p in vuln_t_p:
    #                        print("v_p : "+str(v_p))
                            if abs(tc_in_p - v_p[0]) < v_c[1]+v_p[1]:
                                nt = UT + (max(tc_in_p-v_c[1], v_p[0]-v_p[1]) + min(tc_in_p+v_c[1], v_p[0]+v_p[1]))/2 - tc_in_p + v_c[0] - tfp_c
    #                            print("nt : "+str(nt))
                                (resumod, resu1, appmod, approach1) = findClosestApp(nt, self, plan, t_lim, D, UT, tarplan, parplan, closest_app)
                                if resumod:
                                    resu = resu1
                                    t_lim = resu[0]
                                if appmod:
                                    approach = approach1
                                    closest_app = approach[1]
                                    
            else:
                dt = 0
                while UT+dt<t_lim+self.T/2:
                    for v_c in vuln_t_c:
                        for v_p in vuln_t_p:
                            tc_in_p = (tfp_p+v_c[0]+dt-tfp_c)%P_p
                            if abs(tc_in_p - v_p[0]) < v_c[1]+v_p[1]:
                                nt = UT + (max(tc_in_p-v_c[1], v_p[0]-v_p[1]) + min(tc_in_p+v_c[1], v_p[0]+v_p[1]))/2 - tc_in_p + v_c[0] + dt - tfp_c
                                
                                (resumod, resu1, appmod, approach1) = findClosestApp(nt, self, plan, t_lim, D, UT, tarplan, parplan, closest_app)
                                if resumod:
                                    resu = resu1
                                    t_lim = resu[0]
                                if appmod:
                                    approach = approach1
                                    closest_app = approach[1]
                    dt += self.T
            
        if app_calc[0]:  ##################################### attention periodes courtes ! #########
            if resu[1] == 'NA' or not (resu[1].name == targ.name or resu[1].name == targ.SOI.name):
                dt = UT
                tsol = []
                if self.ECC>=1:
                    for f in app_calc[1:3]:
                        E = (2*np.arctanh(np.sqrt((self.ECC-1)/(1+self.ECC))*np.tan(f/2)))
                        M = self.ECC*np.sinh(E)-E
                        t = M*np.sqrt(-self.SMA**3/SOI.mu)
                        tsol += [t]
                    for ts in tsol:
                        abs_t = UT+ts-tfp_c
                        if UT<abs_t and abs_t<t_lim:
                            dist = np.linalg.norm(self.GetCartFromOrb(abs_t)[0]-app_calc[3].GetCartFromOrb(abs_t)[0])
                            if dist < approach[1]:
                                approach = [3, dist, app_calc[3]]
                else:
                    for f in app_calc[1:3]:
                        E = (2*np.arctan2(np.sqrt(1-self.ECC)*sin(f/2),np.sqrt(1+self.ECC)*cos(f/2)))%(2*np.pi)
                        if f>=2*np.pi:
                            E += 2*np.pi
                        M = E - self.ECC*sin(E)
                        t = M*self.T/(2*np.pi)
                        tsol += [t]
                    while dt < t_lim + self.T/2:
                        for ts in tsol:
                            abs_t = dt+ts-tfp_c
                            if UT<abs_t and abs_t<t_lim:
                                dist = np.linalg.norm(self.GetCartFromOrb(abs_t)[0]-app_calc[3].GetCartFromOrb(abs_t)[0])
                                if dist < approach[1]:
                                    approach = [3, dist, app_calc[3]]
                        dt += self.T
                            
                    
            
        return [resu[0], resu[1], approach]
                    
                    
                
                
                    
            
                    
                    
    def getVulnWin(self, body, ay, ax, D, sinIr, p, delt, quasicirc, apo, per):
        global pos_vel_list
        if body.ECC<1:
            inter1 = self.getVulnAng(body, ay, ax, D, sinIr, p, delt)
        else:
            inter1 = [0,2*np.pi]
        if not all(inter1[i] <= inter1[i+1] for i in range(len(inter1)-1)):
            print("liste des angles pas dans l'ordre")
        if not quasicirc:
            inter2 = self.getVulnRad(body, D, p, apo, per)
        else:
            inter2 = [0,2*np.pi]
        
        vuln_win_b = inter_intersec(inter1, inter2)
        
        if vuln_win_b[0]==0 and vuln_win_b[-1]==2*np.pi and len(vuln_win_b)>2:
            vuln_win = vuln_win_b[2:-1] + [vuln_win_b[1]+2*np.pi]
        else:
            vuln_win = vuln_win_b
            
#        for f in vuln_win:
#            pos_vel_list += [getCartFromf(body, f)]
        
        return vuln_win
        

                    
    def getVulnAng(self, body, ay, ax, D, sinIr, p, delt):
        global pos_vel_list
        Q1 = p*sinIr*(p*sinIr - 2*D*ay)-(1-body.ECC**2)*D**2
        Q2 = p*sinIr*(p*sinIr + 2*D*ay)-(1-body.ECC**2)*D**2
        
        if Q1<0:
            if Q2<0:
                return [0,2*np.pi]
            else:
                cosUrMo1 = (-ax*D**2 + (p*sinIr+D*ay)*np.sqrt(Q2))/(Q2 + D**2)
                cosUrMo2 = (-ax*D**2 - (p*sinIr+D*ay)*np.sqrt(Q2))/(Q2 + D**2)
                
                if abs(cosUrMo1)>1 or abs(cosUrMo2)>1:
                    return [0,2*np.pi]
                else:
                    if cosUrMo1<cosUrMo2:
                        Ur1 = 2*np.pi - np.arccos(cosUrMo2)
                        Ur2 = 2*np.pi - np.arccos(cosUrMo1)
                    else:
                        Ur1 = 2*np.pi - np.arccos(cosUrMo1)
                        Ur2 = 2*np.pi - np.arccos(cosUrMo2)
                    
                    f_1 = (Ur1 + delt - body.LPE)%(2*np.pi)
                    f_2 = (Ur2 + delt - body.LPE)%(2*np.pi)
                    if f_2<f_1:
                        return [0,f_2,f_1,2*np.pi]
                    else:
                        return [f_1,f_2]
        else:
            if Q2<0:
                cosUrPl1 = (-ax*D**2 + (p*sinIr-D*ay)*np.sqrt(Q1))/(Q1 + D**2)
                cosUrPl2 = (-ax*D**2 - (p*sinIr-D*ay)*np.sqrt(Q1))/(Q1 + D**2)
                
                if abs(cosUrPl1)>1 or abs(cosUrPl2)>1:
                    return [0,2*np.pi]
                else:
                    if cosUrPl1<cosUrPl2:
                        Ur1 = np.arccos(cosUrPl1)
                        Ur2 = np.arccos(cosUrPl2)
                    else:
                        Ur1 = np.arccos(cosUrPl2)
                        Ur2 = np.arccos(cosUrPl1)
                        
                    f_1 = (Ur1 + delt - body.LPE)%(2*np.pi)
                    f_2 = (Ur2 + delt - body.LPE)%(2*np.pi)
                    if f_2<f_1:
                        return [0,f_2,f_1,2*np.pi]
                    else:
                        return [f_1,f_2]
            else:
                cosUrPl1 = (-ax*D**2 + (p*sinIr-D*ay)*np.sqrt(Q1))/(Q1 + D**2)
                cosUrPl2 = (-ax*D**2 - (p*sinIr-D*ay)*np.sqrt(Q1))/(Q1 + D**2)
                cosUrMo1 = (-ax*D**2 + (p*sinIr+D*ay)*np.sqrt(Q2))/(Q2 + D**2)
                cosUrMo2 = (-ax*D**2 - (p*sinIr+D*ay)*np.sqrt(Q2))/(Q2 + D**2)
                
                if abs(cosUrPl1)>1 or abs(cosUrPl2)>1 or abs(cosUrMo1)>1 or abs(cosUrMo2)>1:
                    return [0,2*np.pi]
                else:
                    if cosUrPl1<cosUrPl2:
                        Ur1 = np.arccos(cosUrPl1)
                        Ur4 = np.arccos(cosUrPl2)
                    else:
                        Ur1 = np.arccos(cosUrPl2)
                        Ur4 = np.arccos(cosUrPl1)
                        
                    if cosUrMo1<cosUrMo2:
                        Ur2 = 2*np.pi - np.arccos(cosUrMo1)
                        Ur3 = 2*np.pi - np.arccos(cosUrMo2)
                    else:
                        Ur2 = 2*np.pi - np.arccos(cosUrMo2)
                        Ur3 = 2*np.pi - np.arccos(cosUrMo1)
                    
#                UrPl1 = np.arccos(cosUrPl1)
#                UrPl2 = np.arccos(cosUrPl2)
#                UrMo1 = 2*np.pi - np.arccos(cosUrMo1)
#                UrMo2 = 2*np.pi - np.arccos(cosUrMo2)
                
                f_1 = (Ur1 + delt - body.LPE)%(2*np.pi)
                f_2 = (Ur2 + delt - body.LPE)%(2*np.pi)
                f_3 = (Ur3 + delt - body.LPE)%(2*np.pi)
                f_4 = (Ur4 + delt - body.LPE)%(2*np.pi)
                
                if f_2<f_1:
                    return [0,f_2,f_3,f_4,f_1,2*np.pi]
                elif f_4<f_3:
                    return [0,f_4,f_1,f_2,f_3,2*np.pi]
                elif f_3<f_2:
                    return [f_3,f_4,f_1,f_2]
                else:
                    return [f_1,f_2,f_3,f_4]
                    
            
    def getVulnRad(self, body, D, p, apo, per):
        r_int = per-D
        r_ext = apo+D
        cosf_int = (p-r_int)/(body.ECC*r_int)
        cosf_ext = (p-r_ext)/(body.ECC*r_ext)
        if abs(cosf_int)>1:
            if abs(cosf_ext)>1:
                return [0,2*np.pi]
            else:
                return [0, np.arccos(cosf_ext), 2*np.pi - np.arccos(cosf_ext), 2*np.pi]
        else:
            if abs(cosf_ext)>1:
                return [np.arccos(cosf_int), 2*np.pi - np.arccos(cosf_int)]
            else:
                f_1 = np.arccos(cosf_int)
                f_2 = np.arccos(cosf_ext)
                f_3 = 2*np.pi - np.arccos(cosf_ext)
                f_4 = 2*np.pi - np.arccos(cosf_int)
                return [f_1,f_2,f_3,f_4]
        
        

    def NewtMinDist(self, plan, fc_in,fp_in, ayc, axc, ayp, axp, Dp, Dc, cosIr, SOI_rad):
        global pos_vel_list
        global limit_iterations
        global n_it
        e_c = self.ECC
        e_p = plan.ECC
        fc = fc_in
        fp = fp_in
        sq_rad = SOI_rad*SOI_rad
        
        Urp = fp + plan.LPE - Dp
        Urc = fc + self.LPE - Dc
        rp = plan.SMA*(1-e_p**2)/(1+e_p*cos(fp))
        rc = self.SMA*(1-e_c**2)/(1+e_c*cos(fc))
        cosGa = cos(Urc)*cos(Urp)+sin(Urc)*sin(Urp)*cosIr
        A = sin(Urc) + ayc
        B = cos(Urc) + axc
        C = sin(Urp) + ayp
        D = cos(Urp) + axp
        Eq1 = rc*e_c*sin(fc)+rp*(cos(Urp)*A-sin(Urp)*cosIr*B)
        Eq2 = rp*e_p*sin(fp)+rc*(cos(Urc)*C-sin(Urc)*cosIr*D)
        sq_dist_cp = rp*rp + rc*rc - 2*rc*rp*cosGa
        
        #pos_vel_list += [[0,getCartFromf(self, fc)]]
        #pos_vel_list += [[0,getCartFromf(plan, fp)]]
        
        it=0
        while (abs(Eq1)>1000 or abs(Eq2)>1000) and sq_dist_cp>sq_rad:
            it += 1
            cosEc = (e_c+cos(fc))/(1+e_c*cos(fc))
            cosEp = (e_p+cos(fp))/(1+e_p*cos(fp))
            F = rc*e_c*sin(fc)+rp*(A*cos(Urp)-B*cosIr*sin(Urp))
            G = rp*e_p*sin(fp)+rc*(C*cos(Urc)-D*cosIr*sin(Urc))
            dFdfc = rc*e_c*cosEc + rp*cosGa
            dFdfp = -rp*(A*C+B*D*cosIr)/(1+e_p*cos(fp))
            dGdfc = -rc*(A*C+B*D*cosIr)/(1+e_c*cos(fc))
            dGdfp = rp*e_p*cosEp + rc*cosGa
            fc = fc + 1*(F*dGdfp-G*dFdfp)/(dFdfp*dGdfc-dFdfc*dGdfp)
            fp = fp + 1*(G*dFdfc-F*dGdfc)/(dFdfp*dGdfc-dFdfc*dGdfp)
            Urp = fp + plan.LPE - Dp
            Urc = fc + self.LPE - Dc
            rp = plan.SMA*(1-e_p**2)/(1+e_p*cos(fp))
            rc = self.SMA*(1-e_c**2)/(1+e_c*cos(fc))
            cosGa = cos(Urc)*cos(Urp)+sin(Urc)*sin(Urp)*cosIr
            A = sin(Urc) + ayc
            B = cos(Urc) + axc
            C = sin(Urp) + ayp
            D = cos(Urp) + axp
            Eq1 = rc*e_c*sin(fc)+rp*(cos(Urp)*A-sin(Urp)*cosIr*B)
            Eq2 = rp*e_p*sin(fp)+rc*(cos(Urc)*C-sin(Urc)*cosIr*D)
            sq_dist_cp = rp*rp + rc*rc - 2*rc*rp*cosGa
            
            if it > n_it:
                break
        
        #print(it)
        #pos_vel_list += [[0,getCartFromf(self, fc)]]
        #pos_vel_list += [[0,getCartFromf(plan, fp)]]
        return(fp, fc, rp, rc, np.sqrt(sq_dist_cp))
    
            
    
    

    

# 3) Trajectory
# -------------

##class Trajectory:
##        def __init__(self):
##                self.points = []
##
##        def AddPoint(self, t, x, y, z):
##                        self.points.append((t, x, y, z))
##
##        def Print(self, output):
##                for (t, x, y, z) in self.points:
##                        output.write("%f\t%f\t%f\t%f\n" % (t, x, y, z))
##                output.flush()

class Gal:
    def __init__(self,name):
        self.name = name

# 4) Planet
# ---------

class Planet:
    def __init__(self, name, code, apo, per, INC, LPE, LAN, MNA, mu, SOI_rad, clos_rad, alt_lim, SOI):
        self.name = name
        if code!=0:
            self.apo = apo
            self.per = per
            self.INC = INC*2*np.pi/360
            self.LPE = LPE*2*np.pi/360
            self.LAN = LAN*2*np.pi/360
            self.MNA = MNA
            self.mu = mu
            self.SOI_rad = SOI_rad
            self.alt_lim = alt_lim
            self.vel_carac = np.sqrt(mu/alt_lim)
            self.SOI = SOI
            self.enfants = []
            self.ECC = (apo - per)/(apo + per)
            self.SMA = apo/(1+self.ECC)
            self.code = code
            self.Mat_Rot = np.transpose(np.array([[cos(self.LPE)*cos(self.LAN)-sin(self.LPE)*cos(self.INC)*sin(self.LAN),
                                               -sin(self.LPE)*cos(self.LAN)-cos(self.LPE)*cos(self.INC)*sin(self.LAN),
                                                sin(self.INC)*sin(self.LAN)],
                   [cos(self.LPE)*sin(self.LAN)+sin(self.LPE)*cos(self.INC)*cos(self.LAN),
                    cos(self.LPE)*cos(self.LAN)*cos(self.INC)-sin(self.LPE)*sin(self.LAN),
                    -sin(self.INC)*cos(self.LAN)],
                   [sin(self.LPE)*sin(self.INC), cos(self.LPE)*sin(self.INC), cos(self.INC)]]))
            self.T = 2*np.pi*np.sqrt(self.SMA**3/self.SOI.mu)
            self.clos_rad = clos_rad
        else:
            self.mu = mu
            self.SOI_rad = SOI_rad
            self.alt_lim = alt_lim
            self.vel_carac = np.sqrt(mu/alt_lim)/3
            self.SOI = SOI
            self.enfants = []
            self.code = code
            self.Xp = np.array([0,0,0])
            self.Vp = np.array([0,0,0])
            self.clos_rad = clos_rad
            
    def GetAbsPos(self,UT):
        global n_it
        if self.code != 0:
            [X_abs, V_abs] = self.GetCartFromOrb(UT)
            plan = self.SOI
            while plan.code != 0:
                [X_plan, V_plan] = plan.GetCartFromOrb(UT)
                X_abs += X_plan
                V_abs += V_plan
                plan = plan.SOI
        else:
            [X_abs, V_abs] = [0,0]
        return [X_abs, V_abs]
    
    def GetCartFromOrb(self, UT):
        SMA = self.SMA
        ECC = self.ECC
        MNA = self.MNA
        mu = self.SOI.mu
        M = (MNA + np.sqrt(mu/SMA**3)*UT)%(2*np.pi)
        E = getE(ECC, M)
        nu = (2*np.arctan2(np.sqrt(1+ECC)*sin(E/2),np.sqrt(1-ECC)*cos(E/2)))%(2*np.pi)
        r = SMA*(1-ECC*cos(E))
        O = np.array([r*cos(nu), r*sin(nu), 0])
        dO = np.array([-sin(E)*np.sqrt(mu*SMA)/r, np.sqrt(1-ECC**2)*cos(E)*np.sqrt(mu*SMA)/r, 0])
        Xp = O@self.Mat_Rot
        Vp = dO@self.Mat_Rot
        self.Xp = Xp
        self.Vp = Vp
        return [Xp, Vp]
        
    def UpdateEnfants(self, planet):
        self.enfants.append(planet)
    



# 5) Solar system
# -----------

class SolarSystem:
    def __init__(self):
        self.list_plan = {}
        self.plan_names = {}

    def AddPlanet(self, planet, planetcode):
        self.list_plan[planetcode] = planet
        self.plan_names[planet.name] = planet
        planet.SOI.UpdateEnfants(planet)

    def AddSun(self, sun, suncode):
        self.list_plan[suncode] = sun
        self.plan_names[sun.name] = sun
        
        

def GeneticAlgo(startpos, startt, target, endt, numb_burns, gene, pop, F, CR, to_opti_DV, obj, sca_fac_SOI):
    #sat(name, SMA, ECC, INC, LAN, LPE, EPH, MNA, SOI, solar_syst)
    choice = range(pop)
    var_n = numb_burns*4
    cmpt_opti = 0

    satstart = Sat(*startpos)
    cans = []
    t_simu = endt-startt
    inter_t_burn = t_simu/(numb_burns+1)
    if sca_fac_SOI==0:
        sca_SOI = 0
    else:
        sca_SOI = 1
    for i in range(pop):
        burns = []
        for j in range(numb_burns):
            #burn = [[startt+rd()*inter_t_burn*2+j*inter_t_burn, np.array([0, rd()*2000-1000, rd()*2000-500-1000*j/numb_burns])]]
            if j==0:
                burn = [[8*6*3600+rd()*6*3600, np.array([0, rd()*2000-1000, rd()*2000-500-1000*j/numb_burns])]]
            else:
                burn = [[startt+rd()*inter_t_burn*2+j*inter_t_burn, np.array([0, rd()*2000-1000, rd()*2000-500-1000*j/numb_burns])]]
            burns += burn
        (resu, end_pos, end_orb, _, _, mindist_w_targ, mindist_w_targ_par, DV_tot, tempo, int_s) = satstart.MoveToEnd(startt, endt, burns, target, False, sca_SOI)
        fitness = obj(target, resu, end_pos, end_orb, mindist_w_targ, mindist_w_targ_par, DV_tot, tempo, int_s, False)
        cans += [[fitness,burns]]
        if fitness<2:
            cmpt_opti += 1
            if cmpt_opti>=to_opti_DV:
                opti_DV = True
        satstart.reset(*startpos[1:-1])
    
    opti_DV = False
    for g in range(gene):
        if g<gene*sca_fac_SOI:
            sca_SOI = 1-g/(gene*sca_fac_SOI)
        else:
            sca_SOI = 0
            if g%50==0:
                converg = True
                vcan1 = cans[0][0]
                for can in cans:
                    if abs(can[0]-vcan1)>0.000001:
                        converg = False
                        break
                if converg:
                    break
        if g%50==0:
            print(g)
        for p in range(pop):
            [x, a, b, c] = random.sample(choice,4)
            elemx = cans[x][1]
            elema = cans[a][1]
            elemb = cans[b][1]
            elemc = cans[c][1]
            cand = deepcopy(elemx)
            
            R = random.randrange(var_n)
            for v in range(var_n):
                if v==R or rd()<CR:
                    if v%4 ==0:
                        v4 = int(v/4)
                        cand[v4][0] = elema[v4][0] + 1*(elemb[v4][0]-elemc[v4][0])
                    else:
                        fv4 = int(np.floor(v/4))
                        v4 = v%4
                        cand[fv4][1][v4-1] = elema[fv4][1][v4-1] + F*(elemb[fv4][1][v4-1]-elemc[fv4][1][v4-1])
            (resu, end_pos, end_orb, _, _, mindist_w_targ, mindist_w_targ_par, DV_tot, tempo, int_s) = satstart.MoveToEnd(startt, endt, cand, target, False, sca_SOI)
            fitness = obj(target, resu, end_pos, end_orb, mindist_w_targ, mindist_w_targ_par, DV_tot, tempo, int_s, opti_DV)
            satstart.reset(*startpos[1:-1])
            if fitness<cans[x][0]:
                if not opti_DV and fitness<2 and cans[x][0]>2:
                    cmpt_opti += 1
                    if cmpt_opti>=to_opti_DV:
                        opti_DV = True
                cans[x] = [fitness, cand]
    return cans
            
        
def Visualize(pos_vel_dict, orb_dict, show_points):  
    theta = np.linspace(0, 2*np.pi, 100)
    plot_dict = {}
    ax_dict = {}
    for plan in pos_vel_dict:
        plot_dict[plan] = plt.figure()
        ax_dict[plan] = plot_dict[plan].gca(projection='3d')
        ax_dict[plan].text2D(0.05, 0.95, plan, transform=ax_dict[plan].transAxes)
        if plan == 'Kerbol':
            ax_dict[plan].set_xlim(-1e11,1e11)
            ax_dict[plan].set_ylim(-1e11,1e11)
            ax_dict[plan].set_zlim(-1e11,1e11)
            vel_scale = 2e5
        else:
            axis_limit = plan_name_dict[plan].SOI_rad
            ax_dict[plan].set_xlim(-axis_limit,axis_limit)
            ax_dict[plan].set_ylim(-axis_limit,axis_limit)
            ax_dict[plan].set_zlim(-axis_limit,axis_limit)
            vel_mi = np.sqrt(2*plan_name_dict[plan].SOI_rad/axis_limit)
            vel_scale = axis_limit/vel_mi/10000
        if show_points:
            drawPosVel(pos_vel_dict[plan], ax_dict[plan], vel_scale)
        orb_plan = orb_dict[plan]
        for orb in orb_plan:
            drawEllipse(orb[0], orb[1], orb[2], orb[3], orb[4], theta, ax_dict[plan])
        for chi in plan_name_dict[plan].enfants:
            drawEllipse(chi.SMA, chi.ECC, chi.INC, chi.LAN, chi.LPE, theta, ax_dict[plan])
    
    plt.show()

def drawEllipse(SMA, ECC, INC, LAN, LPE, angles, axplot):
    if ECC < 1:
        pos_bod_orb = np.array([SMA*cos(angles) - SMA*ECC, SMA*np.sqrt(1-ECC**2)*sin(angles), 0*angles])
    else:
        new_angles1 = np.array([a for a in angles if 1+ECC*cos(a)>0 and a<np.pi])
        new_angles2 = np.array([a for a in angles if 1+ECC*cos(a)>0 and a>np.pi])
        new_angles = np.append(new_angles2, new_angles1)
        p = SMA*(1-ECC**2)
        pos_bod_orb = np.array([p*cos(new_angles)/(1+ECC*cos(new_angles)), p*sin(new_angles)/(1+ECC*cos(new_angles)), 0*new_angles])
        
    Mat_Rot = np.transpose(np.array([[cos(LPE)*cos(LAN)-sin(LPE)*cos(INC)*sin(LAN), -sin(LPE)*cos(LAN)-cos(LPE)*cos(INC)*sin(LAN), sin(INC)*sin(LAN)],
               [cos(LPE)*sin(LAN)+sin(LPE)*cos(INC)*cos(LAN), cos(LPE)*cos(LAN)*cos(INC)-sin(LPE)*sin(LAN), -sin(INC)*cos(LAN)],
               [sin(LPE)*sin(INC), cos(LPE)*sin(INC), cos(INC)]]))
    
    pos_bod_car = np.transpose(Mat_Rot)@pos_bod_orb
    axplot.plot(pos_bod_car[0], pos_bod_car[1], pos_bod_car[2])
    return pos_bod_orb
    
def drawPosVel(pos_list, axplot, vel_scale):
    pos_x = [0]
    pos_y = [0]
    pos_z = [0]
    for el_pos_vel in pos_list:
        el_pos = el_pos_vel[1][0]
        el_vel = el_pos_vel[1][1]
        pos_x += [el_pos[0]]
        pos_y += [el_pos[1]]
        pos_z += [el_pos[2]]
        axplot.plot([el_pos[0], el_pos[0]+el_vel[0]*vel_scale], [el_pos[1],
                     el_pos[1]+el_vel[1]*vel_scale], [el_pos[2], el_pos[2]+el_vel[2]*vel_scale])
    axplot.scatter(pos_x, pos_y, pos_z)
    
def getCartFromf(body, f):
    r = body.SMA*(1-body.ECC**2)/(1+body.ECC*cos(f))
    mu = body.SOI.mu
    if body.ECC<1:
        E = (2*np.arctan2(np.sqrt(1-body.ECC)*sin(f/2),np.sqrt(1+body.ECC)*cos(f/2)))%(2*np.pi)
        O = np.array([r*cos(f), r*sin(f), 0])
        dO = np.array([-sin(E)*np.sqrt(mu*body.SMA)/r, np.sqrt(1-body.ECC**2)*cos(E)*np.sqrt(mu*body.SMA)/r, 0])
    else:
        O = np.array([r*cos(f), r*sin(f), 0])
        vel = np.sqrt(mu*(2/r-1/body.SMA))
        phi = np.arctan2(body.ECC*sin(f),1+body.ECC*cos(f))%(2*np.pi)
        dO = np.array([-vel*sin(f-phi), vel*cos(f-phi), 0])
    X = O@body.Mat_Rot
    Vc = dO@body.Mat_Rot
    return [X, Vc]
    

solarsyst = SolarSystem()
#planet(name, code, apo, per, INC, LPE, LAN, MNA, mu, SOI_rad, clos_rad, alt_lim, SOI)
#sat(name, SMA, ECC, INC, LAN, LPE, EPH, MNA, SOI, solar_syst)
avoidcases = Gal('avoidcases')
avoidcases2 = Gal('avoidcases2')
avoidcases.SOI = avoidcases2

Kerbol = Planet('Kerbol', 0, 0, 0, 0, 0, 0, 0, 1.1723328e18, 4e11, 0, 2.63e8, avoidcases)

Moho = Planet('Moho', 4, 6315765980, 4210510628, 7, 15, 70, 3.14, 1.6860938e11, 9646663, 6e8, 257000, Kerbol)

Eve = Planet('Eve', 5, 9931011387, 9734357701, 2.1, 0, 15, 3.14, 8.1717302e12, 85109365, 6e8, 790000, Kerbol)
Gilly = Planet('Gilly', 13, 48825000, 14175000, 12, 10, 80, 0.9, 8289449.8, 126123.27, 1.4e6, 20000, Eve)

Kerbin = Planet('Kerbin', 1, 13599840256, 13599840256, 0, 0, 0, 3.14, 3.5315984e12, 84159286, 6e8, 670000, Kerbol)
Mun = Planet('Mun', 2, 12000000, 12000000, 0, 0, 0, 1.7, 6.5138398e10, 2429559.6, 3e6, 207100, Kerbin)
Minmus = Planet('Minmus', 3, 47000000, 47000000, 6, 38, 78, 0.9, 1.7658000e9, 2247428.8, 5e6, 66000, Kerbin)

Duna = Planet('Duna', 6, 21783189163, 19669121365, 0.06, 0, 135.5, 3.14, 3.0136321e11, 47921949, 1.2e9, 370000, Kerbol)
Ike = Planet('Ike', 7, 3296000, 3104000, 0.2, 0, 0, 1.7, 1.8568369e10, 1049598.9, 1.1e6, 143000, Duna)

Dres = Planet('Dres', 15, 46761053692, 34917642714, 5, 90, 280, 3.14, 2.1484489e10, 32832840, 1e9, 144000, Kerbol)

Jool = Planet('Jool', 8, 72212238387, 65334882253, 1.304, 0, 52, 0.1, 2.8252800e14, 2.4559852e9, 5e9, 6200000, Kerbol)
Laythe = Planet('Laythe', 9, 27184000, 27184000, 0, 0, 0, 3.14, 1.9620000e12, 3723645.8, 6e6, 550000, Jool)
Val = Planet('Val', 10, 43152000, 43152000, 0, 0, 0, 0.9, 2.0748150e11, 2406401.4, 6e6, 308000, Jool)
Tylo = Planet('Tylo', 12, 68500000, 68500000, 0.25, 0, 0, 3.14, 2.8252800e12, 10856518, 1.1e7, 612000, Jool)
Bop = Planet('Bop', 11, 158697500, 98302500, 15, 25, 10, 0.9, 2.4868349e9, 1221060.9, 1.5e7, 87000, Jool)
Pol = Planet('Pol', 14, 210624207, 149155794, 4.25, 15, 2, 0.9, 7.2170208e8, 1042138.9, 1.5e7, 45000, Jool)

Eeloo = Planet('Eeloo', 16, 113549713200, 66687926800, 6.15, 260, 50, 3.14, 7.4410815e10, 1.1908294e8, 1e9, 214000, Kerbol)

year = 8640000
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

#GeneticAlgo(startpos, startt, target, endt, numb_burns, gene, pop, F, CR)

#target (cart) = [SOI, 'Cart', [np.array([x,y,z]),np.array([vx,vy,vz])] ]
#target (orb) = [SOI, 'Orb', [SMA, ECC, INC, LAN, LPE] ]

#sat(name, SMA, ECC, INC, LAN, LPE, EPH, MNA, SOI, solar_syst)
#print(self.SMA, self.ECC, self.INC, self.LAN, self.LPE)

limit_iterations = True
n_it = 15

startpos = ['testsat', 7e5, 0, 0, 0, 0, 0, 0, Kerbin, solarsyst]
startt = 8*year
target = [Moho, 'PosCirc', 7e5]
endt = 10*year
numb_burns = 2
gene = 4000
pop = 20
F = 0.5
CR = 0.3
to_opti_DV = 5
obj_f = obj_func
sca_fac_SOI = 0

#pos_vel_list = []
#orb_dict = {'Kerbol':[[25550367075.4, 0.412961220852, 0.0837003047603, 0.15963125154, 3.12084323962]]}

testsat = Sat(*startpos)

#testsat.TimeToEnterSOI(0,endt,Kerbin)
#pos_vel_dict = {'Kerbol':pos_vel_list}

pop_comp = GeneticAlgo(startpos, startt, target, endt, numb_burns, gene, pop, F, CR, to_opti_DV, obj_f, sca_fac_SOI)
pop_comp.sort()
result = pop_comp[0]
print(result)
(a,b,c,d,e,f,g,h,i,j) = testsat.MoveToEnd(startt, endt, pop_comp[0][1], target, True, 0)
Visualize(d,e,True)
#print(testsat.ECC, testsat.INC, testsat.SMA)

#[0.17037881650070055, [[33951905.88778491, np.array([ 1873.32093673,    87.43755529,  2000.4114092 ])]]]
#[-343066.43482102873, 3.817899291954018, 10.913752111375448, 35.077033680880426, 16.939459142798807, 33951905.88778491, 1.0268822481365039, 'Kerbin']
#burnstestsat = result[1]

#testsat.reset(*startpos[1:-1])

#sca_SOI = 1-0/(gene*sca_fac_SOI)

pos_vel_dict = {}
orb_dict = {}
for popo in pop_comp:
    testsat.reset(*startpos[1:-1])
    burnstestsat = popo[1]
    (resu, end_pos, end_orb, pos_vel_dict_p, orb_dict_p, mindist_w_targ, mindist_w_targ_par, DV_tot, tempo, int_s) = testsat.MoveToEnd(startt, endt, burnstestsat, target, True, 0)
    for plan in pos_vel_dict_p:
        if plan in pos_vel_dict:
            pos_vel_dict[plan] += pos_vel_dict_p[plan]
        else:
            pos_vel_dict[plan] = pos_vel_dict_p[plan]
        if plan in orb_dict:
            orb_dict[plan] += orb_dict_p[plan]
        else:
            orb_dict[plan] = orb_dict_p[plan]
    
testsat.reset(*startpos[1:-1])

#Visualize(pos_vel_dict, orb_dict, False)


#target (cart) = [SOI, 'Cart', [np.array([x,y,z]),np.array([vx,vy,vz])] ]
#target (orb) = [SOI, 'Orb', [SMA, ECC, INC, LAN, LPE] ]


#pop_comp.sort()
#pop_comp[0]
#(a,b,c,d,e,f,g,h,i,j) = testsat.MoveToEnd(startt, endt, pop_comp[0][1], target, True, 0)
#Visualize(d,e,True)