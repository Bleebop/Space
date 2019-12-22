import numpy as np
from numpy import cos as cos
from numpy import sin as sin
from numpy import pi as PI

import KepEq
import LambSolve
import config

#Returns the intersections of 2 lists of intervals
#Not used for MGA-1DSM problem
def inter_intersec(l1,l2):
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

#Returns the closest approach of two bodies
#Not used for MGA-1DSM problem
def findClosestApp(nt, body_c, body_p, t_lim, D, UT, tarplan, parplan, closest_app):
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
        if it > config.n_it:
            break
#    print(it)
#    print("dist : "+str(np.sqrt(sq_dist_cp)))
    
    if sq_dist_cp<D*D and nt>UT:
        nt1 = nt - body_p.T*D/(2*PI*body_p.SMA)
        
        H = 1000
        R = 1
        
        it=0
        while abs(H/(2*R))>1:
            it+=1
            H, R = NewtRaphTimeInter(body_c, body_p, body_c.SOI, nt1, D)
            nt1 = nt1 - H/(2*R)
            if it > config.n_it:
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
    
#Newton-Raphson iteration for the closest approach between two bodies
#Not used for MGA-1DSM problem
def NewtRaphTimeClos(body_c, body_p, SOI, nt):
    [rc,drc] = body_c.GetCartFromOrb(nt)
    nrc = np.linalg.norm(rc)
    ddrc = -SOI.mu*rc/nrc**3
    [rp,drp] = body_p.GetCartFromOrb(nt)
    nrp = np.linalg.norm(rp)
    ddrp = -SOI.mu*rp/nrp**3
    sq_dist_cp = np.dot(rc-rp,rc-rp)
    R = np.dot(drc-drp,rc-rp)
    dR = np.dot(ddrc-ddrp,rc-rp) + np.dot(drc-drp,drc-drp)
    
    return (sq_dist_cp, R, dR)
    
#Newton-Raphson iteration for SOI entrance
#Not used for MGA-1DSM problem
def NewtRaphTimeInter(body_c, body_p, SOI, nt, D):
    [rc,drc] = body_c.GetCartFromOrb(nt)
    [rp,drp] = body_p.GetCartFromOrb(nt)
    
    H = np.dot(rc-rp,rc-rp) - D*D
    R = np.dot(drc-drp,rc-rp)
    
    return (H, R)


class Sat:
    def __init__(self, name, SMA, ECC, INC, LAN, LPE, EPH, MNA, SOI):
        self.name = name
        self.SOI = SOI
        self.SMA = SMA
        self.ECC = ECC
        self.INC = INC*2*PI/360
        self.LAN = LAN*2*PI/360
        self.LPE = LPE*2*PI/360
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
            self.T = 2*PI*np.sqrt(SMA**3/SOI.mu)
        self.GetCartFromOrb(EPH)
    
		#Returns the orbital elements in degrees
    def retOrbDeg(self):
        return [self.SMA, self.ECC, self.INC*180/PI, self.LAN*180/PI, self.LPE*180/PI, self.EPH, self.MNA, self.SOI]
    
		#Returns the orbital elements in radians
    def retOrbRad(self):
        return [self.SMA, self.ECC, self.INC, self.LAN, self.LPE, self.EPH, self.MNA, self.SOI]
            
		#Reset the craft state with explicitely given orbital elements
    def reset(self, SMA, ECC, INC, LAN, LPE, EPH, MNA, SOI):
        self.SOI = SOI
        self.SMA = SMA
        self.ECC = ECC
        self.INC = INC*2*PI/360
        self.LAN = LAN*2*PI/360
        self.LPE = LPE*2*PI/360
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
            self.T = 2*PI*np.sqrt(SMA**3/SOI.mu)
        self.GetCartFromOrb(EPH)
            
		#Update the craft period
    def updT(self):
        if self.ECC >= 1:
            self.T = float("inf")
        else:
            self.T = 2*PI*np.sqrt(self.SMA**3/self.SOI.mu)
    
		#Given a list of planets and a decision vector, runs the simulation and return the delta-V used
    def MoveToEnd2(self, plan_list, dec_vec, trace):
        pos_tr = {}
        orb_tr = {}
        
        par_body = plan_list[0].SOI
        DV_tot = 0
        tempo = dec_vec[0]
        DSM_t = 0
        enc_t = 0
        
        pos_vel_st = plan_list[0].GetCartFromOrb(tempo)
        i_fb = pos_vel_st[1]/np.linalg.norm(pos_vel_st[1])
        k_nn = np.cross(pos_vel_st[0], i_fb)
        k_fb = k_nn/np.linalg.norm(k_nn)
        j_fb = np.cross(k_fb, i_fb)
        thet = 2*PI*dec_vec[2]
        phi = np.arccos(2*dec_vec[3]-1)-PI/2
        first_burn = dec_vec[1]*(cos(thet)*cos(phi)*i_fb+sin(thet)*cos(phi)*j_fb+sin(phi)*k_fb)
        nextX = pos_vel_st[0]
        nextV = pos_vel_st[1] + first_burn
        
        for n_pl, plan in enumerate(plan_list[1:-1]):
            var_ind = n_pl*4+2
            DSM_t = tempo+dec_vec[var_ind+3]*dec_vec[var_ind+2]
            enc_t = tempo+dec_vec[var_ind+3]
            
            self.X = nextX
            self.Vc = nextV
            self.GetOrbFromCart(tempo)
        
            if trace:
                if par_body.name in pos_tr:
                    pos_tr[par_body.name] += [[tempo, [self.X,self.Vc]]]
                else:
                    pos_tr[par_body.name] = [[tempo, [self.X,self.Vc]]]
                mean_mo = np.sqrt(par_body.mu/abs(self.SMA**3))
                if par_body.name in orb_tr:
                    orb_tr[par_body.name] += [self.retOrbRad()[0:5] + [self.MNA, self.MNA+mean_mo*(DSM_t-tempo)]]
                else:
                    orb_tr[par_body.name] = [self.retOrbRad()[0:5] + [self.MNA, self.MNA+mean_mo*(DSM_t-tempo)]]
            
            Xdsm, Vsdsm = self.GetCartFromOrb(DSM_t)
            pos_pl_ne = plan.GetCartFromOrb(enc_t)
            Vfdsm, Vf = LambSolve.lamb_solv(Xdsm, pos_pl_ne[0], enc_t-DSM_t, par_body.mu)
            
            if trace:
                self.X = Xdsm
                self.Vc = Vfdsm
                self.GetOrbFromCart(DSM_t)
                if par_body.name in pos_tr:
                    pos_tr[par_body.name] += [[DSM_t, [Xdsm,Vsdsm]]]
                else:
                    pos_tr[par_body.name] = [[DSM_t, [Xdsm,Vsdsm]]]
                pos_tr[par_body.name] += [[DSM_t, [Xdsm,Vfdsm]]]
                pos_tr[par_body.name] += [[enc_t, [pos_pl_ne[0],Vf]]]
                
                mean_mo = np.sqrt(par_body.mu/abs(self.SMA**3))
                if par_body.name in orb_tr:
                    orb_tr[par_body.name] += [self.retOrbRad()[0:5] + [self.MNA, self.MNA+mean_mo*(enc_t-DSM_t)]]
                else:
                    orb_tr[par_body.name] = [self.retOrbRad()[0:5] + [self.MNA, self.MNA+mean_mo*(enc_t-DSM_t)]]
            
            Vrel = Vf - pos_pl_ne[1]
            ecc = 1 + dec_vec[var_ind+4]*plan.alt_lim/plan.mu/Vrel/Vrel
            delt = 2*np.arcsin(1/ecc)
            bet = dec_vec[var_ind+5]
            i_ne = Vrel/np.linalg.norm(Vrel)
            j_nn = np.cross(i_ne, pos_pl_ne[1])
            j_ne = j_nn/np.linalg.norm(j_nn)
            k_ne = np.cross(i_ne, j_ne)
            Vout_un = cos(delt)*i_ne + cos(bet)*sin(delt)*j_ne + sin(bet)*sin(delt)*k_ne
            Vout_rel = np.linalg.norm(Vrel)*Vout_un
            Vout = Vout_rel + pos_pl_ne[1]
            nextX = pos_pl_ne[0]
            nextV = Vout
                
            DV_tot += np.linalg.norm(Vfdsm-Vsdsm)
            
            tempo += dec_vec[var_ind+3]
            
        var_ind = (len(plan_list)-2)*4+2
        DSM_t = tempo+dec_vec[var_ind+3]*dec_vec[var_ind+2]
        enc_t = tempo+dec_vec[var_ind+3]
        
        self.X = nextX
        self.Vc = nextV
        self.GetOrbFromCart(tempo)
    
        if trace:
            if par_body.name in pos_tr:
                pos_tr[par_body.name] += [[tempo, [self.X,self.Vc]]]
            else:
                pos_tr[par_body.name] = [[tempo, [self.X,self.Vc]]]
            mean_mo = np.sqrt(par_body.mu/abs(self.SMA**3))
            if par_body.name in orb_tr:
                orb_tr[par_body.name] += [self.retOrbRad()[0:5] + [self.MNA, self.MNA+mean_mo*(DSM_t-tempo)]]
            else:
                orb_tr[par_body.name] = [self.retOrbRad()[0:5] + [self.MNA, self.MNA+mean_mo*(DSM_t-tempo)]]
        
        Xdsm, Vsdsm = self.GetCartFromOrb(DSM_t)
        pos_pl_ne = plan_list[-1].GetCartFromOrb(enc_t)
        Vfdsm, Vf = LambSolve.lamb_solv(Xdsm, pos_pl_ne[0], enc_t-DSM_t, par_body.mu)
        
        DV_tot += np.linalg.norm(Vfdsm-Vsdsm)
        
        if trace:
            self.X = Xdsm
            self.Vc = Vfdsm
            self.GetOrbFromCart(DSM_t)
            if par_body.name in pos_tr:
                pos_tr[par_body.name] += [[DSM_t, [Xdsm,Vsdsm]]]
            else:
                pos_tr[par_body.name] = [[DSM_t, [Xdsm,Vsdsm]]]
            pos_tr[par_body.name] += [[DSM_t, [Xdsm,Vfdsm]]]
            pos_tr[par_body.name] += [[enc_t, [pos_pl_ne[0],Vf]]]
            
            mean_mo = np.sqrt(par_body.mu/abs(self.SMA**3))
            if par_body.name in orb_tr:
                orb_tr[par_body.name] += [self.retOrbRad()[0:5] + [self.MNA, self.MNA+mean_mo*(enc_t-DSM_t)]]
            else:
                orb_tr[par_body.name] = [self.retOrbRad()[0:5] + [self.MNA, self.MNA+mean_mo*(enc_t-DSM_t)]]
        
        return DV_tot, [pos_pl_ne[0], Vf-pos_pl_ne[1]], pos_tr, orb_tr
            
		#Returns the final position of a craft from a list of burn times and delta-V vectors
		#Not used for MGA-1DSM problem
    def MoveToEnd(self, startt, endt, burns, target, trace):
        pos_vel_dict = {}
        orb_dict = {}
        if trace:
            pos_vel_dict[self.SOI.name] = [[startt,self.GetCartFromOrb(startt)]]
            orb_dict[self.SOI.name] = [[self.SMA, self.ECC, self.INC, self.LAN, self.LPE]]
        
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
            ttoleave = self.TimeToLeaveSOI(currt) + currt
            if ttoleave<next_op[0]:
                next_op = [ttoleave,'leaveSOI','NA']
            ttocrash = self.TimeToCrash(currt) + currt
            if ttocrash<next_op[0]:
                next_op = [ttocrash,'crash','NA']
            [ttoenter, nextSOI, approach] = self.TimeToEnterSOI(currt, next_op[0], target[0])
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
            if it > config.n_it*10:
                print(next_op)
                if next_op[2]!='NA':
                    print(next_op[2].name)
                print(self.ECC, self.INC, self.LAN, self.LPE, self.MNA, self.SMA, self.SOI.name)
                #print('more than '+str(config.n_it*10)+' iterations for move to end')
                if it > config.n_it*10.2:
                    print(burns)
                    break
                
        return (resu, end_pos, end_orb, pos_vel_dict, orb_dict, mindist_w_targ, mindist_w_targ_par, DV_tot, currt, inter_score)
            
		#Returns the absolute position in heliocentric coordinates of the body
    def GetAbsPos(self,UT):
        [X_abs, V_abs] = self.GetCartFromOrb(UT)
        plan = self.SOI
        while plan.code != 0:
            [X_plan, V_plan] = plan.GetCartFromOrb(UT)
            X_abs += X_plan
            V_abs += V_plan
            plan = plan.SOI
        return [X_abs, V_abs]
    
		#Returns Cartesian coordinates (relative to the parent body) from the orbital elements
    def GetCartFromOrb(self, UT):
        SMA = self.SMA
        ECC = self.ECC
        EPH = self.EPH
        MNA = self.MNA
        mu = self.SOI.mu
        if ECC==1:
            ECC=0.99999999
        if ECC<1:
            M = (MNA + np.sqrt(mu/SMA**3)*(UT-EPH))%(2*PI)
            E = KepEq.getE(ECC, M)
            nu = (2*np.arctan2(np.sqrt(1+ECC)*sin(E/2),np.sqrt(1-ECC)*cos(E/2)))%(2*PI)
            r = SMA*(1-ECC*cos(E))
            O = np.array([r*cos(nu), r*sin(nu), 0])
            dO = np.array([-sin(E)*np.sqrt(mu*SMA)/r, np.sqrt(1-ECC**2)*cos(E)*np.sqrt(mu*SMA)/r, 0])
            X = O@self.Mat_Rot
            Vc = dO@self.Mat_Rot
        else:
            N = MNA + np.sqrt(mu/abs(SMA)**3)*(UT-EPH)
            H = KepEq.getH(ECC,N)
            f = (2*np.arctan2(np.sqrt(1+ECC)*np.sinh(H/2),np.sqrt(ECC-1)*np.cosh(H/2)))%(2*PI)
            r = SMA*(1-ECC*np.cosh(H))
            O = np.array([r*cos(f), r*sin(f), 0])
            vel = np.sqrt(mu*(2/r-1/SMA))
            phi = np.arctan2(ECC*sin(f),1+ECC*cos(f))%(2*PI)
            dO = np.array([-vel*sin(f-phi), vel*cos(f-phi), 0])
            X = O@self.Mat_Rot
            Vc = dO@self.Mat_Rot
        self.X = X
        self.Vc = Vc
        return [X, Vc]
    
		#Returns orbital elements from Cartesian coordinates (relative to the parent body)
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
            arglat = np.arctan2(self.X[1],self.X[0])%(2*PI)
        elif self.INC == 90.0:
            self.LAN = np.arctan2(h[0],-h[1])%(2*PI)
            arglat = np.arctan2(self.X[2], np.sqrt(self.X[0]**2+self.X[1]**2))%(2*PI)
        else:
            self.LAN = np.arctan2(h[0],-h[1])%(2*PI)
            sinbet = self.X[2]/r
            lamb = np.arctan2(self.X[1],self.X[0])%(2*PI)
            if (lamb-self.LAN)%(PI) == PI/2:
                arglat = (lamb-self.LAN)%(2*PI)
            else:
                tan_arglat = np.tan(lamb-self.LAN)/cos(self.INC)
                if sinbet>0 and tan_arglat>0:
                    arglat = np.arcsin(sinbet/sin(self.INC))
                elif sinbet<0 and tan_arglat>0:
                    arglat = PI - np.arcsin(sinbet/sin(self.INC))
                elif sinbet<0 and tan_arglat<0:
                    arglat = 2*PI + np.arcsin(sinbet/sin(self.INC))
                elif sinbet>0 and tan_arglat<0:
                    arglat = PI - np.arcsin(sinbet/sin(self.INC))
                else:
                    if self.Vc[2]>0:
                        arglat = 0
                    else:
                        arglat = PI
        if self.ECC>0.0000001:
            cosnu = (self.SMA*(1-self.ECC**2)-r)/(self.ECC*r)
            if cosnu>1:
                cosnu = 1
            elif cosnu<-1:
                cosnu = -1
            if np.dot(self.X,self.Vc)>0:
                nu = np.arccos(cosnu)
            else:
                nu = 2*PI - np.arccos(cosnu)
            self.LPE = (arglat - nu)%(2*PI)
            if sE<0:
                E = (2*np.arctan2(np.sqrt(1-self.ECC)*sin(nu/2),np.sqrt(1+self.ECC)*cos(nu/2)))%(2*PI)
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
                    self.MNA = 2*PI - np.arccos(cosnu)
            else:
                n_un = n/np.linalg.norm(n)
                r_un = self.X/r
                cosnu = np.dot(n_un,r_un)
                self.LPE = 0
                if self.X[2]>0:
                    self.MNA = np.arccos(cosnu)
                else:
                    self.MNA = 2*PI - np.arccos(cosnu)
        self.EPH = UT
        LAN = self.LAN
        LPE = self.LPE
        INC = self.INC
        self.Mat_Rot = np.transpose(np.array([[cos(LPE)*cos(LAN)-sin(LPE)*cos(INC)*sin(LAN), -sin(LPE)*cos(LAN)-cos(LPE)*cos(INC)*sin(LAN), sin(INC)*sin(LAN)],
                   [cos(LPE)*sin(LAN)+sin(LPE)*cos(INC)*cos(LAN), cos(LPE)*cos(LAN)*cos(INC)-sin(LPE)*sin(LAN), -sin(INC)*cos(LAN)],
                   [sin(LPE)*sin(INC), cos(LPE)*sin(INC), cos(INC)]]))
        self.updT()
    
		#Does a burn from a burn time and a delta-V vector, either in fixed coordinates or in prograde/normal/radial coordinates
		#Not used for MGA-1DSM problem
    def Burn(self, cart, DV, UT):
        self.GetCartFromOrb(UT)
        if cart:
            self.Vc = self.Vc + DV
        else:
            vec_tang = self.Vc/np.linalg.norm(self.Vc)
            vec_n = np.cross(self.X,vec_tang)
            vec_norm = vec_n/np.linalg.norm(vec_n)
            vec_rad = np.cross(vec_tang, vec_norm)
            dVcC = DV[2]*vec_tang + DV[0]*vec_rad + DV[1]*vec_norm
            self.Vc = self.Vc + dVcC
        self.GetOrbFromCart(UT)
    
		#Leave the current SOI for the parent SOI
		#Not used for MGA-1DSM problem
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
    
		#Leave the current SOI for one of the children SOI
		#Not used for MGA-1DSM problem
    def EnterSOI(self, nextSOI, UT):
        [pos_c, vel_c] = self.GetCartFromOrb(UT)
        [pos_plan, vel_plan] = nextSOI.GetCartFromOrb(UT)
        nextPos = pos_c - pos_plan
        nextVel = vel_c - vel_plan
        self.X = nextPos
        self.Vc = nextVel
        self.SOI = nextSOI
        self.GetOrbFromCart(UT)
    
		#Returns the time before collision with the orbited body (or inf is the periapsis is high enough)
		#Not used for MGA-1DSM problem
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
                M = (self.MNA + n*(UT - self.EPH))%(2*PI)
                E2 = 2*PI - np.arccos((self.SMA-r)/(self.ECC*self.SMA))
                M2 = E2 - self.ECC*sin(E2)
                t = ((M2-M)%(2*PI))/n
        return t
    
		#Returns the time before leaving the SOI of the orbited body (or inf if not applicable)
		#Not used for MGA-1DSM problem
    def TimeToLeaveSOI(self, UT):
        if self.SOI.code == 0:
            t = float('inf')
        else:
            r = self.SOI.SOI_rad*1.00001                  # /!\ futur problemes ?
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
                    M = (self.MNA + n*(UT - self.EPH))%(2*PI)
                    E1 = np.arccos((self.SMA-r)/(self.ECC*self.SMA))
                    E2 = 2*PI - E1
                    M1 = E1 - self.ECC*sin(E1)
                    M2 = E2 - self.ECC*sin(E2)
                    if (M>=M1) and (M<=M2):
                        t = 0
                    else:
                        t = ((M1-M)%(2*PI))/n
                else:
                    t = float("inf")
        return t

		#Returns the time before encountering another body, the body and a minimal distance.
		#Based on (Hoots, 1984) http://adsabs.harvard.edu/full/1984CeMec..33..143H
		#Bit of a mess
		#Not used for MGA-1DSM problem
    def TimeToEnterSOI(self, UT, nextt, targ):
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
            Mc = (self.MNA + n*(UT - self.EPH))%(2*PI)
        tfp_c = Mc/n
        per = (1-self.ECC)*self.SMA
        Wc = np.array([sin(self.LAN)*sin(self.INC), cos(self.LAN)*sin(self.INC), cos(self.INC)])
        p_c = self.SMA*(1-self.ECC**2)
        for plan in SOI_poss:
            D = plan.SOI_rad
            if not(min(apo, plan.apo) < (max(per, plan.per)-D)):
                SOI_poss_bis.append(plan)
            elif plan.name == targ.name or plan.name == targ.SOI.name:
                approach = [1,(max(per, plan.per)-D)-min(apo, plan.apo), plan]
        
#        print([pl.code for pl in SOI_poss_bis])
        
        for plan in SOI_poss_bis:
            D = plan.SOI_rad
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
                fp_init1 = (np.arctan2(Xc_init1[1],Xc_init1[0])-plan.LAN-plan.LPE)%(2*PI)
            
            ayp = plan.ECC*sin(plan.LPE-Dp)
            axp = plan.ECC*cos(plan.LPE-Dp)
            ayc = self.ECC*sin(self.LPE-Dc)
            axc = self.ECC*cos(self.LPE-Dc)
            
            (fpsol1, fcsol1, rpsol1, rcsol1, Dsol1) = self.NewtMinDist(plan, fc_init1, fp_init1, ayc, axc, ayp, axp, Dp, Dc, cosIr, D)
            
            if quasicirc:
                fc_init2 = fcsol1 + PI
                fp_init2 = fpsol1 + PI
            else:
                fc_init2 = 2*PI - fc_init1
                Oc_init2 = np.array([mean_r_p*cos(fc_init2), mean_r_p*sin(fc_init2), 0])
                Xc_init2 = Oc_init2@self.Mat_Rot
                fp_init2 = (np.arctan2(Xc_init2[1],Xc_init2[0])-plan.LAN-plan.LPE)%(2*PI)
            
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
            Mp = (plan.MNA + 2*PI*UT/P_p)%(2*PI)
            tfp_p = Mp*P_p/(2*PI)
        
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
                    E = (2*np.arctan2(np.sqrt(1-self.ECC)*sin(f/2),np.sqrt(1+self.ECC)*cos(f/2)))%(2*PI)
                    if f>=2*PI:
                        E += 2*PI
                    M = E - self.ECC*sin(E)
                    t = M*self.T/(2*PI)
                    if marq:
                        prevt = t
                    else:
                        vuln_t_c += [[(t+prevt)/2,abs(t-prevt)/2]]
                    marq = not marq
            
            if vuln_ang_p==[0,2*PI]:
                vuln_t_p_b = [[P_p/2,P_p/2]]
                vuln_t_p = [[P_p/2,P_p/2]]
            else:
                vuln_t_p_b = []
                marq = True
                prevt = 0
                for f in vuln_ang_p:
                    E = (2*np.arctan2(np.sqrt(1-plan.ECC)*sin(f/2),np.sqrt(1+plan.ECC)*cos(f/2)))%(2*PI)
                    if f>=2*PI:
                        E += 2*PI
                    M = E - plan.ECC*sin(E)
                    t = M*P_p/(2*PI)
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
                        E = (2*np.arctan2(np.sqrt(1-self.ECC)*sin(f/2),np.sqrt(1+self.ECC)*cos(f/2)))%(2*PI)
                        if f>=2*PI:
                            E += 2*PI
                        M = E - self.ECC*sin(E)
                        t = M*self.T/(2*PI)
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
                  
		#Returns angular windows where an encounter is possible
		#Not used for MGA-1DSM problem
    def getVulnWin(self, body, ay, ax, D, sinIr, p, delt, quasicirc, apo, per):
        if body.ECC<1:
            inter1 = self.getVulnAng(body, ay, ax, D, sinIr, p, delt)
        else:
            inter1 = [0,2*PI]
        if not all(inter1[i] <= inter1[i+1] for i in range(len(inter1)-1)):
            print("liste des angles pas dans l'ordre")
        if not quasicirc:
            inter2 = self.getVulnRad(body, D, p, apo, per)
        else:
            inter2 = [0,2*PI]
        
        vuln_win_b = inter_intersec(inter1, inter2)
        
        if vuln_win_b[0]==0 and vuln_win_b[-1]==2*PI and len(vuln_win_b)>2:
            vuln_win = vuln_win_b[2:-1] + [vuln_win_b[1]+2*PI]
        else:
            vuln_win = vuln_win_b
            
#        for f in vuln_win:
#            pos_vel_list += [getCartFromf(body, f)]
        
        return vuln_win
        
		#Returns angular windows where one body is close to the other body's orbit plane
		#Not used for MGA-1DSM problem
    def getVulnAng(self, body, ay, ax, D, sinIr, p, delt):
        Q1 = p*sinIr*(p*sinIr - 2*D*ay)-(1-body.ECC**2)*D**2
        Q2 = p*sinIr*(p*sinIr + 2*D*ay)-(1-body.ECC**2)*D**2
        
        if Q1<0:
            if Q2<0:
                return [0,2*PI]
            else:
                cosUrMo1 = (-ax*D**2 + (p*sinIr+D*ay)*np.sqrt(Q2))/(Q2 + D**2)
                cosUrMo2 = (-ax*D**2 - (p*sinIr+D*ay)*np.sqrt(Q2))/(Q2 + D**2)
                
                if abs(cosUrMo1)>1 or abs(cosUrMo2)>1:
                    return [0,2*PI]
                else:
                    if cosUrMo1<cosUrMo2:
                        Ur1 = 2*PI - np.arccos(cosUrMo2)
                        Ur2 = 2*PI - np.arccos(cosUrMo1)
                    else:
                        Ur1 = 2*PI - np.arccos(cosUrMo1)
                        Ur2 = 2*PI - np.arccos(cosUrMo2)
                    
                    f_1 = (Ur1 + delt - body.LPE)%(2*PI)
                    f_2 = (Ur2 + delt - body.LPE)%(2*PI)
                    if f_2<f_1:
                        return [0,f_2,f_1,2*PI]
                    else:
                        return [f_1,f_2]
        else:
            if Q2<0:
                cosUrPl1 = (-ax*D**2 + (p*sinIr-D*ay)*np.sqrt(Q1))/(Q1 + D**2)
                cosUrPl2 = (-ax*D**2 - (p*sinIr-D*ay)*np.sqrt(Q1))/(Q1 + D**2)
                
                if abs(cosUrPl1)>1 or abs(cosUrPl2)>1:
                    return [0,2*PI]
                else:
                    if cosUrPl1<cosUrPl2:
                        Ur1 = np.arccos(cosUrPl1)
                        Ur2 = np.arccos(cosUrPl2)
                    else:
                        Ur1 = np.arccos(cosUrPl2)
                        Ur2 = np.arccos(cosUrPl1)
                        
                    f_1 = (Ur1 + delt - body.LPE)%(2*PI)
                    f_2 = (Ur2 + delt - body.LPE)%(2*PI)
                    if f_2<f_1:
                        return [0,f_2,f_1,2*PI]
                    else:
                        return [f_1,f_2]
            else:
                cosUrPl1 = (-ax*D**2 + (p*sinIr-D*ay)*np.sqrt(Q1))/(Q1 + D**2)
                cosUrPl2 = (-ax*D**2 - (p*sinIr-D*ay)*np.sqrt(Q1))/(Q1 + D**2)
                cosUrMo1 = (-ax*D**2 + (p*sinIr+D*ay)*np.sqrt(Q2))/(Q2 + D**2)
                cosUrMo2 = (-ax*D**2 - (p*sinIr+D*ay)*np.sqrt(Q2))/(Q2 + D**2)
                
                if abs(cosUrPl1)>1 or abs(cosUrPl2)>1 or abs(cosUrMo1)>1 or abs(cosUrMo2)>1:
                    return [0,2*PI]
                else:
                    if cosUrPl1<cosUrPl2:
                        Ur1 = np.arccos(cosUrPl1)
                        Ur4 = np.arccos(cosUrPl2)
                    else:
                        Ur1 = np.arccos(cosUrPl2)
                        Ur4 = np.arccos(cosUrPl1)
                        
                    if cosUrMo1<cosUrMo2:
                        Ur2 = 2*PI - np.arccos(cosUrMo1)
                        Ur3 = 2*PI - np.arccos(cosUrMo2)
                    else:
                        Ur2 = 2*PI - np.arccos(cosUrMo2)
                        Ur3 = 2*PI - np.arccos(cosUrMo1)
                    
#                UrPl1 = np.arccos(cosUrPl1)
#                UrPl2 = np.arccos(cosUrPl2)
#                UrMo1 = 2*PI - np.arccos(cosUrMo1)
#                UrMo2 = 2*PI - np.arccos(cosUrMo2)
                
                f_1 = (Ur1 + delt - body.LPE)%(2*PI)
                f_2 = (Ur2 + delt - body.LPE)%(2*PI)
                f_3 = (Ur3 + delt - body.LPE)%(2*PI)
                f_4 = (Ur4 + delt - body.LPE)%(2*PI)
                
                if f_2<f_1:
                    return [0,f_2,f_3,f_4,f_1,2*PI]
                elif f_4<f_3:
                    return [0,f_4,f_1,f_2,f_3,2*PI]
                elif f_3<f_2:
                    return [f_3,f_4,f_1,f_2]
                else:
                    return [f_1,f_2,f_3,f_4]
                    
            
		#Returns angular windows where the craft orbit radius is close to the potential encounter orbit radius
		#Not used for MGA-1DSM problem
    def getVulnRad(self, body, D, p, apo, per):
        r_int = per-D
        r_ext = apo+D
        cosf_int = (p-r_int)/(body.ECC*r_int)
        cosf_ext = (p-r_ext)/(body.ECC*r_ext)
        if abs(cosf_int)>1:
            if abs(cosf_ext)>1:
                return [0,2*PI]
            else:
                return [0, np.arccos(cosf_ext), 2*PI - np.arccos(cosf_ext), 2*PI]
        else:
            if abs(cosf_ext)>1:
                return [np.arccos(cosf_int), 2*PI - np.arccos(cosf_int)]
            else:
                f_1 = np.arccos(cosf_int)
                f_2 = np.arccos(cosf_ext)
                f_3 = 2*PI - np.arccos(cosf_ext)
                f_4 = 2*PI - np.arccos(cosf_int)
                return [f_1,f_2,f_3,f_4]
    
		#Returns the true anomaly, radius for both orbits during their closest encounter, plus the minimal distance
		#Not used for MGA-1DSM problem
    def NewtMinDist(self, plan, fc_in,fp_in, ayc, axc, ayp, axp, Dp, Dc, cosIr, SOI_rad):
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
            
            if it > config.n_it:
                break
        
        #print(it)
        #pos_vel_list += [[0,getCartFromf(self, fc)]]
        #pos_vel_list += [[0,getCartFromf(plan, fp)]]
        return(fp, fc, rp, rc, np.sqrt(sq_dist_cp))

class Planet:
    def __init__(self, name, code, apo, per, INC, LPE, LAN, MNA, mu, SOI_rad, alt_lim, SOI):
        self.name = name
        if code!=0:
            self.apo = apo
            self.per = per
            self.INC = INC*2*PI/360
            self.LPE = LPE*2*PI/360
            self.LAN = LAN*2*PI/360
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
            self.T = 2*PI*np.sqrt(self.SMA**3/self.SOI.mu)
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
    
    def retOrbDeg(self):
        return [self.SMA, self.ECC, self.INC*180/PI, self.LAN*180/PI, self.LPE*180/PI, 0, self.MNA, self.SOI]
    
    def GetAbsPos(self,UT):
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
        M = (MNA + np.sqrt(mu/SMA**3)*UT)%(2*PI)
        E = KepEq.getE(ECC, M)
        nu = (2*np.arctan2(np.sqrt(1+ECC)*sin(E/2),np.sqrt(1-ECC)*cos(E/2)))%(2*PI)
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

class Gal:
    def __init__(self,name):
        self.name = name