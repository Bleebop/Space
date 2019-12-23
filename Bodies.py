import numpy as np
from numpy import cos as cos
from numpy import sin as sin
from numpy import pi as PI

import KepEq
import LambSolve
import config
import OldMethods


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
            
    #Reset the craft state with explicitly given orbital elements
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

class Planet:
    def __init__(self, name, code, apo, per, INC, LPE, LAN, MNA, mu, SOI_rad, alt_lim, SOI):
        self.name = name
        self.solarSyst = []
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
        planet.solarSyst = self

    def AddSun(self, sun, suncode):
        self.list_plan[suncode] = sun
        self.plan_names[sun.name] = sun
        sun.solarSyst = self
    
    def retSun(self):
        return self.list_plan[0]

class Gal:
    def __init__(self,name):
        self.name = name