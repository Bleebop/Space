import numpy as np
from numpy import cos as cos
from numpy import sin as sin
from numpy import pi as PI

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
    return 

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
            print('more than '+str(config.n_it)+' iterations for time of closest approach')
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
                print('more than '+str(config.n_it)+' iterations for SOI entrance')
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
            
#Returns the final position of a craft from a list of burn times and delta-V vectors
#Not used for MGA-1DSM problem
def MoveToEnd(body_c, startt, endt, burns, target, trace):
    pos_vel_dict = {}
    orb_dict = {}
    if trace:
        pos_vel_dict[body_c.SOI.name] = [[startt,body_c.GetCartFromOrb(startt)]]
        orb_dict[body_c.SOI.name] = [[body_c.SMA, body_c.ECC, body_c.INC, body_c.LAN, body_c.LPE,0,2*PI]]
    
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
        ttoleave = TimeToLeaveSOI(body_c, currt) + currt
        if ttoleave<next_op[0]:
            next_op = [ttoleave,'leaveSOI','NA']
        ttocrash = TimeToCrash(body_c, currt) + currt
        if ttocrash<next_op[0]:
            next_op = [ttocrash,'crash','NA']
        [ttoenter, nextSOI, approach] = TimeToEnterSOI(body_c, currt, next_op[0], target[0])
        if approach[2] != 'NA':     #Attempt to reward approach of the target/the parent body of the target
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
            if body_c.SOI.name in pos_vel_dict:
                pos_vel_dict[body_c.SOI.name] += [[currt,body_c.GetCartFromOrb(currt)]]
                pos_vel_dict[body_c.SOI.name] += [[next_op[0],body_c.GetCartFromOrb(next_op[0])]]
            else:
                pos_vel_dict[body_c.SOI.name] = [[currt,body_c.GetCartFromOrb(currt)]]
                pos_vel_dict[body_c.SOI.name] += [[next_op[0],body_c.GetCartFromOrb(next_op[0])]]
            if body_c.SOI.name in orb_dict:
                orb_dict[body_c.SOI.name] += [[body_c.SMA, body_c.ECC, body_c.INC, body_c.LAN, body_c.LPE,0,2*PI]]
            else:
                orb_dict[body_c.SOI.name] = [[body_c.SMA, body_c.ECC, body_c.INC, body_c.LAN, body_c.LPE,0,2*PI]]
                
        if next_op[1]=='enterSOI':
            EnterSOI(body_c,next_op[2],next_op[0])
            if next_op[2].SOI.code == 0:
                per = (1-body_c.ECC)*body_c.SMA
                inter_score += [[next_op[2].alt_lim,per]]
            if next_op[2].name == target[0].name:
                end_pos = body_c.GetCartFromOrb(next_op[0])
                end_orb = [body_c.SMA, body_c.ECC, body_c.INC, body_c.LAN, body_c.LPE, body_c.EPH, body_c.MNA, body_c.SOI]
                resu = True
                fin = True
                if trace:
                    if body_c.SOI.name in pos_vel_dict:
                        pos_vel_dict[body_c.SOI.name] += [[next_op[0],body_c.GetCartFromOrb(next_op[0])]]
                    else:
                        pos_vel_dict[body_c.SOI.name] = [[next_op[0],body_c.GetCartFromOrb(next_op[0])]]
                    if body_c.SOI.name in orb_dict:
                        orb_dict[body_c.SOI.name] += [[body_c.SMA, body_c.ECC, body_c.INC, body_c.LAN, body_c.LPE,0,2*PI]]
                    else:
                        orb_dict[body_c.SOI.name] = [[body_c.SMA, body_c.ECC, body_c.INC, body_c.LAN, body_c.LPE,0,2*PI]]
        elif next_op[1]=='leaveSOI':
            LeaveSOI(body_c,next_op[0])
        elif next_op[1]=='burn':
            Burn(body_c,False,next_op[2],next_op[0])
            i_b += 1
        elif next_op[1]=='end':
            end_pos = body_c.GetCartFromOrb(next_op[0])
            end_orb = [body_c.SMA, body_c.ECC, body_c.INC, body_c.LAN, body_c.LPE, body_c.EPH, body_c.MNA, body_c.SOI]
            resu = True
            fin = True
        elif next_op[1]=='crash':
            end_pos = body_c.GetCartFromOrb(next_op[0])
            end_orb = [body_c.SMA, body_c.ECC, body_c.INC, body_c.LAN, body_c.LPE, body_c.EPH, body_c.MNA, body_c.SOI]
            resu = False
            fin = True
        
        currt = next_op[0]
        if it > config.n_it*10:
            print(next_op)
            if next_op[2]!='NA':
                print(next_op[2].name)
            print(body_c.ECC, body_c.INC, body_c.LAN, body_c.LPE, body_c.MNA, body_c.SMA, body_c.SOI.name)
            #print('more than '+str(config.n_it*10)+' iterations for move to end')
            if it > config.n_it*10.2:
                print(burns)
                break
                    
    return (resu, end_pos, end_orb, pos_vel_dict, orb_dict, mindist_w_targ, mindist_w_targ_par, DV_tot, currt, inter_score)
    
#Does a burn from a burn time and a delta-V vector, either in fixed coordinates or in prograde/normal/radial coordinates
#Not used for MGA-1DSM problem
def Burn(body_c, cart, DV, UT):
    body_c.GetCartFromOrb(UT)
    if cart:
        body_c.Vc = body_c.Vc + DV
    else:
        vec_tang = body_c.Vc/np.linalg.norm(body_c.Vc)
        vec_n = np.cross(body_c.X,vec_tang)
        vec_norm = vec_n/np.linalg.norm(vec_n)
        vec_rad = np.cross(vec_tang, vec_norm)
        dVcC = DV[2]*vec_tang + DV[0]*vec_rad + DV[1]*vec_norm
        body_c.Vc = body_c.Vc + dVcC
    body_c.GetOrbFromCart(UT)
    
#Leave the current SOI for the parent SOI
#Not used for MGA-1DSM problem
def LeaveSOI(body_c, UT):
    nextSOI = body_c.SOI.SOI
    [pos_c, vel_c] = body_c.GetCartFromOrb(UT)
    [pos_plan, vel_plan] = body_c.SOI.GetCartFromOrb(UT)
    nextPos = pos_c + pos_plan
    nextVel = vel_c + vel_plan
    body_c.X = nextPos
    body_c.Vc = nextVel
    body_c.SOI = nextSOI
    body_c.GetOrbFromCart(UT)
    
#Leave the current SOI for one of the children SOI
#Not used for MGA-1DSM problem
def EnterSOI(body_c, nextSOI, UT):
    [pos_c, vel_c] = body_c.GetCartFromOrb(UT)
    [pos_plan, vel_plan] = nextSOI.GetCartFromOrb(UT)
    nextPos = pos_c - pos_plan
    nextVel = vel_c - vel_plan
    body_c.X = nextPos
    body_c.Vc = nextVel
    body_c.SOI = nextSOI
    body_c.GetOrbFromCart(UT)
    
#Returns the time before collision with the orbited body (or inf is the periapsis is high enough)
#Not used for MGA-1DSM problem
def TimeToCrash(body_c, UT):
    per = (1-body_c.ECC)*body_c.SMA
    if per > body_c.SOI.alt_lim:
        t = float('inf')
    else:
        r = body_c.SOI.alt_lim
        n = np.sqrt(body_c.SOI.mu/abs(body_c.SMA)**3)
        if body_c.SMA < 0:
            N = body_c.MNA + n*(UT - body_c.EPH)
            if N<0:
                H = -np.arccosh((body_c.SMA-r)/(body_c.ECC*body_c.SMA))
                N1 = body_c.ECC*np.sinh(H) - H
                t = (N1-N)/n
            else:
                t = float('inf')
        else:
            M = (body_c.MNA + n*(UT - body_c.EPH))%(2*PI)
            E2 = 2*PI - np.arccos((body_c.SMA-r)/(body_c.ECC*body_c.SMA))
            M2 = E2 - body_c.ECC*sin(E2)
            t = ((M2-M)%(2*PI))/n
    return t
    
#Returns the time before leaving the SOI of the orbited body (or inf if not applicable)
#Not used for MGA-1DSM problem
def TimeToLeaveSOI(body_c, UT):
    if body_c.SOI.code == 0:
        t = float('inf')
    else:
        r = body_c.SOI.SOI_rad*1.00001                  # /!\ futur problemes ?
        n = np.sqrt(body_c.SOI.mu/abs(body_c.SMA)**3)
        apo = (1+body_c.ECC)*body_c.SMA
        per = (1-body_c.ECC)*body_c.SMA
        if r < per:
            t = 0
        else:
            if apo < 0:
                N = body_c.MNA + n*(UT - body_c.EPH)
                H = np.arccosh((body_c.SMA-r)/(body_c.ECC*body_c.SMA))
                N1 = body_c.ECC*np.sinh(H) - H
                if (abs(N)<N1):
                    t = (N1-N)/n
                else:
                    t = 0
            elif apo > r:
                M = (body_c.MNA + n*(UT - body_c.EPH))%(2*PI)
                E1 = np.arccos((body_c.SMA-r)/(body_c.ECC*body_c.SMA))
                E2 = 2*PI - E1
                M1 = E1 - body_c.ECC*sin(E1)
                M2 = E2 - body_c.ECC*sin(E2)
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
def TimeToEnterSOI(body_c, UT, nextt, targ):
    SOI = body_c.SOI
    SOI_poss = SOI.enfants
    SOI_poss_bis = []
    SOI_poss_ter = []
    n = np.sqrt(SOI.mu/abs(body_c.SMA)**3)
    approach = [0, float('inf'), 'NA']
    app_calc = [False, float('inf'),float('inf')]
    sun = SOI.solarSyst.retSun()
    resu = [float('inf'), sun.SOI]
    
#        print([pl.code for pl in SOI_poss])
    
    if body_c.ECC >= 1:
        apo = float('inf')
        Mc = body_c.MNA + n*(UT - body_c.EPH)
    else:
        apo = (1+body_c.ECC)*body_c.SMA
        Mc = (body_c.MNA + n*(UT - body_c.EPH))%(2*PI)
    tfp_c = Mc/n
    per = (1-body_c.ECC)*body_c.SMA
    Wc = np.array([sin(body_c.LAN)*sin(body_c.INC), cos(body_c.LAN)*sin(body_c.INC), cos(body_c.INC)])
    p_c = body_c.SMA*(1-body_c.ECC**2)
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
            if (1/sinIr)*sin(body_c.INC)*sin(body_c.LAN-plan.LAN)>0:
                Dp = np.arccos((1/sinIr)*(sin(body_c.INC)*cos(plan.INC)*cos(body_c.LAN-plan.LAN)-sin(plan.INC)*cos(body_c.INC)))
            else:
                Dp = -np.arccos((1/sinIr)*(sin(body_c.INC)*cos(plan.INC)*cos(body_c.LAN-plan.LAN)-sin(plan.INC)*cos(body_c.INC)))
            if (1/sinIr)*sin(plan.INC)*sin(body_c.LAN-plan.LAN)>0:
                Dc = np.arccos((1/sinIr)*(sin(body_c.INC)*cos(plan.INC)-sin(plan.INC)*cos(body_c.INC)*cos(body_c.LAN-plan.LAN)))
            else:
                Dc = -np.arccos((1/sinIr)*(sin(body_c.INC)*cos(plan.INC)-sin(plan.INC)*cos(body_c.INC)*cos(body_c.LAN-plan.LAN)))
        else:
            Dp = plan.LPE
            Dc = body_c.LPE
        
        if apo-per < plan.apo+D-(plan.per-D):
            quasicirc=True
            fc_init1 = Dc-body_c.LPE
            fp_init1 = Dp-plan.LPE
        else:
            quasicirc=False
            #mean_r_p = min([max([(plan.per+plan.apo)/2, per-D]), apo-D])
            mean_r_p = (max([per, plan.per-D])+min(apo, plan.apo+D))/2
            fc_init1 = np.arccos((p_c-mean_r_p)/(body_c.ECC*mean_r_p))
            Oc_init1 = np.array([mean_r_p*cos(fc_init1), mean_r_p*sin(fc_init1), 0])
            Xc_init1 = Oc_init1@body_c.Mat_Rot
            fp_init1 = (np.arctan2(Xc_init1[1],Xc_init1[0])-plan.LAN-plan.LPE)%(2*PI)
        
        ayp = plan.ECC*sin(plan.LPE-Dp)
        axp = plan.ECC*cos(plan.LPE-Dp)
        ayc = body_c.ECC*sin(body_c.LPE-Dc)
        axc = body_c.ECC*cos(body_c.LPE-Dc)
        
        (fpsol1, fcsol1, rpsol1, rcsol1, Dsol1) = NewtMinDist(body_c, plan, fc_init1, fp_init1, ayc, axc, ayp, axp, Dp, Dc, cosIr, D)
        
        if quasicirc:
            fc_init2 = fcsol1 + PI
            fp_init2 = fpsol1 + PI
        else:
            fc_init2 = 2*PI - fc_init1
            Oc_init2 = np.array([mean_r_p*cos(fc_init2), mean_r_p*sin(fc_init2), 0])
            Xc_init2 = Oc_init2@body_c.Mat_Rot
            fp_init2 = (np.arctan2(Xc_init2[1],Xc_init2[0])-plan.LAN-plan.LPE)%(2*PI)
        
        (fpsol2, fcsol2, rpsol2, rcsol2, Dsol2) = NewtMinDist(body_c, plan, fc_init2, fp_init2, ayc, axc, ayp, axp, Dp, Dc, cosIr, D)
        
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

        vuln_ang_c = getVulnWin(body_c, ayc, axc, D, sinIr, p_c, Dc, quasicirc, plan.apo, plan.per)
        vuln_ang_p = getVulnWin(plan, ayp, axp, D, sinIr, p_p, Dp, True, apo, per)
        
        if body_c.ECC>=1:
            vuln_t_c = []
            marq = True
            prevt = 0
            for f in vuln_ang_c:
                E = (2*np.arctanh(np.sqrt((body_c.ECC-1)/(1+body_c.ECC))*np.tan(f/2)))
                M = body_c.ECC*np.sinh(E)-E
                t = M*np.sqrt(-body_c.SMA**3/SOI.mu)
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
                E = (2*np.arctan2(np.sqrt(1-body_c.ECC)*sin(f/2),np.sqrt(1+body_c.ECC)*cos(f/2)))%(2*PI)
                if f>=2*PI:
                    E += 2*PI
                M = E - body_c.ECC*sin(E)
                t = M*body_c.T/(2*PI)
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
#                pos_vel_list += [body_c.GetCartFromOrb(windc[0]-tfp_c, SOI)]
        
#            print(tfp_p, tfp_c)
        rap_c = 0
        rap_p = 0
        for vuln_win_t_c in vuln_t_c:
            rap_c += 2*vuln_win_t_c[1]
        for vuln_win_t_p in vuln_t_p_b:
            rap_p += 2*vuln_win_t_p[1]
        
#            print(vuln_ang_c)
#            print(plan.name)
#            print(rap_c/body_c.T, rap_p/P_p)
        if (rap_c/body_c.T)*(rap_p/P_p)>0.5:
            per_check = min(P_p, body_c.T)/5
            dt = UT-per_check
            R = 0
            prevR = 0
            fl_first = True
            while dt<t_lim+2*per_check:
                prevR = R
                [rc,drc] = body_c.GetCartFromOrb(dt)
                [rp,drp] = plan.GetCartFromOrb(dt)
                R = np.dot(drc-drp,rc-rp)
                if fl_first:
                    fl_first=False
                else:
                    if np.sign(R)!=np.sign(prevR):
                        nt = dt-per_check + per_check*abs(prevR)/(abs(R)+abs(prevR))
                        
                        (resumod, resu1, appmod, approach1) = findClosestApp(nt, body_c, plan, t_lim, D, UT, tarplan, parplan, closest_app)
                        if resumod:
                            resu = resu1
                            t_lim = resu[0]
                        if appmod:
                            approach = approach1
                            closest_app = approach[1]
                dt += per_check
                                        
        elif body_c.ECC>=1:
            for v_c in vuln_t_c:
                if v_c[0]-tfp_c-v_c[1]<t_lim:
                    tc_in_p = (tfp_p+v_c[0]-tfp_c)%P_p
#                    print("tc_in_p : "+str(tc_in_p))
                    for v_p in vuln_t_p:
#                        print("v_p : "+str(v_p))
                        if abs(tc_in_p - v_p[0]) < v_c[1]+v_p[1]:
                            nt = UT + (max(tc_in_p-v_c[1], v_p[0]-v_p[1]) + min(tc_in_p+v_c[1], v_p[0]+v_p[1]))/2 - tc_in_p + v_c[0] - tfp_c
#                            print("nt : "+str(nt))
                            (resumod, resu1, appmod, approach1) = findClosestApp(nt, body_c, plan, t_lim, D, UT, tarplan, parplan, closest_app)
                            if resumod:
                                resu = resu1
                                t_lim = resu[0]
                            if appmod:
                                approach = approach1
                                closest_app = approach[1]
                                                        
        else:
            dt = 0
            while UT+dt<t_lim+body_c.T/2:
                for v_c in vuln_t_c:
                    for v_p in vuln_t_p:
                        tc_in_p = (tfp_p+v_c[0]+dt-tfp_c)%P_p
                        if abs(tc_in_p - v_p[0]) < v_c[1]+v_p[1]:
                            nt = UT + (max(tc_in_p-v_c[1], v_p[0]-v_p[1]) + min(tc_in_p+v_c[1], v_p[0]+v_p[1]))/2 - tc_in_p + v_c[0] + dt - tfp_c
                            
                            (resumod, resu1, appmod, approach1) = findClosestApp(nt, body_c, plan, t_lim, D, UT, tarplan, parplan, closest_app)
                            if resumod:
                                resu = resu1
                                t_lim = resu[0]
                            if appmod:
                                approach = approach1
                                closest_app = approach[1]
                dt += body_c.T
            
    if app_calc[0]:  ##################################### attention periodes courtes ! #########
        if resu[1] == 'NA' or not (resu[1].name == targ.name or resu[1].name == targ.SOI.name):
            dt = UT
            tsol = []
            if body_c.ECC>=1:
                for f in app_calc[1:3]:
                    E = (2*np.arctanh(np.sqrt((body_c.ECC-1)/(1+body_c.ECC))*np.tan(f/2)))
                    M = body_c.ECC*np.sinh(E)-E
                    t = M*np.sqrt(-body_c.SMA**3/SOI.mu)
                    tsol += [t]
                for ts in tsol:
                    abs_t = UT+ts-tfp_c
                    if UT<abs_t and abs_t<t_lim:
                        dist = np.linalg.norm(body_c.GetCartFromOrb(abs_t)[0]-app_calc[3].GetCartFromOrb(abs_t)[0])
                        if dist < approach[1]:
                            approach = [3, dist, app_calc[3]]
            else:
                for f in app_calc[1:3]:
                    E = (2*np.arctan2(np.sqrt(1-body_c.ECC)*sin(f/2),np.sqrt(1+body_c.ECC)*cos(f/2)))%(2*PI)
                    if f>=2*PI:
                        E += 2*PI
                    M = E - body_c.ECC*sin(E)
                    t = M*body_c.T/(2*PI)
                    tsol += [t]
                while dt < t_lim + body_c.T/2:
                    for ts in tsol:
                        abs_t = dt+ts-tfp_c
                        if UT<abs_t and abs_t<t_lim:
                            dist = np.linalg.norm(body_c.GetCartFromOrb(abs_t)[0]-app_calc[3].GetCartFromOrb(abs_t)[0])
                            if dist < approach[1]:
                                approach = [3, dist, app_calc[3]]
                    dt += body_c.T
                                            
    return [resu[0], resu[1], approach]
        
#Returns angular windows where an encounter is possible
#Not used for MGA-1DSM problem
def getVulnWin(body, ay, ax, D, sinIr, p, delt, quasicirc, apo, per):
    if body.ECC<1:
        inter1 = getVulnAng(body, ay, ax, D, sinIr, p, delt)
    else:
        inter1 = [0,2*PI]
    if not all(inter1[i] <= inter1[i+1] for i in range(len(inter1)-1)):
        print("liste des angles pas dans l'ordre")
    if not quasicirc:
        inter2 = getVulnRad(body, D, p, apo, per)
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
def getVulnAng(body, ay, ax, D, sinIr, p, delt):
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
def getVulnRad(body, D, p, apo, per):
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
def NewtMinDist(body_c, plan, fc_in,fp_in, ayc, axc, ayp, axp, Dp, Dc, cosIr, SOI_rad):
    e_c = body_c.ECC
    e_p = plan.ECC
    fc = fc_in
    fp = fp_in
    sq_rad = SOI_rad*SOI_rad
    
    Urp = fp + plan.LPE - Dp
    Urc = fc + body_c.LPE - Dc
    rp = plan.SMA*(1-e_p**2)/(1+e_p*cos(fp))
    rc = body_c.SMA*(1-e_c**2)/(1+e_c*cos(fc))
    cosGa = cos(Urc)*cos(Urp)+sin(Urc)*sin(Urp)*cosIr
    A = sin(Urc) + ayc
    B = cos(Urc) + axc
    C = sin(Urp) + ayp
    D = cos(Urp) + axp
    Eq1 = rc*e_c*sin(fc)+rp*(cos(Urp)*A-sin(Urp)*cosIr*B)
    Eq2 = rp*e_p*sin(fp)+rc*(cos(Urc)*C-sin(Urc)*cosIr*D)
    sq_dist_cp = rp*rp + rc*rc - 2*rc*rp*cosGa
    
    #pos_vel_list += [[0,getCartFromf(body_c, fc)]]
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
        Urc = fc + body_c.LPE - Dc
        rp = plan.SMA*(1-e_p**2)/(1+e_p*cos(fp))
        rc = body_c.SMA*(1-e_c**2)/(1+e_c*cos(fc))
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
    #pos_vel_list += [[0,getCartFromf(body_c, fc)]]
    #pos_vel_list += [[0,getCartFromf(plan, fp)]]
    return(fp, fc, rp, rc, np.sqrt(sq_dist_cp))
                