#From https://arxiv.org/abs/1403.2705

import numpy as np
import math
import config

def lamb_solv(r1, r2, t, mu):
    c = r1 - r2
    c_n = np.linalg.norm(c)
    r1_n = np.linalg.norm(r1)
    r2_n = np.linalg.norm(r2)
    s = (c_n + r1_n + r2_n)/2
    Ir1 = r1/r1_n
    Ir2 = r2/r2_n
    Ih_nn = np.cross(Ir1,Ir2)
    Ih = Ih_nn/np.linalg.norm(Ih_nn)
    lamb = np.sqrt(1 - c_n/s)
    if Ih[2]<0:
        lamb = - lamb
        It1 = np.cross(Ir1,Ih)
        It2 = np.cross(Ir2,Ih)
    else:
        It1 = np.cross(Ih,Ir1)
        It2 = np.cross(Ih,Ir2)
    T = np.sqrt(2*mu/s/s/s)*t
    x, y = findxy_lamb(lamb, T)
    gam = np.sqrt(mu*s/2)
    ro = (r1_n-r2_n)/c_n
    sig = np.sqrt(1-ro*ro)
    
    Vr1 = gam*((lamb*y-x)-ro*(lamb*y+x))/r1_n
    Vr2 = -gam*((lamb*y-x)+ro*(lamb*y+x))/r2_n
    Vt1 = gam*sig*(y + lamb*x)/r1_n
    Vt2 = gam*sig*(y + lamb*x)/r2_n
    v1 = Vr1*Ir1 + Vt1*It1
    v2 = Vr2*Ir2 + Vt2*It2
    return v1, v2
		
		
def findxy_lamb(lamb, T):
    sq_lamb = lamb*lamb
    T0 = np.arccos(lamb)+lamb*np.sqrt(1-sq_lamb)
    T1 = 2*(1-lamb**3)/3
    if T>=T0:
        x = (T0/T)**(2/3)-1
    elif T<T1:
        x = 5*T1*(T1-T)/(2*T*(1-lamb**5))+1
    else:
        x = (T0/T)**math.log2(T1/T0)-1
    prevx = -10
    it = 0
    while abs(x-prevx)>1e-5:
        it +=1
        prevx = x
        y = np.sqrt(1-sq_lamb*(1-x*x))
        if abs(x-1)>0.0001:
            normal = True
            
            coPsi = x*y+lamb*(1-x*x)
            if x>1:
                psi = np.arccosh(coPsi)
            else:
#                if abs(coPsi)>1:
#                    print(coPsi, x, y, lamb)
                psi = np.arccos(coPsi)
            Tx = (psi/np.sqrt(abs(1-x*x))-x+lamb*y)/(1-x*x)
        else:
            normal = False
            
            eta = y-lamb*x
            S1 = (1-lamb-x*eta)/2
            Q = 4*F_hyper(S1)/3
            Tx = (Q*eta**3+4*lamb*eta)/2
            
        if x!=1 and (x!=0 or sq_lamb!=1):
            dTx = (3*Tx*x - 2 + 2*lamb**3*x/y)/(1-x*x)
            ddTx = (3*Tx + 5*x*dTx + 2*(1-sq_lamb)*lamb**3/y**3)/(1-x*x)
            dddTx = (7*x*ddTx + 8*dTx - 6*(1-sq_lamb)*lamb**5*x/y**5)/(1-x*x)
            
            x = x - (Tx-T)*(dTx*dTx-(Tx-T)*ddTx/2)/(dTx*(dTx*dTx-(Tx-T)*ddTx)+dddTx*(Tx-T)*(Tx-T)/6)
        else:
            x += 1e-4
            
        if it > config.n_it:
            print('more than '+str(config.n_it)+' iterations for lambert solver')
            break
    y = np.sqrt(1-sq_lamb*(1-x*x))
    return x, y
		
def F_hyper(z):
    delt = 1
    u = 1
    sig = 1
    n = 0
    while u>1e-8:
        n += 1
        if n%2==0:
            gam = n*(n-3)/(2*n+1)/(2*n+3)
        else:
            gam = (n+2)*(n+5)/(2*n+1)/(2*n+3)
        delt = 1/(1-z*gam*delt)
        u = u*(delt-1)
        sig = sig + u
        
        if n > config.n_it:
            print('more than '+str(config.n_it)+' iterations for hypergeometric eval')
            break
    return sig