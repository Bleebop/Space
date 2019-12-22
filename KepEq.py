import numpy as np
from numpy import cos as cos
from numpy import sin as sin
from numpy import pi as PI

import config

#Newton-Raphson iteration for getting the eccentric anomaly (elliptic trajectory)
def NewtRaphELL(x,e,M):
    res = x - (x - e*sin(x) - M)/(1-e*cos(x))
    return res

#First guess for Newton-Raphson (elliptic)
def NewtStartELL(e, M):
    t34 = e*e
    t35 = e*t34
    t33 = cos(M)
    return M+(0.5*t35+e+(t34+3*t33*t35/2)*t33)*sin(M)

#Newton-Raphson iteration for getting the hyperbolic anomaly (hyperbolic trajectory)
def NewtRaphHYP(x,e,N):
    res = x - (e*np.sinh(x) - N - x)/(e*np.cosh(x) - 1)
    return res

#First guess for Newton-Raphson (hyperbolic)
def NewtStartHYP(e, N):
    if N/e>3:
        H = np.log(N/e)+0.85
    else:
        H = N/(e-1)
        if H*H>6*(e-1):
            H = np.exp(np.log(6*N)/3)
    return H

#Returns the eccentric anomaly of a body in elliptic trajectory
def getE(ECC, M):
    E = NewtStartELL(ECC,M)
    diff = 100
    it = 0
    while diff > 1e-10:
        it += 1
        E_bis = NewtRaphELL(E, ECC, M)
        diff = abs(E - E_bis)
        E = E_bis
        if it > config.n_it:
            print('more than '+str(config.n_it)+' iterations for E ; ECC = '+str(ECC)+' ; M = '+str(M))
            break
#    print('it Ell : '+str(it))
    return E
    
#Returns the hyperbolic anomaly of a body in hyperbolic trajectory
def getH(ECC, N):
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
        if it > config.n_it:
            print('more than '+str(config.n_it)+' iterations for H')
            break
#    print('it Hyp : '+str(it))
    if is_neg:
        H = -H
    return H