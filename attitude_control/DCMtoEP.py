# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 16:22:15 2018

@author: lele
"""

import numpy as np

from numpy import deg2rad as r
from numpy import rad2deg as d
from numpy import cos as c
from numpy import sin as s

def DCMtoEP(dcm):
    trC = np.trace(dcm)
    sqb0 = (1+trC)/4
    sqb1 = (1+2*dcm[0][0]-trC)/4
    sqb2 = (1+2*dcm[1][1]-trC)/4
    sqb3 = (1+2*dcm[2][2]-trC)/4
    if max(sqb0,sqb1,sqb2,sqb3) == sqb0:
        b0 = np.sqrt(sqb0)
        b1 = (dcm[1][2]-dcm[2][1])/4/b0
        b2 = (dcm[2][0]-dcm[0][2])/4/b0
        b3 = (dcm[0][1]-dcm[1][0])/4/b0
    elif max(sqb0,sqb1,sqb2,sqb3) == sqb1:
        b1 = np.sqrt(sqb1)
        b0 = (dcm[1][2]-dcm[2][1])/4/b1
        b2 = (dcm[0][1]+dcm[1][0])/4/b1
        b3 = (dcm[2][0]+dcm[0][2])/4/b1
    elif max(sqb0,sqb1,sqb2,sqb3) == sqb2:
        b2 = np.sqrt(sqb2)
        b0 = (dcm[2][0]-dcm[0][2])/4/b2
        b1 = (dcm[0][1]+dcm[1][0])/4/b2
        b3 = (dcm[1][2]+dcm[2][1])/4/b2
    else:
        b3 = np.sqrt(sqb3)
        b0 = (dcm[0][1]-dcm[1][0])/4/b3
        b1 = (dcm[2][0]+dcm[0][2])/4/b3
        b2 = (dcm[1][2]+dcm[2][1])/4/b3
    if b0<0:
        b0 = -b0
        b1 = -b1
        b2 = -b2
        b3 = -b3
    return np.array([b0,b1,b2,b3])
    
def EP2DCM(b):
    return np.array([[b[0]**2+b[1]**2-b[2]**2-b[3]**2,2*(b[1]*b[2]+b[0]*b[3]),2*(b[1]*b[3]-b[0]*b[2])],
                     [2*(b[1]*b[2]-b[0]*b[3]),b[0]**2-b[1]**2+b[2]**2-b[3]**2,2*(b[2]*b[3]+b[0]*b[1])],
                     [2*(b[1]*b[3]+b[0]*b[2]),2*(b[2]*b[3]-b[0]*b[1]),b[0]**2-b[1]**2-b[2]**2+b[3]**2]])
    
def Eul2DCM(ang,axis,deg):
    if deg:
        angs = r(ang)
    else:
        angs = ang
    MatRes = np.eye(3)
    for n, th in enumerate(angs):
        if axis[n]==1:
            MatInt = np.array([[1,0,0],[0,c(th),s(th)],[0,-s(th),c(th)]])
        elif axis[n]==2:
            MatInt = np.array([[c(th),0,-s(th)],[0,1,0],[s(th),0,c(th)]])
        else:
            MatInt = np.array([[c(th),s(th),0],[-s(th),c(th),0],[0,0,1]])
        MatRes = np.dot(MatInt,MatRes)
    return MatRes
    
def sumEP(ep1,ep2):
    matEP2 = np.array([[ep2[0],-ep2[1],-ep2[2],-ep2[3]],
                       [ep2[1],ep2[0],ep2[3],-ep2[2]],
                       [ep2[2],-ep2[3],ep2[0],ep2[1]],
                       [ep2[3],ep2[2],-ep2[1],ep2[0]]])
    return np.dot(matEP2,ep1)
    
def diffEP(ept,epint,search_for_ep1):
    if search_for_ep1:
        matEP2 = np.array([[epint[0],-epint[1],-epint[2],-epint[3]],
                           [epint[1],epint[0],epint[3],-epint[2]],
                           [epint[2],-epint[3],epint[0],epint[1]],
                           [epint[3],epint[2],-epint[1],epint[0]]])
        return np.dot(np.transpose(matEP2),ept)
    else:
        matEP1 = np.array([[epint[0],-epint[1],-epint[2],-epint[3]],
                           [epint[1],epint[0],-epint[3],epint[2]],
                           [epint[2],epint[3],epint[0],-epint[1]],
                           [epint[3],-epint[2],epint[1],epint[0]]])
        return np.dot(np.transpose(matEP1),ept)
        
def EPrates(ep,ome):
    matEP = np.array([[-ep[1],-ep[2],-ep[3]],
                      [ep[0],-ep[3],ep[2]],
                      [ep[3],ep[0],-ep[1]],
                      [-ep[2],ep[1],ep[0]]])
    return np.dot(matEP,ome)/2
    
def omeFromEP(ep,eprates):
    matEP = np.array([[ep[0],-ep[1],-ep[2],-ep[3]],
                      [ep[1],ep[0],-ep[3],ep[2]],
                      [ep[2],ep[3],ep[0],-ep[1]],
                      [ep[3],-ep[2],ep[1],ep[0]]])
    return 2*np.dot(np.transpose(matEP),eprates)


def CRPrates(crp,ome):
    matCRP = np.array([[1+crp[0]**2,crp[0]*crp[1]-crp[2],crp[0]*crp[2]+crp[1]],
                       [crp[0]*crp[1]+crp[2],1+crp[1]**2,crp[2]*crp[1]-crp[0]],
                       [crp[0]*crp[2]-crp[1],crp[1]*crp[2]+crp[0],1+crp[2]**2]])
    return np.dot(matCRP,ome)/2
    
def CRP2DCM(crp):
    mat_res = np.array([[1+crp[0]**2-crp[1]**2-crp[2]**2,2*(crp[0]*crp[1]+crp[2]),2*(crp[0]*crp[2]-crp[1])],
                        [2*(crp[0]*crp[1]-crp[2]),1-crp[0]**2+crp[1]**2-crp[2]**2,2*(crp[1]*crp[2]+crp[0])],
                        [2*(crp[0]*crp[2]+crp[1]),2*(crp[1]*crp[2]-crp[0]),1-crp[0]**2-crp[1]**2+crp[2]**2]])
    return mat_res/(1+np.linalg.norm(crp)**2)

def MRPadd(mrp1, mrp2):
    num = (1-np.linalg.norm(mrp1)**2)*mrp2+(1-np.linalg.norm(mrp2)**2)*mrp1-2*np.cross(mrp2,mrp1)
    deno = 1+np.linalg.norm(mrp1)**2*np.linalg.norm(mrp2)**2-2*np.dot(mrp1,mrp2)
    return num/deno
    
def MRPrates(mrp,ome):
    sqNor = np.linalg.norm(mrp)**2
    matMRP = np.array([[1-sqNor+2*mrp[0]**2,2*(mrp[0]*mrp[1]-mrp[2]),2*(mrp[0]*mrp[2]+mrp[1])],
                       [2*(mrp[0]*mrp[1]+mrp[2]),1-sqNor+2*mrp[1]**2,2*(mrp[2]*mrp[1]-mrp[0])],
                       [2*(mrp[0]*mrp[2]-mrp[1]),2*(mrp[2]*mrp[1]+mrp[0]),1-sqNor+2*mrp[2]**2]])
    return np.dot(matMRP,ome)/4
    
def omeFromMRP(mrp,mrpRates):
    sqNor = np.linalg.norm(mrp)**2
    matMRP = np.array([[1-sqNor+2*mrp[0]**2,2*(mrp[0]*mrp[1]-mrp[2]),2*(mrp[0]*mrp[2]-mrp[1])],
                       [2*(mrp[0]*mrp[1]+mrp[2]),1-sqNor+2*mrp[1]**2,2*(mrp[2]*mrp[1]-mrp[0])],
                       [2*(mrp[0]*mrp[2]+mrp[1]),2*(mrp[2]*mrp[1]+mrp[0]),1-sqNor+2*mrp[2]**2]])
    return np.dot(np.tranpose(matMRP),mrpRates)*4/(1+sqNor)**2
    
def DevenK(w,bv,nv):
    B = np.zeros([3,3])
    for n,wk in enumerate(w):
        B = B + wk*np.outer(bv[n],nv[n])
    sig = np.trace(B)
    S = B + np.transpose(B)
    Z = np.array([B[1][2]-B[2][1],B[2][0]-B[0][2],B[0][1]-B[1][0]])
    Kh = np.append([sig],Z)
    Kb = np.append(np.transpose([Z]),S-sig*np.eye(3,3),1)
    K = np.append([Kh], Kb, 0)
    return K

def Matrixify(vec):
    return np.array([[0,-vec[2],vec[1]],[vec[2],0,-vec[0]],[-vec[1],vec[0],0]])
    
def OLAE(w,bv,nv):
    svec = bv+nv
    dvec = bv-nv
    sm = Matrixify(svec[0])
    dm = np.transpose([dvec[0]])
    Wm = np.zeros([3*len(w),3*len(w)])
    Wm[0,0] = w[0]
    Wm[1,1] = w[0]
    Wm[2,2] = w[0]
    for n,_ in enumerate(svec[1:]):
        dm = np.append(dm,np.transpose([dvec[n+1]]),0)
        sm = np.append(sm,Matrixify(svec[n+1]),0)
        Wm[3*n+3,3*n+3] = w[n+1]
        Wm[3*n+4,3*n+4] = w[n+1]
        Wm[3*n+5,3*n+5] = w[n+1]
    swsInv = np.linalg.inv(np.dot(np.transpose(sm),np.dot(Wm,sm)))
    return np.transpose(np.dot(swsInv,np.dot(np.transpose(sm),np.dot(Wm,dm))))[0]