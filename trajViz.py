import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from numpy import cos as cos
from numpy import sin as sin
from numpy import pi as PI

import KepEq
import config

#Visualization of the trajectory
def Visualize(pos_vel_dict, orb_dict, plan_name_dict, show_points):  
    theta = np.linspace(0, 2*PI, 100)
    plot_dict = {}
    ax_dict = {}
    for plan in pos_vel_dict:
        plot_dict[plan] = plt.figure()
        ax_dict[plan] = plot_dict[plan].gca(projection='3d')
        ax_dict[plan].text2D(0.05, 0.95, plan, transform=ax_dict[plan].transAxes)
        if plan == 'Kerbol':										#TODO add a parameter
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
            drawTrajPart(*orb, ax_dict[plan])
        for chi in plan_name_dict[plan].enfants:
            drawTraj(chi.SMA, chi.ECC, chi.INC, chi.LAN, chi.LPE, theta, ax_dict[plan])
    
    plt.show()

def drawTraj(SMA, ECC, INC, LAN, LPE, angles, axplot):
    if ECC < 1:
        pos_bod_orb = np.array([SMA*cos(angles) - SMA*ECC, SMA*np.sqrt(1-ECC**2)*sin(angles), 0*angles])
    else:
        new_angles1 = np.array([a for a in angles if 1+ECC*cos(a)>0 and a<PI])
        new_angles2 = np.array([a for a in angles if 1+ECC*cos(a)>0 and a>PI])
        new_angles = np.append(new_angles2, new_angles1)
        p = SMA*(1-ECC**2)
        pos_bod_orb = np.array([p*cos(new_angles)/(1+ECC*cos(new_angles)), p*sin(new_angles)/(1+ECC*cos(new_angles)), 0*new_angles])
        
    Mat_Rot = np.transpose(np.array([[cos(LPE)*cos(LAN)-sin(LPE)*cos(INC)*sin(LAN), -sin(LPE)*cos(LAN)-cos(LPE)*cos(INC)*sin(LAN), sin(INC)*sin(LAN)],
               [cos(LPE)*sin(LAN)+sin(LPE)*cos(INC)*cos(LAN), cos(LPE)*cos(LAN)*cos(INC)-sin(LPE)*sin(LAN), -sin(INC)*cos(LAN)],
               [sin(LPE)*sin(INC), cos(LPE)*sin(INC), cos(INC)]]))
    
    pos_bod_car = np.transpose(Mat_Rot)@pos_bod_orb
    axplot.plot(pos_bod_car[0], pos_bod_car[1], pos_bod_car[2])
    return pos_bod_orb
    
#For incomplete ellipses and hyperbolas
def drawTrajPart(SMA, ECC, INC, LAN, LPE, M1, M2, axplot):
    if ECC < 1:
        E1 = KepEq.getE(ECC, M1)
        E2 = KepEq.getE(ECC, M2)
        rev = np.floor(E2/(2*PI))
        nu1 = (2*np.arctan2(np.sqrt(1+ECC)*sin(E1/2),np.sqrt(1-ECC)*cos(E1/2)))%(2*PI)
        nu2 = (2*np.arctan2(np.sqrt(1+ECC)*sin(E2/2),np.sqrt(1-ECC)*cos(E2/2)))%(2*PI)
        nu2 += rev*2*PI
        angles = np.linspace(nu1, nu2, 100)
        
        p = SMA*(1-ECC**2)
        pos_bod_orb = np.array([p*cos(angles)/(1+ECC*cos(angles)), p*sin(angles)/(1+ECC*cos(angles)), 0*angles])
    else:
        H1 = KepEq.getH(ECC,M1)
        H2 = KepEq.getH(ECC,M2)
        f1 = (2*np.arctan2(np.sqrt(1+ECC)*np.sinh(H1/2),np.sqrt(ECC-1)*np.cosh(H1/2)))
        f2 = (2*np.arctan2(np.sqrt(1+ECC)*np.sinh(H2/2),np.sqrt(ECC-1)*np.cosh(H2/2)))
        angles = np.linspace(f1, f2, 100)
        
        p = SMA*(1-ECC**2)
        pos_bod_orb = np.array([p*cos(angles)/(1+ECC*cos(angles)), p*sin(angles)/(1+ECC*cos(angles)), 0*angles])
        
    Mat_Rot = np.transpose(np.array([[cos(LPE)*cos(LAN)-sin(LPE)*cos(INC)*sin(LAN), -sin(LPE)*cos(LAN)-cos(LPE)*cos(INC)*sin(LAN), sin(INC)*sin(LAN)],
               [cos(LPE)*sin(LAN)+sin(LPE)*cos(INC)*cos(LAN), cos(LPE)*cos(LAN)*cos(INC)-sin(LPE)*sin(LAN), -sin(INC)*cos(LAN)],
               [sin(LPE)*sin(INC), cos(LPE)*sin(INC), cos(INC)]]))
    
    pos_bod_car = np.transpose(Mat_Rot)@pos_bod_orb
    axplot.plot(pos_bod_car[0], pos_bod_car[1], pos_bod_car[2])
    return pos_bod_orb

#Draw positions and velocities
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