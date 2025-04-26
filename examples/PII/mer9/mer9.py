# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 18:47:27 2023

@author: balaz
"""

import numpy as np
import matplotlib.pyplot as plt
import praktikum as p
import pandas as pd
from math import sqrt

data = np.array(pd.read_excel('mer9.xlsx'))

def pf(P, U, I):
    return P/(U*I)

def phi(PF):
    if isinstance(PF, np.ndarray):
        PF[PF>1] = 1
    else:
        if PF > 1:
            PF = 1
    return np.arccos(PF)

def max_chyb(f, p, std):
    return max(np.abs(f(p+std)-f(p)), np.abs(f(p-std)-f(p)))

#%% 1 -- R
P_R = 2.58
s_PR = sqrt(0.005**2 + (P_R*0.5/100+10*0.001)**2)

U_R = 50.7
s_UR = U_R*0.4/100+5*0.1

I_R = 0.051
s_IR = I_R*0.4/100+5*0.001

pf_R = pf(P_R, U_R, I_R)
s_pfR = p.prenos_chyb(pf, [s_PR, s_UR, s_IR], [P_R, U_R, I_R])

print(p.round_to(pf_R, s_pfR))
#%% 1 -- C
P_C = 0.011
s_PC = sqrt(0.001**2 + (P_C*0.5/100+10*0.001)**2)

U_C = 50.2
s_UC = U_C*0.4/100+5*0.1

I_C = 0.162
s_IC = I_C*0.4/100+5*0.001

pf_C = pf(P_C, U_C, I_C)
s_pfC = p.prenos_chyb(pf, [s_PC, s_UC, s_IC], [P_C, U_C, I_C])

print(p.round_to(pf_C, s_pfC))
#%% 1 -- L
P_L = 0.591
s_PL = sqrt(0.003**2 + (P_L*0.5/100+10*0.001)**2)

U_L = 50.4
s_UL = U_L*0.4/100+5*0.1

I_L = 0.032
s_IL = I_L*0.4/100+5*0.001

pf_L = pf(P_L, U_L, I_L)
s_pfL = p.prenos_chyb(pf, [s_PL, s_UL, s_IL], [P_L, U_L, I_L])

print(p.round_to(pf_L, s_pfL))

#%% 2 
phi_R = phi(pf_R)
s_phiR = max_chyb(phi, pf_R, s_pfR)

phi_L = phi(pf_L)
s_phiL = max_chyb(phi, pf_L, s_pfL)

phi_C = phi(pf_C)
s_phiC = max_chyb(phi, pf_C, s_pfC)

print(p.round_to(phi_R, s_phiR), p.round_to(phi_R/np.pi, s_phiR/np.pi))
print(p.round_to(phi_L, s_phiL), p.round_to(phi_L/np.pi, s_phiL/np.pi))
print(p.round_to(phi_C, s_phiC), p.round_to(phi_C/np.pi, s_phiC/np.pi))

table0 = p.default_table(pd.DataFrame({
    ' ': ['odpor', 'cievka', 'kondenzátor'],
    'P [W]': p.readable([P_R, P_L, P_C], [s_PR, s_PL, s_PC]),
    'U [V]': p.readable([U_R, U_L, U_C], [s_UR, s_UL, s_UC]),
    'I [A]': p.readable([I_R, I_L, I_C], [s_IR, s_IL, s_IC]),
    'PF [-]': p.readable([pf_R, pf_L, pf_C], [s_pfR, s_pfL, s_pfC]),
    '$\\varphi$ [$\pi \times$rad]': p.readable([phi_R/np.pi, phi_L/np.pi, phi_C/np.pi], [s_phiR/np.pi, s_phiL/np.pi, s_phiC/np.pi])
    }), 'table0', ' ... ')

#%% 3 -- RC seriovo
C_s = data[5:26, 12].astype(np.float64)
s_Cs = C_s*1/100

P_s = data[5:26, 13].astype(np.float64)
s_Ps = data[5:26, 14].astype(np.float64)
s_Ps = np.sqrt(s_Ps**2 + (P_s*0.5/100+10*0.001)**2)

U_s = data[5:26, 15].astype(np.float64)
s_Us = U_s*0.4/100+5*0.1

I_s = data[5:26, 16].astype(np.float64)
s_Is = I_s*0.4/100+5*0.001

pf_s = pf(P_s, U_s, I_s)
s_pfs = p.prenos_chyb_multi(pf, [s_Ps, s_Us, s_Is], [P_s, U_s, I_s])

phi_s = -phi(pf_s)
s_phis = [max_chyb(phi, pf_s[i], s_pfs[i]) for i in range(len(pf_s))]

#%% 3 -- RC parallel
C_p = data[2:26, 19].astype(np.float64)
s_Cp = C_p*1/100

P_p = data[2:26, 20].astype(np.float64)
s_Pp = data[2:26, 21].astype(np.float64)
s_Pp = np.sqrt(s_Pp**2 + (P_p*0.5/100+10*0.001)**2)

U_p = data[2:26, 22].astype(np.float64)
s_Up = U_p*0.4/100+5*0.1

I_p = data[2:26, 23].astype(np.float64)
s_Ip = I_p*0.4/100+5*0.001

pf_p = pf(P_p, U_p, I_p)
s_pfp = p.prenos_chyb_multi(pf, [s_Pp, s_Up, s_Ip], [P_p, U_p, I_p])

phi_p = phi(pf_p)
s_phip = [max_chyb(phi, pf_p[i], s_pfp[i]) for i in range(len(pf_p))]

table1 = p.default_table(pd.DataFrame({
    'C [$\mu F$]': p.readable(C_s, s_Cs),
    'P [W]': p.readable(P_s, s_Ps),
    'U [V]': p.readable(U_s, s_Us),
    'I [A]': p.readable(I_s, s_Is),
    'PF [-]': p.readable(pf_s, s_pfs),
    '$\\varphi$ [$\pi \\times$rad]': p.readable(phi_s/np.pi, np.array(s_phis)/np.pi),
    }), 'table1', '...')

table2 = p.default_table(pd.DataFrame({
    'C [$\mu F$]': p.readable(C_p, s_Cp),
    'P [W]': p.readable(P_p, s_Pp),
    'U [V]': p.readable(U_p, s_Up),
    'I [A]': p.readable(I_p, s_Ip),
    'PF [-]': p.readable(pf_p, s_pfp),
    '$\\varphi$ [$\pi \\times$rad]': p.readable(phi_p/np.pi, np.array(s_phip)/np.pi),
    }), 'table2', '...')

def phi_s_fit(xdata, om_R):
    return np.arctan(-1/(om_R*xdata))

def phi_p_fit(xdata, om_R):
    return np.arctan(om_R*xdata)

def pf_s_fit(xdata, om_R):
    return np.cos(np.abs(np.arctan(-1/(om_R*xdata))))

def pf_p_fit(xdata, om_R):
    return np.cos(np.arctan(om_R*xdata))

om_R_s, errs_s = p.curve_fit(phi_s_fit, C_s, phi_s, imports={'np':np}, p0 = [0.25], err = [s_Cs, s_phis])
om_R_p, errs_p = p.curve_fit(phi_p_fit, C_p, phi_p, imports={'np':np}, p0 = [om_R_s], err = [s_Cp, s_phip])

om_R_spf, errs_spf = p.curve_fit(pf_s_fit, C_s, pf_s, imports={'np':np}, p0 = [om_R_s], err = [s_Cs, s_pfs])
om_R_ppf, errs_ppf = p.curve_fit(pf_p_fit, C_p, pf_p, imports={'np':np}, p0 = [om_R_p], err = [s_Cp, s_pfp])

p.default_plot([C_s, C_p], [pf_s, pf_p], 'C [$\mu F$]', 'PF [-]', 
               legend = ['sériové zapojenie', 'paralelné zapojenie'], 
               xerror = [s_Cs, s_Cp], yerror = [s_pfs, s_pfp],
               fit = [[pf_s_fit, *om_R_spf], [pf_p_fit, *om_R_ppf]])

p.default_plot([C_s, C_p], [phi_s, phi_p], 'C [$\mu F$]', '$\\varphi$ [rad]', 
               legend = ['sériové zapojenie', 'paralelné zapojenie'], 
               xerror = [s_Cs, s_Cp], yerror = [s_phis, s_phip],
               fit = [[phi_s_fit, *om_R_s], [phi_p_fit, *om_R_p]])

#%% 4 -- osciloskop

C_o = data[2:12, 25].astype(np.float64)
s_Co = C_o*1/100

val = data[2:12, 27].astype(np.float64)
spec = data[2:12, 28].astype(np.float64)

phi_o = -np.pi*val*spec
s_phio = np.pi*spec/sqrt(3)

table3 = p.default_table(pd.DataFrame({
    'C [$\mu F$]': p.readable(C_o, s_Co),
    '$\\varphi$ [$\pi \\times$rad]': p.readable(phi_o/np.pi, s_phio/np.pi)
    }), 'table3', ' ... ') 

om_R_o, errs_o = p.curve_fit(phi_s_fit, C_o, phi_o, imports={'np':np}, p0 = [om_R_s], err = [s_Co, s_phio])

p.default_plot([C_s, C_o], [phi_s, phi_o], 'C [$\mu F$]', '$\\varphi$ [rad]',
               legend=['Wattmeter', 'osciloskop'],
               xerror=[s_Cs, s_Co], yerror=[s_phis, s_phio],
               fit = [[phi_s_fit, *om_R_s], [phi_s_fit, *om_R_o]])

#%% 5 -- RLC
def phi_RLC_fit(C, om_LpR, om_R):
    return np.arctan(om_LpR-1/(om_R*C))
def pf_RLC_fit(C, om_LpR, om_R):
    return np.cos(np.arctan(om_LpR-1/(om_R*C)))

C_RLC = data[5:, 29].astype(np.float64)
s_CRLC = C_RLC*1/100

P_RLC = data[5:, 30].astype(np.float64)
s_PRLC = data[5:, 31].astype(np.float64)
s_PRLC = np.sqrt(s_PRLC**2 + (P_RLC*0.5/100+10*0.001)**2)

U_RLC = data[5:, 32].astype(np.float64)
s_URLC = U_RLC*0.4/100+5*0.1

I_RLC = data[5:, 33].astype(np.float64)
s_IRLC = I_RLC*0.4/100+5*0.001

pf_RLC = pf(P_RLC, U_RLC, I_RLC)
s_pfRLC = p.prenos_chyb_multi(pf, [s_PRLC, s_URLC, s_IRLC], [P_RLC, U_RLC, I_RLC])

phi_RLC = phi(pf_RLC)
s_phiRLC = [max_chyb(phi, pf_RLC[i], s_pfRLC[i]) for i in range(len(pf_RLC))]

phi_RLC[C_RLC > 2.49] = -phi_RLC[C_RLC > 2.49]
phi_RLC = - phi_RLC

table4 = p.default_table(pd.DataFrame({
    'C [$\mu F$]': p.readable(C_RLC, s_CRLC),
    'P [W]': p.readable(P_RLC, s_PRLC),
    'U [V]': p.readable(U_RLC, s_URLC),
    'I [A]': p.readable(I_RLC, s_IRLC),
    'PF [-]': p.readable(pf_RLC, s_pfRLC),
    '$\\varphi$ [$\pi \\times$rad]': p.readable(phi_RLC/np.pi, np.array(s_phiRLC)/np.pi),
    }), 'table4', '...')

parms_pf, errs_RLCpf = p.curve_fit(pf_RLC_fit, C_RLC, pf_RLC, imports={'np':np}, err = [s_CRLC, s_pfRLC])
parms_phi, errs_RLCphi = p.curve_fit(phi_RLC_fit, C_RLC, phi_RLC, imports={'np':np}, p0 = parms_pf, err = [s_CRLC, s_phiRLC])

p.default_plot([C_RLC], [pf_RLC], 'C [$\mu F$]', 'PF [-]', 
               xerror = [s_CRLC], yerror = [s_pfRLC],
               fit = [[pf_RLC_fit, *parms_pf]])

p.default_plot([C_RLC], [phi_RLC], 'C [$\mu F$]', '$\\varphi$ [rad]', 
               xerror = [s_CRLC], yerror = [s_phiRLC],
               fit = [[phi_RLC_fit, *parms_phi]])

#%% diskuse

def R(U0, I0, pf):
    return U0/I0*pf

R_r = R(U_R, I_R, pf_R)
s_Rr = p.prenos_chyb(R, [s_UR, s_IR, s_pfR], [U_R, I_R, pf_R])
print(p.round_to(R_r, s_Rr))

R_L = R(U_L, I_L, pf_L)
s_RL = p.prenos_chyb(R, [s_UL, s_IL, s_pfL], [U_L, I_L, pf_L])
print(p.round_to(R_L, s_RL))

R_C = R(U_C, I_C, pf_C)
s_RC = p.prenos_chyb(R, [s_UC, s_IC, s_pfC], [U_C, I_C, pf_C])
print(p.round_to(R_C, s_RC))

def CX(U0, I0, phi0, f0):
    return 1/(U0/I0*np.sin(phi0)*2*np.pi*f0)
def LX(U0, I0, phi0, f0):
    return U0/I0*np.sin(phi0)/(2*np.pi*f0)

L_L = LX(U_L, I_L, phi_L, 50)
s_LL = p.prenos_chyb(LX, [s_UL, s_IL, s_phiL, 0.05], [U_L, I_L, phi_L, 50])
print(p.round_to(L_L, s_LL))

C_C = CX(U_C, I_C, phi_C, 50)
s_CC = p.prenos_chyb(CX, [s_UC, s_IC, s_phiC, 0.05], [U_C, I_C, phi_C, 50])
print(p.round_to(C_C*10**6, s_CC*10**6))
#%%
def div(om_r, om):
    return om_r/om

print(p.round_to(div(om_R_s[0]*10**6, 50*np.pi*2), p.prenos_chyb(div, [errs_s[0]*10**6, 0.05*2*np.pi], [om_R_s[0]*10**6, 50*np.pi*2])))