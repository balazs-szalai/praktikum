# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 14:55:45 2023

@author: Balázs local
"""

import numpy as np
import scipy.optimize as opt
import pandas as pd
from praktikum.format_latex import format_to_latex, prenos_chyb_latex, default_table
from praktikum.main_funcs import prenos_chyb, round_to, default_plot, reset_counter, rand_plot_colors
import matplotlib.pyplot as plt
plt.close("all")

def Re(r, rho, Q_V, eta):
    return rho*Q_V/(np.pi*r*eta)
print(format_to_latex(Re))
print(prenos_chyb_latex(Re))
#%%

def Q_Vp(r, p, eta, l):
    return np.pi*r**4/(8*eta*l)*p
print(format_to_latex(Q_Vp, parms = ['r', 'Delta p', 'eta', 'l']))

# def dp(h, rho, g):
#     return h*rho*g
# print(format_to_latex(dp))
#%%
def dp(k, r, l, rho, v_s):
    return k*(l/r)*(1/2)*rho*v_s**2
print(prenos_chyb_latex(dp))

#%%
def p_static(h, rho, g):
    return h*rho*g
print(prenos_chyb_latex(p_static))

#%%
def rho(t, p):
    return 999.842594 + 6.793952*10**(-2)*t - 9.095290*10**(-3)*t**2 + 1.001685*10**(-4)*t**3 - 1.120083*10**(-6)*t**4 + 6.536332*10**(-9)*t**5 + 0.050*10**(-3)*(1013.25-p/100)
print(format_to_latex(rho))

T_v = 21.1
p_a = 99500
rho_v = rho(T_v, 0), np.sqrt(prenos_chyb(rho, np.diag([0.1**2, 200**2]), (T_v, p_a))**2 + 10**(-8))
print(prenos_chyb_latex(rho))
print(round_to(*rho_v))

#%%
def Q_V(V, t):
    return V/t
print(prenos_chyb_latex(Q_V))

def a(Q_V, p):
    return Q_V/p

#%% A
sigma_t = 0.3 #s
sigma_l = np.sqrt(0.1/3)
sigma_d = 0.05*np.sqrt(1/3)

l_a = [25.1, 25.2, 25.2] #cm
s_la  = np.sqrt(np.std(l_a)**2 + sigma_l**2)
l_a = np.mean(l_a)
d_a = [2.2, 2.25, 2.2] #mm
s_ra = np.sqrt(np.std(d_a)**2 + sigma_d**2)/2
r_a = np.mean(d_a)/2

h_a = np.array([2,4.5,7,10,12,14,16,18,20,22,26]) #cm
s_ha = np.array([0.1]*len(h_a))/np.sqrt(3) #cm

V_a = np.array([9.4, 24, 46, 47, 47, 88, 96, 94, 92, 94, 90]) #ml
s_Va = np.array([0.2, 0.5, 1, 1, 1, 2, 2, 2, 2, 2, 2])/np.sqrt(3) #ml

t_a = np.array([44.78, 30.75, 33.81, 24.78, 20.47, 33.13, 32.88, 28.72, 25.56, 23.66, 19.59]) #s
s_ta = np.array([sigma_t]*len(t_a)) #np.sqrt((np.std(t_a)**2 + sigma_t**2)/len(t_a)) #s

#%% B
l_b = [25, 25.1, 25.1] #cm
s_lb  = np.sqrt((np.std(l_b)**2 + sigma_l**2)/3)
l_b = np.mean(l_b)
d_b = [2.65, 2.55, 2.6] #mm
s_rb = np.sqrt((np.std(d_b)**2 + sigma_d**2)/3)/2
r_b = np.mean(d_b)/2

h_b0 = np.array([1, 10, 9, 8, 7, 6, 5, 4, 3, 2, np.array([12, 15.5]), np.array([15, 17]), np.array([20, 21]), np.array([23.5, 24]), np.array([25.1, 25.5]), np.array([11.2, 12.5])], dtype = object) #cm
h_b = np.array([np.mean(i) for i in h_b0])
s_hb = np.array([0.1/np.sqrt(3)]*10 + [0]*(len(h_b)-10))
s_hb = np.array(s_hb + np.array([np.std(i) for i in h_b0])) #cm

max_dhb = np.array([np.array([(i[1] - np.mean(i))/np.mean(i), k]) if isinstance(i, np.ndarray) else [0, k]  for k, i in enumerate(h_b0)])
dhh_b = 100*np.max(max_dhb[:, 0])

V_b = np.array([9.2, 90, 96, 96, 96, 98, 100, 49, 48, 23, 98, 90, 94, 94, 96, 92]) #ml
s_Vb = np.array([0.2, 2, 2, 2, 2, 2, 2, 1, 1, 0.5, 2, 2, 2, 2, 2, 2])/np.sqrt(3) #ml

t_b = np.array([56, 19.44, 22.35, 25.06, 28.07, 33.22, 40.31, 24.35, 31.50, 32.10, 19.10, 16.41, 15.78, 14.59, 14.65, 18.69]) #s
s_tb = np.array([sigma_t]*len(t_b)) #np.sqrt((np.std(t_a)**2 + sigma_t**2)/len(t_a)) #s

#%% C
l_c = [20.4, 20.2, 20.3] #cm
s_lc  = np.sqrt((np.std(l_c)**2 + sigma_l**2)/3)
l_c = np.mean(l_c)
d_c = [3, 3, 2.95] #mm
s_rc = np.sqrt((np.std(d_c)**2 + sigma_d**2)/3)/2
r_c = np.mean(d_c)/2

h_c0 = np.array([1, 2, 2.5, 3, 3.5, 4, 4.5, 5, np.array([5.9, 6.4]), np.array([6.5, 7.6]), np.array([7.4, 8.8]), np.array([7.4, 9.5]), np.array([9, 11.1]), np.array([10.4, 11.6]), np.array([11, 12.1]), 12.6, 13.3, 14.5, 15.5, 17, 18, 19, 20, 21, 22, 23, 24], dtype = object) #cm
h_c = np.array([np.mean(i) for i in h_c0])
s_hc = np.array([0.1/np.sqrt(3)]*8 + [0]*(len(h_c)-8-12) + [0.2, 0.2] + [0.1]*10)
s_hc = np.array(s_hc + np.array([np.std(i) for i in h_c0])) #cm

max_dhc = np.array([np.array([(i[1] - np.mean(i))/np.mean(i), k]) if isinstance(i, np.ndarray) else [0, k]  for k, i in enumerate(h_c0)])
dhh_c = 100*np.max(max_dhc[:, 0])

V_c = np.array([9.4, 48, 48, 96, 98, 96, 96, 96, 96, 96, 92, 94, 94, 94, 92, 96, 94, 96, 98, 98, 96, 94, 100, 98, 96, 88, 96]) #ml
s_Vc = np.array([0.2] + [2]*(len(V_c)-1))/np.sqrt(3) #ml

t_c = np.array([42.82, 23.54, 19.37, 28.15, 28.75, 23.60, 22.25, 20.37, 17.82, 16.53, 15.41, 15, 14.31, 13.97, 13.69, 13.78, 13.41, 13.09, 12.91, 12.09, 11.59, 11.29, 11.34, 11.22, 10.5, 9.19, 10.19]) #s
s_tc = np.array([sigma_t]*len(t_c)) #np.sqrt((np.std(t_a)**2 + sigma_t**2)/len(t_a)) #s

#%%
def table(l_a, s_la, r_a, s_ra, h_a, s_ha, V_a, s_Va, t_a, s_ta, name):
    tablea = pd.DataFrame(columns=['l [cm]', 'r [mm]', 'h [cm]', 'V [ml]', 't [s]'])
    tablea['l [cm]'] = [round_to(l_a, s_la)]+['']*(len(h_a)-1)
    tablea['r [mm]'] = [round_to(r_a, s_ra)]+['']*(len(h_a)-1)
    tablea['h [cm]'] = [round_to(x, std) for x, std in zip(h_a, s_ha)]
    tablea['V [ml]'] = [round_to(x, std) for x, std in zip(V_a, s_Va)]
    tablea['t [s]'] = [round_to(x, std) for x, std in zip(t_a, s_ta)]
    tablea.index += 1
    tablea = default_table(tablea, f'table{name.lower()}', f'Všetky zamerané hodnoty včetne chýb pre trubicu {name}')
    return tablea

tablea = table(l_a, s_la, r_a, s_ra, h_a, s_ha, V_a, s_Va, t_a, s_ta, 'A')
tableb = table(l_b, s_lb, r_b, s_rb, h_b, s_hb, V_b, s_Vb, t_b, s_tb, 'B')
tablec = table(l_c, s_lc, r_c, s_rc, h_c, s_hc, V_c, s_Vc, t_c, s_tc, 'C')
#%%
plt.close('all')
reset_counter()
g = 9.81373 #m/s**2
colors = rand_plot_colors(3)

def qvdp(V_a, s_Va, t_a, s_ta, h_a, s_ha):
    qv_a = Q_V(V_a, t_a)
    s_qva = np.array([prenos_chyb(Q_V, np.diag([s_Va[i]**2, s_ta[i]**2]), (V_a[i], t_a[i])) for i in range(len(V_a))])
    
    s_ha /= 100
    dp_a = p_static(h_a/100, rho_v[0], g)
    s_dpa = np.array([prenos_chyb(p_static, np.diag([s_ha[i]**2, rho_v[1]**2, 0]), (h_a[i]/100, rho_v[0], g)) for i in range(len(h_a))])
    return qv_a, s_qva, dp_a, s_dpa

qv_a, s_qva, dp_a, s_dpa = qvdp(V_a, s_Va, t_a, s_ta, h_a, s_ha)
qv_b, s_qvb, dp_b, s_dpb = qvdp(V_b, s_Vb, t_b, s_tb, h_b, s_hb)
qv_c, s_qvc, dp_c, s_dpc = qvdp(V_c, s_Vc, t_c, s_tc, h_c, s_hc)

fig, ax = default_plot([dp_a, dp_b, dp_c], [qv_a, qv_b, qv_c], '$\Delta p \enspace [Pa]$', '$Q_V \enspace [ml/s]$', legend = ['A', 'B', 'C'],
                       yerror = [s_qva, s_qvb, s_qvc], xerror = [s_dpa, s_dpb, s_dpc], save = False, colors=colors)

#%%
def lin(x, a):
    return a*x

def r(eta, l, a):
    return (8*eta*l*a/np.pi)**(1/4)
print(prenos_chyb_latex(r))

eta = 0.975*10**(-3)
s_eta = 0.002*10**(-3)

a_a, sa_a = opt.curve_fit(lin, dp_a[np.where(dp_a < 3000)], qv_a[np.where(dp_a < 3000)])
ax.plot(np.linspace(0, 2700), lin(np.linspace(0, 2700), *a_a), '--', color = colors[0], alpha = 0.5)

a_b, sa_b = opt.curve_fit(lin, dp_b[np.where(dp_b < 1000)], qv_b[np.where(dp_b < 1000)])
ax.plot(np.linspace(0, 1100), lin(np.linspace(0, 1100), *a_b), '--', color = colors[1], alpha = 0.5)

a_c, sa_c = opt.curve_fit(lin, dp_c[np.where(dp_c < 500)], qv_c[np.where(dp_c < 500)])
ax.plot(np.linspace(0, 600), lin(np.linspace(0, 600), *a_c), '--', color = colors[2], alpha = 0.5)

sa_a = np.sqrt(sa_a[0,0] + np.sum([prenos_chyb(a, np.diag([s_qva[i]**2, s_dpa[i]**2]), (qv_a[i], dp_a[i]))**2 for i in range(len(s_qva))])/len(s_qva))
sa_b = np.sqrt(sa_b[0,0] + np.sum([prenos_chyb(a, np.diag([s_qvb[i]**2, s_dpb[i]**2]), (qv_b[i], dp_b[i]))**2 for i, k in enumerate(dp_b<1000) if k])/len(np.where(dp_b<1000)))
sa_c = np.sqrt(sa_c[0,0] + np.sum([prenos_chyb(a, np.diag([s_qvc[i]**2, s_dpc[i]**2]), (qv_c[i], dp_c[i]))**2 for i, k in enumerate(dp_c<500)])/len(np.where(dp_c<500)))

eta *= 1000 #mm/kg.s
s_eta *= 1000
l_a *= 10 #mm
l_b *= 10 #mm
l_c *= 10 #mm
s_la *= 10 #mm
s_lb *= 10 #mm
s_lc *= 10 #mm


r_a1 = r(eta, l_a, a_a[0])
s_ra1 = prenos_chyb(r, np.diag([s_eta**2, s_la**2, sa_a**2]), (eta, l_a, a_a[0]))

r_b1 = r(eta, l_b, a_b[0])
s_rb1 = prenos_chyb(r, np.diag([s_eta**2, s_lb**2, sa_b**2]), (eta, l_b, a_b[0]))

r_c1 = r(eta, l_c, a_c[0])
s_rc1 = prenos_chyb(r, np.diag([s_eta**2, s_lc**2, sa_c**2]), (eta, l_c, a_c[0]))

print(round_to(r_a1, s_ra1, unit = 'mm', latex=False))
print(round_to(r_b1, s_rb1, unit = 'mm', latex=False))
print(round_to(r_c1, s_rc1, unit = 'mm', latex=False))
#%%
def k(p, r, l, Q_V, rho):
    return 2*np.pi**2*p*r**5/(l*rho*Q_V**2)
#print(format_to_latex(k))
#print(prenos_chyb_latex(k))

def lam(Re):
    return 16/Re

def turb(Re):
    return 0.133/Re**(1/4)

Re_a = Re(r_a1, rho_v[0], qv_a, eta)
Re_b = Re(r_b1, rho_v[0], qv_b, eta)
Re_c = Re(r_c1, rho_v[0], qv_c, eta)

s_rea = [prenos_chyb(Re, np.diag([s_ra1**2, rho_v[1]**2, s_qva[i]**2, s_eta**2]), (r_a1, rho_v[0], qv_a[i], eta)) for i in range(len(qv_a))]
s_reb = [prenos_chyb(Re, np.diag([s_rb1**2, rho_v[1]**2, s_qvb[i]**2, s_eta**2]), (r_b1, rho_v[0], qv_b[i], eta)) for i in range(len(qv_b))]
s_rec = [prenos_chyb(Re, np.diag([s_rc1**2, rho_v[1]**2, s_qvc[i]**2, s_eta**2]), (r_c1, rho_v[0], qv_c[i], eta)) for i in range(len(qv_c))]

k_a = k(dp_a, r_a1, l_a, qv_a, rho_v[0])
k_b = k(dp_b, r_b1, l_b, qv_b, rho_v[0])
k_c = k(dp_c, r_c1, l_c, qv_c, rho_v[0])

s_ka = [prenos_chyb(k, np.diag([s_dpa[i]**2, s_ra1**2, s_la**2, s_qva[i]**2, rho_v[1]]), (dp_a[i], r_a1, l_a, qv_a[i], rho_v[0])) for i in range(len(qv_a))]
s_kb = [prenos_chyb(k, np.diag([s_dpb[i]**2, s_rb1**2, s_lb**2, s_qvb[i]**2, rho_v[1]]), (dp_b[i], r_b1, l_b, qv_b[i], rho_v[0])) for i in range(len(qv_b))]
s_kc = [prenos_chyb(k, np.diag([s_dpc[i]**2, s_rc1**2, s_lc**2, s_qvc[i]**2, rho_v[1]]), (dp_c[i], r_c1, l_c, qv_c[i], rho_v[0])) for i in range(len(qv_c))]

fig1, ax1 = default_plot([Re_a, Re_b, Re_c], [k_a, k_b, k_c], 'Re', 'k', 
                         legend = ['A', 'B', 'C'], xerror= [s_rea, s_reb, s_rec],
                         yerror=[s_ka, s_kb, s_kc])
ax1.plot(np.linspace(10, 2300, 1000), lam(np.linspace(10, 2300, 1000)), '--', label = 'teoretycký závislosť pre laminárne prúdenie')
ax1.plot(np.linspace(10, 2300, 1000), turb(np.linspace(10, 2300, 1000)), '--', label = 'teoretycký závislosť pre turbulentné prúdenie')
ax1.legend()