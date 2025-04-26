# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 10:46:37 2023

@author: balaz
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = np.array(pd.read_excel('mer2.xlsx').iloc[1:, :], dtype = np.float64).T.reshape(5, 2, 13)
columns = np.array(pd.read_excel('mer2.xlsx').columns)
cols = []
for column in columns:
    try:
        cols.append(np.float64(column))
    except:
        continue
U_g = np.array(cols)
from praktikum.main_funcs import default_plot, round_to, markers, curve_fit
from praktikum.format_latex import default_table, prenos_chyb_eval, prenos_chyb_latex
#%%
U = data[:, 0, :]
I = data[:, 1, :]

s_I = []
for i in I:
    s_I.append([])
    for i_a in i:
        if i_a < 0.4:
            s_I[-1].append(i_a*0.008+0.0002*2)
        elif i_a < 4:
            s_I[-1].append(i_a*0.012+0.002*2)
s_I = np.array(s_I)

s_U = []
for u in U:
    s_U.append([])
    for u_a in u:
        s_U[-1].append(u_a*0.001+0.01*3)
s_U = np.array(s_U)

s_U_g = []
for u in U_g:
    s_U_g.append(u*0.001+0.0001*3)
s_U_g = np.array(s_U_g)

I_a = {U_g[i]: {'U': U[i], 'I': I[i]} for i in range(5)}

data = pd.read_excel('mer2.xlsx', 'Sheet2', skiprows = 1)

A_f_d = {100_000: {'f': np.array(data['f [Hz]']), 'U': np.array(data['U_vyst']), 'U_a': np.array(data['U_a'])},
         5_000  : {'f': np.array(data['f']), 'U': np.array(data['U_vyst.1']), 'U_a.1': np.array(data['U_a'])}}

data = pd.read_excel('mer2.xlsx', 'Sheet3', skiprows=1)

A_R_d = {1_000: {'R': np.array(data['R_a']), 'U': np.array(data['U_vyst']), 'U_a': np.array(data['U_a'])}}


table1 = pd.DataFrame({key: np.array([round_to(val, err) for val, err in zip(vals, errs)]) for key, vals, errs in zip([f'U{i}' if i%2 == 0 else f'I{i}' for i in range(10)], [U[i//2] if i%2 == 0 else I[i//2] for i in range(10)], [s_U[i//2] if i%2 == 0 else s_I[i//2] for i in range(10)])})
table1  = default_table(table1, 'I_a', 'namerané hodnoty anódovej charakteristiky, $I_a$ je prúd tečiaci  cez triódu, $U_a$ je anódové napätie a $U_g$ je mriežkové napätie')

fig = plt.figure()
ax = fig.add_subplot(projection = '3d')
for i in range(5):
    ax.errorbar(U[i], [-U_g[i]]*13, I[i]*U[i], s_I[i], marker = markers[i], label = f'$U_g = {-U_g[i]}$ V')
ax.plot_surface(*np.meshgrid(np.linspace(0, 120), np.linspace(0, -2)), np.array([np.linspace(200, 200)]*50), color = 'blue', alpha = 0.1)

def P(U, J):
    return U*J
print(prenos_chyb_eval(P, [s_U[0][-1], s_I[0][-1]], [U[0][-1], I[0][-1]]))




#%%
def fit(x, k, D):
    return k*(u_g + D*x)**(3/2)
def fit0(x, k):
    return k*x**(3/2)
#%%
fit_I_a = {}
i = 0
fit_I_a[U_g[0]] = curve_fit(fit0, U[i][I[i]>0.03], I[i][I[i]>0.03], [s_U[i][I[i]>0.03], s_I[i][I[i]>0.03]], global_vars={'u_g': -U_g[0]}, p0 = [0.01])
for i, u_g in enumerate(U_g[1:], 1):
    if any(I[i] > 0.03):
        fit_I_a[u_g] = curve_fit(fit, U[i][I[i]>0.03], I[i][I[i]>0.03], [s_U[i][I[i]>0.03], s_I[i][I[i]>0.03]], global_vars={'u_g': -u_g}, p0 = [0.4, 0.02])

fit_I_a[U_g[0]] = [[fit_I_a[U_g[0]][0]], [np.sqrt(fit_I_a[U_g[0]][1][i]**2+fit_I_a[U_g[0]][2][i]**2) for i in range(1)]]
for key, val in fit_I_a.items():
    if len(val[0]) == 2:
        fit_I_a[key] = [[*val[0]], [np.sqrt(val[1][i]**2+val[2][i]**2) for i in range(2)]]
def inv(x):
    return 1/x

for key, val in fit_I_a.items():
    if len(val[0]) == 2:
        fit_I_a[key] = [[val[0][0], 1/val[0][1]], [val[1][0], prenos_chyb_eval(inv, [val[1][1]], [val[0][1]])]]
# ax.plot_surface(*np.meshgrid(np.linspace(0, 120), np.linspace(0, -2)), np.array([np.linspace(1, 1)]*50), color = 'blue', alpha = 0.1)
    
#%%
def fit(x, u_g, k, mu):
    return k*(u_g + x/mu)**(3/2)
def fit0(x, u_g, k):
    return k*x**(3/2)
from praktikum.main_funcs import rand_plot_colors
colors = rand_plot_colors(10)
fig, ax = default_plot(U, I, 'U [V]', 'I [mA]',legend = [f'$U_g = {-U_g[i]}$ V' for i in range(len(U_g))])
ax.plot(np.linspace(0, 120), fit0(np.linspace(0, 120),-U_g[0], 0.00125996), '--', color = colors[0])
ax.plot(np.linspace(0, 120), fit(np.linspace(0, 120),-U_g[1], 0.40657198691061974, 45.74999008555068),'--', color = colors[1])
ax.plot(np.linspace(0, 120), fit(np.linspace(0, 120),-U_g[2], 0.36233005799975954, 54.38174225380609),'--', color = colors[2])
ax.plot(np.linspace(0, 120), fit(np.linspace(0, 120),-U_g[3], 0.13954391835838484, 53.554042408854535),'--', color = colors[3])

ax.plot(np.linspace(0, 120), np.linspace(120/5, 0), label = 'zatažovacia priamka pre $R_a = 5k \Omega$')
ax.plot(np.linspace(0, 120), np.linspace(120/100, 0), label = 'zatažovacia priamka pre $R_a = 100k \Omega$')
# ax.plot(np.linspace(0, 120), np.linspace(0, 120/100))
# ax.plot(np.linspace(0, 120), np.linspace(0, 120/5))
#%%
import scipy.optimize as opt
from praktikum.main_funcs import prenos_chyb

def f_to_solve(x, u_g, k, mu, R_a):
    return 120/R_a-x/R_a-fit(x, u_g, k, mu)
def f_to_solve0(x, u_g, k, R_a):
    return 120/R_a-x/R_a-fit0(x, u_g, k)
def interpol(x, x0, y0, x1, y1):
    return x*(y1-y0)/(x1-x0)+y0-x0*(y1-y0)/(x1-x0)
def f_to_solve1(x, R_a):
    return 120/R_a-x/R_a-interpol(x, U[-1][-2], I[-1][-2], U[-1][-1], I[-1][-1])
U00_100 = opt.fsolve(lambda x: f_to_solve0(x, -U_g[0], 0.00125996, 100), 100)
U05_100 = opt.fsolve(lambda x: f_to_solve(x, -U_g[1], 0.40657198691061974, 45.74999008555068, 100), 100)
U10_100 = opt.fsolve(lambda x: f_to_solve(x, -U_g[2], 0.36233005799975954, 54.38174225380609, 100), 100)
U15_100 = opt.fsolve(lambda x: f_to_solve(x, -U_g[3], 0.13954391835838484, 53.554042408854535, 100), 100)
U20_100 = opt.fsolve(lambda x: f_to_solve1(x, 100), 100)

U_100 = [*U00_100, *U05_100, *U10_100, *U15_100, *U20_100]

U00_5 = opt.fsolve(lambda x: f_to_solve0(x, -U_g[0], 0.00125996, 5), 110)
U05_5 = opt.fsolve(lambda x: f_to_solve(x, -U_g[1], 0.40657198691061974, 45.74999008555068, 5), 110)
U10_5 = opt.fsolve(lambda x: f_to_solve(x, -U_g[2], 0.36233005799975954, 54.38174225380609, 5), 110)
U15_5 = opt.fsolve(lambda x: f_to_solve(x, -U_g[3], 0.13954391835838484, 53.554042408854535, 5), 110)
U20_5 = opt.fsolve(lambda x: f_to_solve1(x, 5), 110)

U_5 = [*U00_5, *U05_5, *U10_5, *U15_5, *U20_5]

I00_100 = fit0(U00_100, -U_g[0], 0.00125996)
I05_100 = fit(U05_100, -U_g[1], 0.40657198691061974, 45.74999008555068)
I10_100 = fit(U10_100, -U_g[2], 0.36233005799975954, 54.38174225380609)
I15_100 = fit(U15_100, -U_g[3], 0.13954391835838484, 53.554042408854535)
I20_100 = interpol(U20_100, U[-1][-2], I[-1][-2], U[-1][-1], I[-1][-1])

I_100 = [*I00_100, *I05_100, *I10_100, *I15_100, *I20_100]

I00_5 = fit0(U00_5, -U_g[0], 0.00125996)
I05_5 = fit(U05_5, -U_g[1], 0.40657198691061974, 45.74999008555068)
I10_5 = fit(U10_5, -U_g[2], 0.36233005799975954, 54.38174225380609)
I15_5 = fit(U15_5, -U_g[3], 0.13954391835838484, 53.554042408854535)
I20_5 = interpol(U20_5, U[-1][-2], I[-1][-2], U[-1][-1], I[-1][-1])

I_5 = [*I00_5, *I05_5, *I10_5, *I15_5, *I20_5]



from praktikum.main_funcs import part_der
Ri_100 = []
Ri_100.append(part_der(fit0, np.array([*U00_100, -U_g[0], 0.00125996]), 0)/1000)
Ri_100.append(part_der(fit, [*U05_100, -U_g[1], 0.40657198691061974, 45.74999008555068], 0)/1000)
Ri_100.append(part_der(fit, [*U10_100, -U_g[2], 0.36233005799975954, 54.38174225380609], 0)/1000)
Ri_100.append(part_der(fit, [*U10_100, -U_g[3], 0.13954391835838484, 53.554042408854535], 0)/1000)
Ri_100.append((I[-1][-2] - I[-1][-1]) / (U[-1][-2] - U[-1][-1])/1000)

Ri_5 = []
Ri_5.append(part_der(fit0, [*U00_5, -U_g[0], 0.00125996], 0)/1000)
Ri_5.append(part_der(fit, [*U05_5, -U_g[1], 0.40657198691061974, 45.74999008555068], 0)/1000)
Ri_5.append(part_der(fit, [*U10_5, -U_g[2], 0.36233005799975954, 54.38174225380609], 0)/1000)
Ri_5.append(part_der(fit, [*U10_5, -U_g[3], 0.13954391835838484, 53.554042408854535], 0)/1000)
Ri_5.append((I[-1][-2] - I[-1][-1]) / (U[-1][-2] - U[-1][-1])/1000)

def A(mu, Ri, Ra):
    return mu*(Ra)/(Ri+Ra)

table5 = pd.DataFrame({'$U_g$ [V]': np.array([round(i, 2) for i in U_g]).astype(np.str_), 
                       '$U$ [V]': np.array([round(i, 2) for i in U_100]).astype(np.str_), '$I$ [mA]': np.array([round(i, 2) for i in I_100]).astype(np.str_), 
                       '$R_i$ [k\Omega]': np.array([round(1/i/1000, 0) for i in Ri_100]).astype(np.str_), 'A [-]': np.array([round(A(mu, 1/ri, 100000), 1) for mu, ri in zip([0, 46, 54, 54, 0], Ri_100)]).astype(np.str_),
                       '$U1$ [V]': np.array([round(i, 2) for i in U_5]).astype(np.str_), '$I1$ [mA]': np.array([round(i, 2) for i in I_5]).astype(np.str_),
                       '$R_i1$ [k\Omega]': np.array([round(1/i/1000, 0) for i in Ri_5]).astype(np.str_), 'A1 [-]': np.array([round(A(mu, 1/ri, 100000), 1) for mu, ri in zip([0, 46, 54, 54, 0], Ri_5)]).astype(np.str_),})
table5 = default_table(table5, 'table5', 'pracovné body triódy')
#%%

s_U_vyst = {100000: [], 5000: []}
for u in A_f_d[100000]['U']:
    s_U_vyst[100000].append(u*0.05)
for u in A_f_d[5000]['U']:
    s_U_vyst[5000].append(u*0.05)

table2 = pd.DataFrame({'f [Hz]': A_f_d[100000]['f'], '$U_vyst$ [V] ($R_a = 100 \enspace k \Omega$)': [round_to(u, s) for u, s in zip(A_f_d[100000]['U'], s_U_vyst[100000])], '$U_vyst$ [V] ($R_a = 5 \enspace k \Omega$)': [round_to(u, s) for u, s in zip(A_f_d[5000]['U'], s_U_vyst[5000])]})
table2 = default_table(table2, 'table2', 'výstupné napätie v závislosti na frukvencie pri vstupnom napätí 0.2 V')

fig, ax = default_plot([A_f_d[100000]['f'], A_f_d[5000]['f']], [A_f_d[100000]['U'], A_f_d[5000]['U']], 'f [Hz]', '$U_{vyst}$ [V]', legend=['$R_a = 100 k \Omega$', '$R_a = 5 k \Omega$'], xerror=[[0]*len(A_f_d[100000]['f'])]*2, yerror=[s_U_vyst[100000], s_U_vyst[5000]])

def A(U_vst, U_vyst):
    return U_vyst/U_vst

table3 = pd.DataFrame({'f [Hz]': A_f_d[100000]['f'], 'A [-] ($R_a = 100 \enspace k \Omega$)': [round_to(u/0.2, s/0.2) for u, s in zip(A_f_d[100000]['U'], s_U_vyst[100000])], 'A [-] ($R_a = 5 \enspace k \Omega$)': [round_to(u/0.2, s/0.2) for u, s in zip(A_f_d[5000]['U'], s_U_vyst[5000])]})
table3 = default_table(table3, 'table3', 'frekvenčný závislosť zosilnenie $A$ pri vstupnom napätí $U_vst = 0.2 \enspace V$ a mriežkovom napätí $U_g = -1 \enspace V$')
fig, ax = default_plot([A_f_d[100000]['f'], A_f_d[5000]['f']], [A_f_d[100000]['U']/0.2, A_f_d[5000]['U']/0.2], 'f [Hz]', 'A [-]', legend=['$R_a = 100 k \Omega$', '$R_a = 5 k \Omega$'], xerror=[[0]*len(A_f_d[100000]['f'])]*2, yerror=[np.array(s_U_vyst[100000])/0.2, np.array(s_U_vyst[5000])/0.2])
#%%

s_U_vyst_R = []
for u in A_R_d[1000]['U']:
    s_U_vyst_R.append(u*0.05)

table4 = pd.DataFrame({'R [$\Omega$]': [round_to(r, r*0.001) for r in A_R_d[1000]['R']], 'A [-]': [round_to(u/0.2, s/0.2) for u, s in zip(A_R_d[1000]['U'], s_U_vyst_R)]})
table4 = default_table(table4, 'table4', 'zosilnenie v závislosti na frekvencie pri vstupnom napätí 0.2 V a mriežkovom napätí -1 V')
fig, ax = default_plot([A_R_d[1000]['R']], [A_R_d[1000]['U']/0.2], 'R [$\Omega$]', 'A [-]', legend=['f = 1 kHz'], xerror=[np.array(A_R_d[1000]['R'])*0.001], yerror=[np.array(s_U_vyst_R)/0.2])