# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 15:48:51 2023

@author: balaz
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from praktikum.main_funcs import default_plot, round_to, markers, curve_fit, prenos_chyb, readable, rand_plot_colors
from praktikum.format_latex import default_table, prenos_chyb_eval, prenos_chyb_latex


data = np.array(pd.read_excel('mer3.xlsx', skiprows=2).iloc[:, :3], np.float64)

x1 = data[:, 0]-13.1
U = data[:, 1]
s_odc = data[:, 2]

s_x = 0.17
s_U = np.sqrt(2*(0.003*U+0.5e-6)**2+s_odc**2)


def k(mu_0, f, r, n):
    return 1/(mu_0*2*np.pi*f*np.pi*r**2*n)
print(prenos_chyb_latex(k, parms = ['mu_0', 'f', 'r', 'n']))

def k(r, f):
    return 1/(1.25664e-6*2*np.pi*f*1000*np.pi*r**2)
s_k = prenos_chyb(k, np.diag([1e-4**2, 0.05**2]), [1.28/100, 50])
k = k(1.28/100, 50)

def H(U, k):
    return U*k
s_H1 = [prenos_chyb(H, np.diag([s_U[i]**2, s_k**2]), [U[i], k]) for i in range(len(s_U))]
H1 = [H(U[i], k) for i in range(len(U))]

table1 = pd.DataFrame({'x [cm]': readable(x1, [0.17]*len(x1)),
                       'U [mV]': readable(U*1000, s_U*1000),
                       'H [A/m]': readable(H1, s_H1)})
table1 = default_table(table1, 'table1', 
                       'namerané hodnoty poloh, a napätí v detekčnom cievke a vypočítané hodnoty intenzity magnetického pola pre dve súosé cievky vo vzdialenosti $2a = 20$ cm pre prípad, že prúdy majú súhlasný smer',
                       header='$2a = 20$ cm, súhlasný smer prúdov')



#%%
data = np.array(pd.read_excel('mer3.xlsx', 'Sheet2', skiprows=2).iloc[:, :3], np.float64)

x3 = data[:, 0]-8.8
U = data[:, 1]
s_odc = data[:, 2]

s_x = 0.17
s_U = np.sqrt(2*(0.003*U+0.5e-6)**2+s_odc**2)
#%%

s_H3 = [prenos_chyb(H, np.diag([s_U[i]**2, s_k**2]), [U[i], k]) for i in range(len(s_U))]
H3 = [H(U[i], k) for i in range(len(U))]

table3 = pd.DataFrame({'x [cm]': readable(x3, [0.17]*len(x3)),
                       'U [mV]': readable(U*1000, s_U*1000),
                       'H [A/m]': readable(H3, s_H3)})
table3 = default_table(table3, 'table3', 
                       'namerané hodnoty poloh, a napätí v detekčnom cievke a vypočítané hodnoty intenzity magnetického pola pre dve súosé cievky vo vzdialenosti $2a = R = 10.4$ cm pre prípad, že prúdy majú súhlasný smer',
                       header='$2a = R$, súhlasný smer prúdov')



#%%
data = np.array(pd.read_excel('mer3.xlsx', 'Sheet3', skiprows=2).iloc[:, :3], np.float64)

x4 = data[:, 0]-8.8
U = data[:, 1]
s_odc = data[:, 2]

s_x = 0.17
s_U = np.sqrt(2*(0.003*U+0.5e-6)**2+s_odc**2)

s_H4 = [prenos_chyb(H, np.diag([s_U[i]**2, s_k**2]), [U[i], k]) for i in range(len(s_U))]
H4 = [H(U[i], k) for i in range(len(U))]

table4 = pd.DataFrame({'x [cm]': readable(x4, [0.17]*len(x4)),
                       'U [mV]': readable(U*1000, s_U*1000),
                       'H [A/m]': readable(H4, s_H4)})
table4 = default_table(table4, 'table4', 
                       'namerané hodnoty poloh, a napätí v detekčnom cievke a vypočítané hodnoty intenzity magnetického pola pre dve súosé cievky vo vzdialenosti $2a = R = 10.4$ cm pre prípad, že prúdy majú nesúhlasný smer',
                       header='$2a = R$, nesúhlasný smer prúdov')


#%%
data = np.array(pd.read_excel('mer3.xlsx', 'Sheet4', skiprows=2).iloc[:, :3], np.float64)

x2 = data[:, 0]-13.1
U = data[:, 1]
s_odc = data[:, 2]

s_x = 0.17
s_U = np.sqrt(2*(0.003*U+0.5e-6)**2+s_odc**2)

s_H2 = [prenos_chyb(H, np.diag([s_U[i]**2, s_k**2]), [U[i], k]) for i in range(len(s_U))]
H2 = [H(U[i], k) for i in range(len(U))]

table2 = pd.DataFrame({'x [cm]': readable(x2, [0.17]*len(x2)),
                       'U [mV]': readable(U*1000, s_U*1000),
                       'H [A/m]': readable(H2, s_H2)})
table2 = default_table(table2, 'table2', 
                       'namerané hodnoty poloh, a napätí v detekčnom cievke a vypočítané hodnoty intenzity magnetického pola pre dve súosé cievky vo vzdialenosti $2a = 20$ cm pre prípad, že prúdy majú súhlasný smer',
                       header='$2a = 20$ cm, nesúhlasný smer prúdov')

#%%

def H(x, N, I, R, a, pm):
    return abs(N*I*R**2/2*(1/(R**2+(a+x)**2)**(3/2) + pm*1/(R**2+(a-x)**2)**(3/2)))

fig, ax = default_plot([x1, x2, x3, x4], [H1, H2, H3, H4], 'x [cm]', 'H [A/m]', 
                       legend = ['$2a = 20$ cm, súhlasný smer prúdov', 
                                 '$2a = 20$ cm, nesúhlasný smer prúdov', 
                                 '$2a = R = 10.4$ cm, súhlasný smer prúdov', 
                                 '$2a = R = 10.4$ cm, nesúhlasný smer prúdov'],
                       xerror = [[0.17]*len(x1), [0.17]*len(x2), [0.17]*len(x3), [0.17]*len(x4)], 
                       yerror = [s_H1, s_H2, s_H3, s_H4])

x = np.linspace(-7.5, 7.5, 1000)
s_H_teor_p0 = [prenos_chyb(H, np.diag([0, 0, 0.03**2, (0.1/100)**2, (0.1/100)**2, 0]), [i/100, 100, 2.95/2, 10.4/100, 10/100, 1]) for i in x]
s_H_teor_p1 = [prenos_chyb(H, np.diag([0, 0, 0.03**2, (0.1/100)**2, (0.1/100)**2, 0]), [i/100, 100, 2.95/2, 10.4/100, 11.4/2/100, 1]) for i in x]
s_H_teor_n0 = [prenos_chyb(H, np.diag([0, 0, 0.03**2, (0.1/100)**2, (0.1/100)**2, 0]), [i/100, 100, 2.95/2, 10.4/100, 10/100, -1]) for i in x]
s_H_teor_n1 = [prenos_chyb(H, np.diag([0, 0, 0.03**2, (0.1/100)**2, (0.1/100)**2, 0]), [i/100, 100, 2.95/2, 10.4/100, 11.4/2/100, -1]) for i in x]

ax.plot(x, H(x/100, 100, 2.95/2, 10.4/100, 10/100, 1), '--', color = rand_plot_colors(4)[0])
ax.fill_between(x, H(x/100, 100, 2.95/2, 10.4/100, 10/100, 1)+s_H_teor_p0, H(x/100, 100, 2.95/2, 10.4/100, 10/100, 1)-s_H_teor_p0, alpha = 0.5, color = rand_plot_colors(4)[0])

ax.plot(x, H(x/100, 100, 2.95/2, 10.4/100, 10/100, -1), '--', color = rand_plot_colors(4)[1])
ax.fill_between(x, H(x/100, 100, 2.95/2, 10.4/100, 10/100, -1)+s_H_teor_n0, H(x/100, 100, 2.95/2, 10.4/100, 10/100, -1)-s_H_teor_n0, alpha = 0.5, color = rand_plot_colors(4)[1])

ax.plot(x, H(x/100, 100, 2.95/2, 10.4/100, 11.4/2/100, 1), '--', color = rand_plot_colors(4)[2])
ax.fill_between(x, H(x/100, 100, 2.95/2, 10.4/100, 11.4/2/100, 1)+s_H_teor_p1, H(x/100, 100, 2.95/2, 10.4/100, 11.4/2/100, 1)-s_H_teor_p1, alpha = 0.5, color = rand_plot_colors(4)[2])

ax.plot(x, H(x/100, 100, 2.95/2, 10.4/100, 11.4/2/100, -1), '--',  color = rand_plot_colors(4)[3])
ax.fill_between(x, H(x/100, 100, 2.95/2, 10.4/100, 11.4/2/100, -1)+s_H_teor_n1, H(x/100, 100, 2.95/2, 10.4/100, 11.4/2/100, -1)-s_H_teor_n1, alpha = 0.5, color = rand_plot_colors(4)[3])
#%%

def H(M, J, R):
    return 8/5/np.sqrt(5)*M*J/R
s_H = prenos_chyb_eval(H, [0, 0.03, 0.1/100], [100, 2.95/2, 10.4/100])
H = H(100, 2.95/2, 10.4/100)
print(round_to(H, s_H))

#%%
data = np.array(pd.read_excel('mer3.xlsx', 'Sheet2', skiprows=2).iloc[:, :3], np.float64)

x3 = data[:, 0]-8.3
U = data[:, 1]
s_odc = data[:, 2]

s_U = np.sqrt(np.std(U)**2 + sum(2*(0.003*U+0.5e-6)**2+s_odc**2)/len(U)**2)
#%%
def div(H, U):
    return H/U
s_k = prenos_chyb_eval(div, [s_H, s_U], [H, np.mean(U)])
print(round_to(H/np.mean(U), s_k))
#%%
def k(r, f):
    return 1/(1.25664e-6*2*np.pi*f*1000*np.pi*r**2)
s_k = prenos_chyb(k, np.diag([1e-4**2, 0.05**2]), [1.28/100, 50])
k = k(1.28/100, 50)

def H(U, k):
    return U*k

data = np.array(pd.read_excel('mer3.xlsx', 'Sheet5').iloc[:, :3], np.float64)

d = data[:, 0]
U = data[:, 1]
s_odc = data[:, 2]

s_d = 0.2
s_U = np.sqrt(2*(0.003*U+0.5e-6)**2+s_odc**2)

s_H5 = [prenos_chyb(H, np.diag([s_U[i]**2, s_k**2]), [U[i], k]) for i in range(len(s_U))]
H5 = [H(U[i], k) for i in range(len(U))]

table5 = pd.DataFrame({'d [cm]': readable(d, [0.2]*len(d)),
                       'U [mV]': readable(U*1000, s_U*1000),
                       'H [A/m]': readable(H5, s_H5)})
table5 = default_table(table5, 'table5', 
                       'namerané hodnoty vzdialenosti a napätí v detekčnom cievke a vypočítané hodnoty intenzity magnetického pola pre dve súosé cievky so súhlasným smerom prúdov v cievkach')

#%%
def H(x, M, J, R, a, pm):
    return M*J*R**2/2*(1/(R**2+(a+x)**2)**(3/2) + pm*1/(R**2+(a-x)**2)**(3/2))
print(prenos_chyb_latex(H))
#%%
fig, ax = default_plot([d], [H5], 'd [cm]', 'H [A/m]', 
                       xerror=[[0.2]*len(d)], yerror=[s_H5])

x = np.linspace(7, 20, 1000)

s_H_teor = [prenos_chyb(H, np.diag([0, 0, 0.03**2, (0.1/100)**2, (0.1/100)**2, 0]), [0, 100, 2.95/2, 10.4/100, i/2/100, 1]) for i in x]
ax.plot(x, H(0, 100, 2.95/2, 10.4/100, x/2/100, 1), color = rand_plot_colors(1)[0])
ax.fill_between(x, H(0, 100, 2.95/2, 10.4/100, x/2/100, 1)+s_H_teor, H(0, 100, 2.95/2, 10.4/100, x/2/100, 1)-s_H_teor, alpha = 0.5, color=rand_plot_colors(1)[0])
#%%

data = np.array(pd.read_excel('mer3.xlsx', 'Sheet6', skiprows=1).iloc[:, :2], np.float64)

x6 = data[:, 0]
U = data[:, 1]

s_x = 0.1
s_U = 0.003*U+0.5e-6

def k(r, f):
    return 1/(1.25664e-6*2*np.pi*f*370*np.pi*r**2)
s_k = prenos_chyb(k, np.diag([1e-4**2, 0.05**2]), [1.05/100, 50])
k = k(1.05/100, 50)
print(round_to(k, s_k))

def H(U, k):
    return U*k
s_H6 = [prenos_chyb(H, np.diag([s_U[i]**2, s_k**2]), [U[i], k]) for i in range(len(s_U))]
H6 = [H(U[i], k) for i in range(len(U))]

table6 = pd.DataFrame({'x [cm]': readable(x6, [0.1]*len(x6)),
                       'U [mV]': readable(U*1000, s_U*1000),
                       'H [A/m]': readable(H6, s_H6)})
table6 = default_table(table6, 'table6', 
                       'namerané hodnoty polôh a npätí detekčnej cievky a k tomu vypočítané intenzity magnetického poľa')

def H(x, N, I, l, r_2, r_1):
    return I*N/(2*l*(r_2-r_1))*((l/2+x)*np.log((r_2+np.sqrt(r_2**2+(l/2+x)**2))/(r_1+np.sqrt(r_1**2+(l/2+x)**2)))+(l/2-x)*np.log((r_2+np.sqrt(r_2**2+(l/2-x)**2))/(r_1+np.sqrt(r_1**2+(l/2-x)**2))))

x = np.linspace(-20, 20, 1000)

fig, ax = default_plot([x6], [H6], 'x [cm]', 'H [A/m]',
                       xerror=[[s_x]*len(x6)], yerror=[s_H6])

s_H_teor = [prenos_chyb(H, np.diag([0, 0, 0.006**2, (1/1000)**2, (1/1000)**2, (1/1000)**2]), [i/100, 4204, 0.099, 400/1000, 80/2/1000, 140/2/1000]) for i in x]
ax.plot(x, H(x/100, 4204, 0.099, 400/1000, 80/2/1000, 140/2/1000), color = rand_plot_colors(1)[0])
ax.fill_between(x, H(x/100, 4204, 0.099, 400/1000, 80/2/1000, 140/2/1000)+s_H_teor, H(x/100, 4204, 0.099, 400/1000, 80/2/1000, 140/2/1000)-s_H_teor, alpha = 0.5, color=rand_plot_colors(1)[0])