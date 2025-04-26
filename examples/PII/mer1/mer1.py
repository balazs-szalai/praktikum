# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 16:42:59 2023

@author: balaz
"""

import numpy as np
import pandas as pd


data = pd.read_excel('mer1.xlsx', '50')
mer_dat_50 = {'R_L': np.array(data.iloc[1:, 0], np.float64), 'U': np.array(data.iloc[1:, 1], np.float64)}
mer_dat_100 = {'R_L': np.array(data.iloc[1:29, 5], np.float64), 'U': np.array(data.iloc[1:29, 6], np.float64)}
mer_dat_int = {'int': np.array(data.iloc[:12, 10], np.float64), 'U_OC': np.array(data.iloc[:12, 11], np.float64), 'I_SC': np.array(data.iloc[:12, 12], np.float64)}

from praktikum.main_funcs import default_plot, round_to
from praktikum.format_latex import default_table, prenos_chyb_latex, prenos_chyb_eval

s_U_50 = np.array([0.00015*i +0.004*1e-6 if i<0.2 else 0.00015*i +0.03*1e-5 for i in mer_dat_50['U']])
s_U_100 = np.array([0.00015*i +0.004*1e-6 if i<0.2 else 0.00015*i +0.03*1e-5 for i in mer_dat_100['U']])
s_U_int = np.array([0.00015*i +0.004*1e-6 if i<0.2 else 0.00015*i +0.03*1e-5 for i in mer_dat_int['U_OC']])
s_I_int = np.array([0.00020*i +0.005*1e-6 if i<200 else 0.00015*i +0.005*1e-5 for i in mer_dat_int['I_SC']])/1000

# table1 = pd.DataFrame({'$R_l$ [$\Omega$]': mer_dat_50['R_L'].astype(np.str_), '$U$ [mV]':[round_to(i*1000, si*1000) for i, si in zip(mer_dat_50['U'], s_U_50)]})
# # print(default_table(table1, 'mer_dat_50', 'zamerané hodnoty napätia $U$ v závislosti na odpore záťaže $R_L$ pri výkonu žiarenia 50\\% z $P_{sun}$'))

# table2 = pd.DataFrame({'intenzita [P_{sun}]': mer_dat_int['int'].astype(np.str_),
#                         '$I_{SC}$ [$\mu$A]':[round_to(i*1000, si*1000) for i, si in zip(mer_dat_int['I_SC'], s_I_int)],
#                         '$U_{OC}$ [mV]':[round_to(i*1000, si*1000) for i, si in zip(mer_dat_int['U_OC'], s_U_int)]})#
# print(default_table(table2, 'mer_dat_int', 'zamerané hodnoty napätia $U_{OC}$ a $I_{SC}$ v závislosti na výkonu žiarenia'))

# table3 = pd.DataFrame({'$R_l$ [$\Omega$]': mer_dat_100['R_L'].astype(np.str_), '$U$ [mV]':[round_to(i*1000, si*1000) for i, si in zip(mer_dat_100['U'], s_U_100)]})
# print(default_table(table3, 'mer_dat_100', 'zamerané hodnoty napätia $U$ v závislosti na odpore záťaže $R_L$ pri výkonu žiarenia 100\\% z $P_{sun}$'))

q = 1.602e-19
k_B = 1.380649e-23
def n(a, T, q, k_B):
    return q/(a*k_B*T)
print(prenos_chyb_latex(n))

def fit(x, a, b):
    return np.e**(a*x+b)

I_SC = mer_dat_int['I_SC']/1000
U_OC = mer_dat_int['U_OC']

logI = np.log(I_SC)[1:]
U_OC = U_OC[1:]

parms, cov = np.polyfit(U_OC, logI, 1, cov = True)
a, b = parms

I_SC = mer_dat_int['I_SC']/1000
U_OC = mer_dat_int['U_OC']

# default_plot([U_OC], [I_SC], '$U_{OC}$ [V]', '$I_{SC}$ [A]', fit = [[fit, a, b]])
#%%
logI = np.log(I_SC)[1:]
U_OC = U_OC[1:]

parms, cov = np.polyfit(U_OC, logI, 1, cov = True)
a, b = parms
s_a0 = cov[0,0]
s_b0 = cov[1,1]
s_a = np.sqrt(0.04357707315995229**2+s_a0)
s_b = np.sqrt(0.0230993135550955**2+s_b0)
print(round_to(n(a, 301, q, k_B), prenos_chyb_eval(n, [s_a, 4, 0, 0], [a, 301, q, k_B])))

def I_0(b):
    return 2.718281828459045**b
print(prenos_chyb_latex(I_0))
print(round_to(I_0(b), prenos_chyb_eval(I_0, [s_b], [b])))


#%%

a = []
b = []

n=100000

# # sus = np.array([np.random.uniform(low = -s_U_int[1:][i], high = s_U_int[1:][i], size = n) for i in range(len(s_U_int[1:]))]).T
# # sis = np.array([np.random.uniform(low = -s_I_int[1:][i]/I_SC[1:][i], high = s_I_int[1:][i]/I_SC[1:][i], size = n) for i in range(len(s_I_int[1:]))]).T

U50 = mer_dat_50['U']
R_L50 = mer_dat_50['R_L']

s_I_50 = s_U_50/R_L50
I50 = I = U50/R_L50

P50 = U50*I50
s_P_50 = P50*(s_U_50/U50+s_I_50/I50)

U100 = mer_dat_100['U']
R_L100 = mer_dat_100['R_L']

s_I_100 = s_U_100/R_L100
I100 = I = U100/R_L100

P100 = U100*I100
s_P_100 = P100*(s_U_100/U100+s_I_100/I100)

for i in range(n):
    a0, b0 = np.polyfit(np.random.normal(I50[:7], s_I_int[:7]), np.random.normal(U50[:7], s_U_50[:7]), 1)
    a.append(a0)
    b.append(b0)

a = np.array(a)
b = np.array(b)

s_a50 = np.std(a)
s_b50 = np.std(b)
#%%
a, b = [], []
for i in range(n):
    a0, b0 = np.polyfit(np.random.normal(U100[:2], s_U_100[:2]), np.random.normal(I100[:2], s_I_int[:2]), 1)
    a.append(a0)
    b.append(b0)

a = np.array(a)
b = np.array(b)

s_a100 = np.std(a)
s_b100 = np.std(b)

#%%
def lin(x, a, b):
    return a*x+b
parms, cov = np.polyfit(I50[:7], U50[:7], 1, cov = True)
a50, b50 = parms
s_a50_0, s_b50_0 = cov[0,0], cov[1,1]

s_a50_c = np.sqrt(s_a50**2+s_a50_0)
print(round_to(-a50, s_a50_c))

# a100, b100 = np.polyfit(U100[:3], I100[:3], 1)

# fig, ax = default_plot([R_L50, R_L100], [P50, P100], '$R_L$ [$\Omega$]', 'P [W]', legend = ['50% intenzita z $P_{sun}$', '100% intenzita z $P_{sun}$'])#, yerror=[s_P_50, s_P_100], xerror = [[0]*34, [0]*28])
# fig1, ax1 = default_plot([U50, U100], [I50, I100], '$U$ [V]', 'I [A]', legend = ['50% intenzita z $P_{sun}$', '100% intenzita z $P_{sun}$'], fit=[[lin, a50, b50], [lin, a100, b100]])#, yerror=[s_P_50, s_P_100], xerror = [[0]*34, [0]*28])
#%%
def FF(U_mp, I_mp, U_OC, I_SC):
    return (U_mp*I_mp)/(U_OC*I_SC)
print(prenos_chyb_latex(FF))

I_SC = mer_dat_int['I_SC']/1000
U_OC = mer_dat_int['U_OC']

print(round_to(FF(0.38722, 0.07592549019607844, U_OC[10], I_SC[10]), prenos_chyb_eval(FF, [0.005779276194588149, 0.0013279674780935539, s_U_int[10], s_I_int[10]], (0.38722, 0.07592549019607844, U_OC[10], I_SC[10]))))
#%%
area = 3
def eta(U_mp, I_mp, area, P_sun):
    return (U_mp*I_mp)/(area*P_sun)
print(prenos_chyb_latex(eta))

print(round_to(eta(0.38722, 0.07592549019607844, 3, 0.1), prenos_chyb_eval(eta, [0.005779276194588149, 0.0013279674780935539,0, 0.1*0.07], [0.38722, 0.07592549019607844, 3, 0.1])))
#%%
from praktikum.format_latex import format_to_latex
def R_S(U_1, U_2, I_SC1, I_SC2):
    return (U_2-U_1)/(I_SC1-I_SC2)
print(prenos_chyb_latex(R_S))

def interpol(y, x0, y0, x1, y1):
    return (y-y0+(y1-y0)/(x1-x0)*x0)/((y1-y0)/(x1-x0))

print(round_to(R_S(0.4472676826868353, 0.5504186947956508, 0.094574, 0.048583), prenos_chyb_eval(R_S, [6.739015240302529e-05, 8.28628042193476e-05, 1.89148e-05, 9.71661e-06], (0.4472676826868353, 0.5504186947956508, 0.094574, 0.048583))))