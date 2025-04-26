# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 17:36:39 2023

@author: Balázs local
"""

from praktikum.main_funcs import prenos_chyb, round_to
from praktikum.format_latex import prenos_chyb_latex, format_to_latex, default_table

import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
import pandas as pd

def lin(x, a):
    return a*x

def lin2(x, a, x0):
    return a*(x-x0)

def a(l, t):
    return l/t

def alpha(a, l_0):
    return a/l_0

print(format_to_latex(alpha))
print()
print(prenos_chyb_latex(alpha))
print()
print(prenos_chyb_latex(a))
print()
print(format_to_latex(a, ("\\Delta l", "\\Delta t")))




#%% tepelna roztaznost tuhych teles
from praktikum.main_funcs import default_plot, reset_counter, rand_plot_colors
reset_counter()
plt.close("all")

data = pd.read_excel('data.xls', 'List1')

def data_t(df, mat):
    t = np.array(df.iloc[:,0])
    i0 = int(np.where(t == mat)[0])
    i1 = 0
    for i, k in enumerate(t[i0+2:]):
        if k is np.nan:
            i1 = i
            break
    if i1 == 0:
        t = np.array(t[i0+2:], dtype=np.float64)
    else :
        t = np.array(t[i0+2:i0+2+i1], dtype=np.float64)
    return t

def data_l(df, mat):
    t = np.array(df.iloc[:,0])
    l = np.array(df.iloc[:,1])
    i0 = int(np.where(t == mat)[0])
    i1 = 0
    for i, k in enumerate(t[i0+2:]):
        if k is np.nan:
            i1 = i
            break
    if i1 == 0:
        l = np.array(l[i0+2:], dtype=np.float64)
    else :
        l = np.array(l[i0+2:i0+2+i1], dtype=np.float64)
    return l

t_med = data_t(data, 'Meď')
t_hlinik = data_t(data, "Hliník")
t_ocel = data_t(data, "Ocel")
t_mosadz = data_t(data, "Mosadz")
t_sklo = data_t(data, "Sklo")

l_med = data_l(data, 'Meď')
l_hlinik = data_l(data, "Hliník")
l_ocel = data_l(data, "Ocel")
l_mosadz = data_l(data, "Mosadz")
l_sklo = data_l(data, "Sklo")

l_0med = 59.7 #cm
l_0hlinik = 59.8
l_0ocel = 59.7
l_0mosadz = 59.7
l_0sklo = 59.9

chyba_l = 0.01/np.sqrt(3)
chyba_t = 1/np.sqrt(3)

table1 = pd.DataFrame(columns = ['$l_0$ [cm]'], index = ['meď','hliník','ocel','mosadz','sklo'])
table1['$l_0$ [cm]'] = [round_to(x, 0.1/np.sqrt(3)) for x in [l_0med,l_0hlinik,l_0ocel,l_0mosadz,l_0sklo]]
# table1 = table1.style.to_latex(clines="all;data",
#                           column_format = '|l|l|',
#                           label = 'table1',
#                           caption = '$l_0$ dĺžky tyčí')

#%% vypocty
covs = []
parms = []

for t, l in zip([t_med,t_hlinik,t_ocel,t_mosadz,t_sklo], [l_med,l_hlinik,l_ocel,l_mosadz,l_sklo]):
    parms0, cov0 = opt.curve_fit(lin, t-min(t), l)
    parms.append(parms0[0])
    covs.append(cov0[0][0])

colors = rand_plot_colors(5)
default_plot([t_med,t_hlinik,t_ocel,t_mosadz,t_sklo],
              [l_med,l_hlinik,l_ocel,l_mosadz,l_sklo], 't [°C]', 'Δ l [mm]',
              ['meď','hliník','ocel','mosadz','sklo'],
              #[chyba_t]*5, [chyba_l]*5, 
              save = True)

default_plot([t_med-min(t_med),t_hlinik-min(t_hlinik),t_ocel-min(t_ocel),t_mosadz-min(t_mosadz),t_sklo-min(t_sklo)],
             [l_med,l_hlinik,l_ocel,l_mosadz,l_sklo], 'Δ t [°C]', 'Δ l [mm]',
             ['meď','hliník','ocel','mosadz','sklo'],
             #[chyba_t]*5, [chyba_l]*5, 
             fit = [[lin, parms[0]], [lin, parms[1]], [lin, parms[2]], [lin, parms[3]], [lin, parms[4]]], colors = colors, save = True)

# table2 = pd.DataFrame(columns = ['$t \enspace[\degree C]$', '$\Delta t \enspace[\degree C]$', '$\Delta l$ [mm]','meď','hliník','ocel','mosadz','sklo'])
# table2['$t \enspace[\degree C]$'] = 
# table2 = default_table(table2, "table2", 'Namerené hodnoty, ktoré sú vykreslené do grafov')

std_a_2 = []
for t, l in zip([t_med,t_hlinik,t_ocel,t_mosadz,t_sklo], [l_med,l_hlinik,l_ocel,l_mosadz,l_sklo]):
    std_a = sum(prenos_chyb(a, np.diag([chyba_l**2, chyba_t**2]), [l[i], t[i]])**2 for i in range(1, len(t)))/(len(t)-1)
    std_a_2.append(std_a)

std_a = np.sqrt(np.array(std_a_2)+covs)

table2 = pd.DataFrame(columns = ['a [mm/°C]'], index = ['meď','hliník','ocel','mosadz','sklo'])
table2['a [mm/°C]'] = [round_to(parms[0], std_a[0]),
                       round_to(parms[1], std_a[1]),
                       round_to(parms[2], std_a[2]),
                       round_to(parms[3], std_a[3]),
                       round_to(parms[4], std_a[4])]
table2 = table2.style.to_latex(clines="all;data",
                          column_format = '|l|l|',
                          label = 'table2',
                          caption = '$a$ koeficienty fitovaných primok')

alphas = [(alpha(p, l*10)*10**6, prenos_chyb(alpha, np.diag([std**2, 1/3]), [p, l*10])*10**6) for p, l, std in zip(parms, [l_0med,l_0hlinik,l_0ocel,l_0mosadz,l_0sklo], std_a)]

table3 = pd.DataFrame(columns = ['\\alpha [1/°C]'], index = ['meď','hliník','ocel','mosadz','sklo'])
table3['\\alpha [1/°C]'] = [round_to(*i) for i in alphas]
table3 = table3.style.to_latex(clines="all;data",
                          column_format = '|l|l|',
                          label = 'table3',
                          caption = '$\\alpha$ koeficienty tepelných roťažností')

#%% korekce
t_med = np.append(t_med[1:6], t_med[9:])
l_med = np.append(l_med[1:6], l_med[9:])-l_med[1]

covs = []
parms = []

for t, l in zip([t_med,t_hlinik,t_ocel,t_mosadz,t_sklo], [l_med,l_hlinik,l_ocel,l_mosadz,l_sklo]):
    parms0, cov0 = opt.curve_fit(lin, t-min(t), l)
    parms.append(parms0[0])
    covs.append(cov0[0][0])

default_plot([t_med,t_hlinik,t_ocel,t_mosadz,t_sklo],
              [l_med,l_hlinik,l_ocel,l_mosadz,l_sklo], 't [°C]', 'Δ l [mm]',
              ['meď','hliník','ocel','mosadz','sklo'],
              #[chyba_t]*5, [chyba_l]*5, 
              save = True)

default_plot([t_med-min(t_med),t_hlinik-min(t_hlinik),t_ocel-min(t_ocel),t_mosadz-min(t_mosadz),t_sklo-min(t_sklo)],
             [l_med,l_hlinik,l_ocel,l_mosadz,l_sklo], 'Δ t [°C]', 'Δ l [mm]',
             ['meď','hliník','ocel','mosadz','sklo'],
             #[chyba_t]*5, [chyba_l]*5, 
             fit = [[lin, parms[0]], [lin, parms[1]], [lin, parms[2]], [lin, parms[3]], [lin, parms[4]]], colors = colors, save = True)

std_a_2 = []
for t, l in zip([t_med,t_hlinik,t_ocel,t_mosadz,t_sklo], [l_med,l_hlinik,l_ocel,l_mosadz,l_sklo]):
    std_a = sum(prenos_chyb(a, np.diag([chyba_l**2, chyba_t**2]), [l[i], t[i]])**2 for i in range(1, len(t)))/(len(t)-1)
    std_a_2.append(std_a)

std_a = np.sqrt(np.array(std_a_2)+covs)

table4 = pd.DataFrame(columns = ['a [mm/°C]', '\\alpha [$\frac{1}{{10^6} \degree C}$]'], index = ['meď','hliník','ocel','mosadz','sklo'])
table4['a [mm/°C]'] = [round_to(parms[0], std_a[0]),
                       round_to(parms[1], std_a[1]),
                       round_to(parms[2], std_a[2]),
                       round_to(parms[3], std_a[3]),
                       round_to(parms[4], std_a[4])]

alphas = [(alpha(p, l*10)*10**6, prenos_chyb(alpha, np.diag([std**2, 1/3]), [p, l*10])*10**6) for p, l, std in zip(parms, [l_0med,l_0hlinik,l_0ocel,l_0mosadz,l_0sklo], std_a)]
table4['\\alpha [$\frac{1}{{10^6} \degree C}$]'] = [round_to(*i) for i in alphas]
table4 = table4.style.to_latex(clines="all;data",
                          column_format = '|l|l|l|',
                          label = 'table4',
                          caption = '$a$ a $\\alpha$ korigované koeficienty')

#==============================================================================================================
#%% Moja otázka opravovateľovi, ktorý prečíta tento skript:
    # Nie som si istý, že ako sa dá dopočítať chyby fitovaných parametrov (v prípade ked máme z nich viac) 
        # ktoré prídu z niestoty merania. Kovariančná matica dáva len štatistické chyby a metódu prenášanie 
        # chýb sa dá používať len v prípade keď máme len jeden fitovaný parameter.
       # Ja by som to vyriešil tak, že používam numerický odhad pomocou metódy prenos chýb (funkcia prenos_chyb),
        # ako na funkcie s počtom premennéch rovnajúce sa s počtom zmeraných dát,
        # ale pre veľa dáta to je vľmi náročný výpočet (aj keby som optimalizoval moju funkciu).
    # Síce tento problém pri tejto experimente sa nevyskytol, ale v iných by sa to mohol vyskytnúť
#==============================================================================================================
