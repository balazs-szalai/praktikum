# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 14:27:59 2024

@author: balazs
"""

import numpy as np
import scipy.special as spec
import cv2
from math import sin
import praktikum as p
import matplotlib.pyplot as plt
import scipy.optimize as opt
from math import sqrt, cos, sin
import pandas as pd

data = pd.read_excel('mer8.xlsx')

table1 = p.default_table(pd.DataFrame({
    ' ': ['objektív 443431', 'objektív 468225', 'lupa z šošovky F100'],
    '$d_{zp}$ [mm]': p.readable([12, 2.1, 20], [1, 0.1, 1]),
    }), 'table1', 'caption')

x1 = np.array(data.iloc[3:15, 2], float)*10
sx1 = 1

x1_ = np.array(data.iloc[3:15, 3], float)
sx1_ = 2

x2 = np.array(data.iloc[3:10, 7], float)*10
sx2 = 1

x2_ = np.array(data.iloc[3:10, 8], float)
sx2_ = 0.5

x3 = np.array(data.iloc[2:7, 12], float)*10
sx3 = 2

x3_ = np.array(data.iloc[2:7, 13], float)
sx3_ = 1

table2 = p.default_table(pd.DataFrame({
    '$x$ [mm]': p.readable(x1, [sx1]*len(x1)),
    '$x\\prime$ [mm]': p.readable(x1_, [sx1_]*len(x1)),
    '$x$ [mm] ' : p.pad(p.readable(x1, [sx2]*len(x2)), len(x1)),
    '$x\\prime$ [mm] ': p.pad(p.readable(x2_, [sx2_]*len(x2)), len(x1)),
    '$x$ [mm]  ': p.pad(p.readable(x3, [sx3]*len(x3)), len(x1)),
    '$x\\prime$ [mm]  ': p.pad(p.readable(x3_, [sx3_]*len(x3)), len(x1))
    }), 'table2', 'caption', 
    header=[(2, 'objektív 443431'), (2, 'objektív 468225'), (2, 'lupa z šošovky F100')])

#%% zvecs
lin = lambda x, a, b: a*x + b

a1, b1 = p.lin_fit(x1_, x1, err = [[sx1_]*len(x1), [sx1]*len(x1)])
a1 = (3, 1)

a2, b2 = p.lin_fit(x2_, x2, err = [[sx2_]*len(x2), [sx1]*len(x2)])
a3, b3 = p.lin_fit(x3_, x3, err = [[sx3_]*len(x3), [sx3]*len(x3)])

p.default_plot([x1_, x2_, x3_], [x1, x2, x3], '$x^\prime$ [mm]', '$x$ [mm]',
               legend = ['objektív 443431', 'objektív 468225', 'lupa z šošovky F100'],
               xerror = [[sx1_]*len(x1), [sx2_]*len(x2), [sx3_]*len(x3)],
               yerror = [[sx1]*len(x1), [sx2]*len(x2), [sx3]*len(x3)],
               fit = [[lin, a1[0], b1[0]], [lin, a2[0], b2[0]], [lin, a3[0], b3[0]]])
beta_ob = [a1[0], a2[0], a2[0], a1[0]]
sbeta_ob = [a1[1], a2[1], a2[1], a1[1]]
#%% mikr

x1 = np.array(data.iloc[21:32, 2], float)*10
sx1 = 1

x1_ = np.array(data.iloc[21:32, 3], float)
sx1_ = 0.1 

x2 = np.array(data.iloc[21:28, 6], float)*10
sx2 = 1

x2_ = np.array(data.iloc[21:28, 7], float)
sx2_ = 0.1 

x3 = np.array(data.iloc[21:27, 10], float)*10
sx3 = 1

x3_ = np.array(data.iloc[21:27, 11], float)
sx3_ = 0.1 

x4 = np.array(data.iloc[21:32, 14], float)*10
sx4 = 10

x4_ = np.array(data.iloc[21:32, 15], float)
sx4_ = 0.1

table3 = p.default_table(pd.DataFrame({
    '$x$ [mm]': p.readable(x1, [sx1]*len(x1)),
    '$x\\prime$ [mm]': p.readable(x1_, [sx1_]*len(x1)),
    '$x$ [mm] ' : p.pad(p.readable(x1, [sx2]*len(x2)), len(x1)),
    '$x\\prime$ [mm] ': p.pad(p.readable(x2_, [sx2_]*len(x2)), len(x1)),
    '$x$ [mm]  ': p.pad(p.readable(x3, [sx3]*len(x3)), len(x1)),
    '$x\\prime$ [mm]  ': p.pad(p.readable(x3_, [sx3_]*len(x3)), len(x1)),
    '$x$ [mm]   ': p.readable(x4, [sx4]*len(x4)),
    '$x\\prime$ [mm]   ': p.readable(x4_, [sx4_]*len(x4))
    }), 'table3', 'caption', 
    header=[(2, 'objektív 443431, okulár 6x'), (2, 'objektív 468225, okulár 6x'), 
            (2, 'objektív 468225, okulár 10x'), (2, 'objektív 443431, okulár 10x')])

#%%

# a1, b1 = p.lin_fit(x1_, x1, err = [[sx1_]*len(x1), [sx1]*len(x1)])
# a2, b2 = p.lin_fit(x2_, x2, err = [[sx2_]*len(x2), [sx2]*len(x2)])
# a3, b3 = p.lin_fit(x3_, x3, err = [[sx3_]*len(x3), [sx3]*len(x3)])
a4, b4 = p.lin_fit(x4_, x4, err = [[sx4_]*len(x4), [sx4]*len(x4)])

# p.default_plot([x1_, x2_, x3_, x4_], [x1, x2, x3, x4], '$x^\prime$ [mm]', '$x$ [mm]',
#                legend = ['objektív 443431, okulár 6x', 'objektív 468225, okulár 6x', 
#                          'objektív 468225, okulár 10x', 'objektív 443431, okulár 10x'],
#                xerror = [[sx1_]*len(x1), [sx2_]*len(x2), [sx3_]*len(x3), [sx4_]*len(x4)],
#                yerror = [[sx1]*len(x1), [sx2]*len(x2), [sx3]*len(x3), [sx4]*len(x4)],
#                fit = [[lin, a1[0], b1[0]], [lin, a2[0], b2[0]], 
#                       [lin, a3[0], b3[0]], [lin, a4[0], b4[0]]])

#%%
a = np.array([a1, a2, a3, a4])
dzp = np.array([7.2, 1, 0.8, 5.7])
sdzp = 0.1

beta_ob = np.array(beta_ob)
sbeta_ob = np.array(sbeta_ob)

def d_Z(beta_ob, d_zp):
    return beta_ob*d_zp

dZ = beta_ob*dzp
sdZ = p.prenos_chyb_multi(d_Z, [sbeta_ob, [sdzp]*len(dzp)], [beta_ob, dzp])

table4 = p.default_table(pd.DataFrame({
    'konfigurácia mikroskopu': ['objektív 443431, okulár 6x', 'objektív 468225, okulár 6x', 
                                 'objektív 468225, okulár 10x', 'objektív 443431, okulár 10x'],
    'Z [-]': p.readable(a[:, 0], a[:, 1]),
    '$d_zp$ [mm]': p.readable(dzp, [sdzp]*len(dzp)),
    '$d_Z$ [mm]': p.readable(dZ, sdZ)
    }), 'table4', 'caption')
#%%

lamb = np.array(data.iloc[35:, 0], float)
alph = np.array(data.iloc[35:, 1:4], float)

table5 = p.default_table(pd.DataFrame({
    '': lamb.astype(int),
    ' ': p.readable(alph[:, 0], [2]*5),
    '  ': p.readable(alph[:, 1], [2]*5),
    '   ': p.readable(alph[:, 2], [2]*5)
    }), 'table5', 'caption', header=[(1, '$\\lambda$ [nm]'), (3, '$\\alpha$ [$\degree$]')])

salph = np.sqrt(np.std(alph, axis = 1)**2 + 2**2/5)
alpha = np.mean(alph, axis = 1)
#%%

k, b = p.lin_fit(lamb, alpha, err = [[0]*len(lamb), salph])
#%%
p.default_plot(lamb, alpha, '$\lambda$ [nm]', '$\\alpha$ [$\degree$]',
               xerror = [[0]*5],
               yerror = [salph],
               fit = [[lin, k[0], b[0]]])


