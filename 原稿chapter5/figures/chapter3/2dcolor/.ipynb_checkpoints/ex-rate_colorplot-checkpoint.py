#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 12:44:09 2019

@author: ryo
"""


import math
import numpy as np
import matplotlib.pyplot as plt



def excitation_rate_2color(f_width,frep, rityou ,P_760, P_894):
    #物理定数など
    e0 = 8.854 * (10**-12)  # Fm**-1
    c = 2.998*(10**8) # m/s 
    h_ = 1.054*10**(-34) #J*s
    e = 1.602 * 10**(-19)
    bracket = 4.1*10**(-22)
    r = 2*math.pi*2.18*1000000

    n = f_width/frep
    delta = 0

    I_760 = P_760 /(((0.25*10**(-3))**2)*math.pi)
    I_894 = P_894 / (((0.25*10**(-3))**2)*math.pi)
    E_760 = (2*I_760/(n*c*e0))**0.5
    E_894 = (2*I_894/(n*c*e0))**0.5

    for i in range(int(n/2)):
        delta += ((f_width/2+rityou) + frep*i)**(-1) 
    for i in range(int(n/2)):
        delta += ((f_width/2+rityou) - frep*(i+1))**(-1)

    W = e**2 * E_760 * E_894 * bracket * delta * 0.5 / (h_**2)

    r_N = W**2/r

    return r_N


d = 1*10**(-3)
x = np.arange(0, 200*10**(-3), d)
y = np.arange(0, 200*10**(-3), d)
X, Y = np.meshgrid(x, y)
Z = excitation_rate_2color(5*10**12,2*10**9,1.6*10**9,X, Y)

levels = np.arange(0,34000,2000)
#levels = np.append(levels,20000)
fig, ax = plt.subplots(figsize=(8, 6))
contour = ax.contourf(X*1000, Y*1000, Z,levels)
CS = ax.contour(X*10**3, Y*10**3, Z, [10000], colors = ["w"])

ax.set_xlabel("760nm Power [mW]")
ax.set_ylabel("894nm Power [mW]")
ax.set_title("$f_{\mathrm{width}}=5$ THz, $f_{\mathrm{rep}}=1.6$ GHz, $\delta_{\mathrm{min}}=2$ GHz")
fig.colorbar(contour)
plt.savefig("5THz-1.6GHz-2GHz.png")