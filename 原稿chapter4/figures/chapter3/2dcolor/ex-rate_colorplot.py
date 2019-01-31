#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 12:44:09 2019

@author: ryo
"""


import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


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

def fmt(x, pos):
    a, b = '{:.2e}'.format(x).split('e')
    b = int(b)
    return r'${} \times 10^{{{}}}$'.format(a, b)


"""
d = 1*10**(-3)
x = np.arange(0, 1000*10**(-3), d)
y = np.arange(0, 1000*10**(-3), d)
X, Y = np.meshgrid(x, y)
Z = excitation_rate_2color(5*10**12,1.6*10**9,2*10**9,X, Y)

levels = np.arange(0,600000,20000)
#levels = np.append(levels,20000)
fig, ax = plt.subplots(figsize=(8, 6))
contour = ax.contourf(X, Y, Z,levels)
CS = ax.contour(X, Y, Z, [10000], colors = ["w"])

ax.set_xlabel("760nm Power [W]")
ax.set_ylabel("894nm Power [W]")
ax.set_title("$f_{\mathrm{width}}=5$ THz, $f_{\mathrm{rep}}=1.6$ GHz, $\delta_{\mathrm{min}}=2$ GHz")
fig.colorbar(contour, format=ticker.FuncFormatter(fmt))
plt.savefig("5THz-16GHz-2GHz_1W.png")
"""
"""
delta = 1*10**(-3)
x = np.arange(0, 200*10**(-3), delta)
y = np.arange(0, 200*10**(-3), delta)
X, Y = np.meshgrid(x, y)
Z = excitation_rate_2color(5*10**12,1.6*10**9,2*10**9,X, Y)

levels = np.arange(0,30000,2000)
#levels = np.append(levels,20000)
fig, ax = plt.subplots(figsize=(8, 6))
contour = ax.contourf(X*1000, Y*1000, Z,levels)
CS = ax.contour(X*10**3, Y*10**3, Z, [10000], colors = ["w"])

ax.set_xlabel("760nm Power [mW]")
ax.set_ylabel("890nm Power [mW]")
ax.set_title("$f_{\mathrm{width}}=5$ THz, $f_{\mathrm{rep}}=1.6$ GHz, $\delta_{\mathrm{min}}=2$ GHz")
fig.colorbar(contour, format=ticker.FuncFormatter(fmt))
plt.savefig("5THz-16GHz-2GHz_200mW.png")
"""


d = 1*10**(-3)
x = np.arange(0, 1000*10**(-3), d)
y = np.arange(0, 100*10**(-3), d)
X, Y = np.meshgrid(x, y)
Z = excitation_rate_2color(5*10**12,120*10**6,0.4*10**9,X, Y)

levels = np.arange(0,95000,5000)
#levels = np.append(levels,20000)
fig, ax = plt.subplots(figsize=(8, 6))
contour = ax.contourf(X*1000, Y*1000, Z,levels)
CS = ax.contour(X*1000, Y*1000, Z, [10000], colors = ["w"])

ax.set_xlabel("760nm Power [W]")
ax.set_ylabel("894nm Power [W]")
ax.set_title("scattering rate [$\mathrm{s^{-1}}$]")
fig.colorbar(contour, format=ticker.FuncFormatter(fmt))
plt.savefig("5THz-120MHz-04GHz_new.png")

"""
delta = 1*10**(-3)
x = np.arange(0, 200*10**(-3), delta)
y = np.arange(0, 200*10**(-3), delta)
X, Y = np.meshgrid(x, y)
Z = excitation_rate_2color(5*10**12,120*10**6,0.4*10**9,X, Y)

levels = np.arange(0,36000,2000)
#levels = np.append(levels,20000)
fig, ax = plt.subplots(figsize=(8, 6))
contour = ax.contourf(X*1000, Y*1000, Z,levels)
CS = ax.contour(X*10**3, Y*10**3, Z, [10000], colors = ["w"])

ax.set_xlabel("760nm Power [mW]")
ax.set_ylabel("890nm Power [mW]")
ax.set_title("$f_{\mathrm{width}}=5$ THz, $f_{\mathrm{rep}}=120$ MHz, $\delta_{\mathrm{min}}=0.4$ GHz")
fig.colorbar(contour, format=ticker.FuncFormatter(fmt))
plt.savefig("5THz-120MHz-04GHz_200mW.png")

"""
"""
d = 1*10**(-3)
x = np.arange(0, 1000*10**(-3), d)
y = np.arange(0, 30*10**(-3), d)
X, Y = np.meshgrid(x, y)
Z = excitation_rate_2color(5*10**12, 1.6*10**9, 0.4*10**9,X, Y)

levels = np.arange(0,26000,2000)
#levels = np.append(levels,20000)
fig, ax = plt.subplots(figsize=(8, 6))
contour = ax.contourf(X*1000, Y*1000, Z,levels)
CS = ax.contour(X*1000, Y*1000, Z, [10000], colors = ["w"])

ax.set_xlabel("760nm Power [mW]")
ax.set_ylabel("894nm Power [mW]")
ax.set_title("scattering rate [$\mathrm{s^{-1}}$]")
fig.colorbar(contour, format=ticker.FuncFormatter(fmt))
plt.savefig("5THz-16GHz-04GHz_new.png")


d = 1*10**(-3)
x = np.arange(0, 200*10**(-3), d)
y = np.arange(0, 200*10**(-3), d)
X, Y = np.meshgrid(x, y)
Z = excitation_rate_2color(5*10**12, 1.6*10**9, 0.4*10**9,X, Y)

levels = np.arange(0,34000,2000)
#levels = np.append(levels,20000)
fig, ax = plt.subplots(figsize=(8, 6))
contour = ax.contourf(X*1000, Y*1000, Z,levels)
CS = ax.contour(X*1000, Y*1000, Z, [10000], colors = ["w"])

ax.set_xlabel("760nm Power [mW]")
ax.set_ylabel("894nm Power [mW]")
ax.set_title("$f_{\mathrm{width}}=5$ THz, $f_{\mathrm{rep}}=1.6$ GHz, $\delta_{\mathrm{min}}=0.4$ GHz")
fig.colorbar(contour, format=ticker.FuncFormatter(fmt))
plt.savefig("5THz-16GMHz-04GHz_200mW.png")


d = 1*10**(-3)
x = np.arange(0, 200*10**(-3), d)
y = np.arange(0, 200*10**(-3), d)
X, Y = np.meshgrid(x, y)
Z = excitation_rate_2color(5*10**12, 1.6*10**9, 0.3*10**9,X, Y)

levels = np.arange(0,36000,2000)
#levels = np.append(levels,20000)
fig, ax = plt.subplots(figsize=(8, 6))
contour = ax.contourf(X*1000, Y*1000, Z,levels)
CS = ax.contour(X*1000, Y*1000, Z, [10000], colors = ["w"])

ax.set_xlabel("760nm Power [mW]")
ax.set_ylabel("894nm Power [mW]")
ax.set_title("$f_{\mathrm{width}}=5$ THz, $f_{\mathrm{rep}}=1.6$ GHz, $\delta_{\mathrm{min}}=0.3$ GHz")
fig.colorbar(contour, format=ticker.FuncFormatter(fmt))
plt.savefig("5THz-16GMHz-03GHz_200mW.png")


d = 1*10**(-3)
x = np.arange(0, 200*10**(-3), d)
y = np.arange(0, 200*10**(-3), d)
X, Y = np.meshgrid(x, y)
Z = excitation_rate_2color(5*10**12, 1.6*10**9, 0.1*10**9,X, Y)

levels = np.arange(0,40000,2000)
#levels = np.append(levels,20000)
fig, ax = plt.subplots(figsize=(8, 6))
contour = ax.contourf(X*1000, Y*1000, Z,levels)
CS = ax.contour(X*1000, Y*1000, Z, [10000], colors = ["w"])

ax.set_xlabel("760nm Power [mW]")
ax.set_ylabel("894nm Power [mW]")
ax.set_title("$f_{\mathrm{width}}=5$ THz, $f_{\mathrm{rep}}=1.6$ GHz, $\delta_{\mathrm{min}}=0.1$ GHz")
fig.colorbar(contour, format=ticker.FuncFormatter(fmt))
plt.savefig("5THz-16GMHz-01GHz_200mW.png")


d = 1*10**(-3)
x = np.arange(0, 1000*10**(-3), d)
y = np.arange(0, 100*10**(-3), d)
X, Y = np.meshgrid(x, y)
Z = excitation_rate_2color(5*10**12, 1.6*10**9, 0.8*10**9,X, Y)

levels = np.arange(0,84000,4000)
#levels = np.append(levels,20000)
fig, ax = plt.subplots(figsize=(8, 6))
contour = ax.contourf(X*1000, Y*1000, Z,levels)
CS = ax.contour(X*1000, Y*1000, Z, [10000], colors = ["w"])

ax.set_xlabel("760nm Power [mW]")
ax.set_ylabel("894nm Power [mW]")
ax.set_title("scattering rate [$\mathrm{s^{-1}}$]")
fig.colorbar(contour, format=ticker.FuncFormatter(fmt))
plt.savefig("5THz-16GHz-08GHz_new.png")



d = 1*10**(-3)
x = np.arange(0, 1000*10**(-3), d)
y = np.arange(0, 1000*10**(-3), d)
X, Y = np.meshgrid(x, y)
Z = excitation_rate_2color(5*10**12, 1.6*10**9, 0.8*10**9,X, Y)

levels = np.arange(0,900000,100000)
#levels = np.append(levels,20000)
fig, ax = plt.subplots(figsize=(8, 6))
contour = ax.contourf(X, Y, Z,levels)
CS = ax.contour(X, Y, Z, [10000], colors = ["w"])

ax.set_xlabel("760nm Power [W]")
ax.set_ylabel("894nm Power [W]")
ax.set_title("$f_{\mathrm{width}}=5$ THz, $f_{\mathrm{rep}}=1.6$ GHz, $\delta_{\mathrm{min}}=0.8$ GHz")
fig.colorbar(contour, format=ticker.FuncFormatter(fmt))
plt.savefig("5THz-16GMHz-08GHz_1W.png")


d = 1*10**(-3)
x = np.arange(0, 1000*10**(-3), d)
y = np.arange(0, 1000*10**(-3), d)
X, Y = np.meshgrid(x, y)
Z = excitation_rate_2color(5*10**12, 1.6*10**9, 0.1*10**9,X, Y)

levels = np.arange(0,1000000,100000)
#levels = np.append(levels,20000)
fig, ax = plt.subplots(figsize=(8, 6))
contour = ax.contourf(X, Y, Z,levels)
CS = ax.contour(X, Y, Z, [10000], colors = ["w"])

ax.set_xlabel("760nm Power [W]")
ax.set_ylabel("894nm Power [W]")
ax.set_title("$f_{\mathrm{width}}=5$ THz, $f_{\mathrm{rep}}=1.6$ GHz, $\delta_{\mathrm{min}}=0.1$ GHz")
fig.colorbar(contour, format=ticker.FuncFormatter(fmt))
plt.savefig("5THz-16GMHz-01GHz_1W.png")

"""
"""
d = 1*10**(-3)
x = np.arange(0, 1000*10**(-3), d)
y = np.arange(0, 40*10**(-3), d)
X, Y = np.meshgrid(x, y)
Z = excitation_rate_2color(5*10**12,120*10**6, 0.2*10**9,X, Y)

levels = np.arange(0,44000,4000)
#levels = np.append(levels,20000)
fig, ax = plt.subplots(figsize=(8, 6))
contour = ax.contourf(X*1000, Y*1000, Z,levels)
CS = ax.contour(X*1000, Y*1000, Z, [10000], colors = ["w"])

ax.set_xlabel("760nm Power [mW]")
ax.set_ylabel("894nm Power [mW]")
ax.set_title("scattering rate [$\mathrm{s^{-1}}$]")
fig.colorbar(contour, format=ticker.FuncFormatter(fmt))
plt.savefig("5THz-120MHz-02GHz_new.png")


delta = 1*10**(-3)
x = np.arange(0, 200*10**(-3), delta)
y = np.arange(0, 200*10**(-3), delta)
X, Y = np.meshgrid(x, y)
Z = excitation_rate_2color(5*10**12,120*10**6,0.2*10**9,X, Y)

levels = np.arange(0,42000,2000)
#levels = np.append(levels,20000)
fig, ax = plt.subplots(figsize=(8, 6))
contour = ax.contourf(X*1000, Y*1000, Z,levels)
CS = ax.contour(X*10**3, Y*10**3, Z, [10000], colors = ["w"])

ax.set_xlabel("760nm Power [mW]")
ax.set_ylabel("890nm Power [mW]")
ax.set_title("$f_{\mathrm{width}}=5$ THz, $f_{\mathrm{rep}}=120$ MHz, $\delta_{\mathrm{min}}=0.2$ GHz")
fig.colorbar(contour, format=ticker.FuncFormatter(fmt))
plt.savefig("5THz-120MHz-02GHz_200mW.png")
"""






