"""
Author: Mikhail Schee
Created: 2021-04-21

This plots the time-independent Schrodinger equation in 1D, assuming hbar^2/m = 1

-(1/2)*(d^2 P/dx^2) + V(x)*P = E*P

for a rectangular potential:

            0,      x < 0
    V(x) = V_0,     0 < x < L
            0,      x > L

Define k^2 = 2mE/hbar^2 and l^2 = 2m(V_0-E)/hbar^2 and get this system of equations:

    0 = d^2_x P_1 + k^2 P_1
    0 = d^2_x P_2 - l^2 P_2
    0 = d^2_x P_3 + k^2 P_3

If the wave comes from low x, then we can write

    P_1 = A*exp(ikx) + B*exp(-ikx)
    P_2 = C*exp(-lx) + D*exp(lx)
    P_3 = F*exp(ikx)

Apply the 4 boundary conditions that P and dPdx must be continuous
    across x = 0,L and solve for the coefficients while imposing
    |A|^2 = 1 for simplicity

This demonstrates wave tunneling

"""

import numpy as np
import matplotlib.pyplot as plt

# Enable dark mode plotting
plt.style.use('dark_background')

x = np.linspace(-1,1,100)

E = 20
V_0 = 1.3*E

# Domain
L = 0.5       # barrier width
x_0 = 0
flank = 2*L
nx = 50
x1 = np.linspace(-flank, 0, 2*nx)
x2 = np.linspace(0, L, nx)
x3 = np.linspace(L, flank+L, 2*nx)

k = E
l = V_0 - E

# Coefficients
F = (2j*k*l*np.exp(-1j*k*L))/((k**2-l**2)*np.sinh(l*L) + 2j*k*l*np.cosh(l*L))
C = -(1j*k-l)*F*np.exp(1j*k*L + l*L)/(2*l)
D =  (1j*k+l)*F*np.exp(1j*k*L - l*L)/(2*l)
B = C + D - 1

def psi_1(x):
    return 1*np.exp(1j*k*x) + B*np.exp(-1j*k*x)
def psi_2(x):
    return C*np.exp(-l*x) + D*np.exp(l*x)
def psi_3(x):
    return F*np.exp(1j*k*x)

print("BC1-2:", psi_1(0)-psi_2(0))
print("BC2-3:", psi_2(L)-psi_3(L))

psi1=psi_1(x1)
psi2=psi_2(x2)
psi3=psi_3(x3)

def sq_amp(psi):
    # return (psi*np.conj(psi)).real
    return psi.real

fig,ax = plt.subplots()
plt.plot(x1, sq_amp(psi1), label=r'$\psi_1$')
plt.plot(x2, sq_amp(psi2), label=r'$\psi_2$')
plt.plot(x3, sq_amp(psi3), label=r'$\psi_3$')
plt.show()
