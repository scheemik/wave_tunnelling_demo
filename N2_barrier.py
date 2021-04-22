"""
Author: Mikhail Schee
Created: 2021-04-21

This plots the inviscid, non-rotational, boussinesq, streamfunction equation in 1D,

(N/omega)^2*(d^2 P/dz^2) + N^2(z)*P = omega^2*P

for a mixed layer barrier:

           N_0,     z < 0
    N(x) =  0,      0 < z < L
           N_0,     z > L

Define m^2 = (k^2/omega^2)*(1 - N^2/omega^2) and l^2 = k^2/omega^2 and assume
    N^2 > omega^2 so (1 - N^2\omega^2) < 0 to get this system of equations:

    0 = d^2_z P_1 + m^2 P_1
    0 = d^2_z P_2 - l^2 P_2
    0 = d^2_z P_3 + m^2 P_3

If the wave comes from low z, then we can write

    P_1 = A*exp(ikz) + B*exp(-ikz)
    P_2 = C*exp(-lz) + D*exp(lz)
    P_3 = F*exp(ikz)

Apply the 4 boundary conditions that P and dPdx must be continuous
    across z = 0,L and solve for the coefficients while imposing
    |A|^2 = 1 for simplicity

This demonstrates wave tunneling

"""

import numpy as np
import matplotlib.pyplot as plt

# Enable dark mode plotting
plt.style.use('dark_background')

N_omega = 4
omega = 0.7

# Domain
L = 0.001       # barrier width
kL = 1
k = kL/L
z_0 = 0
flank = 4*L
nz = 50
z  = np.linspace(-flank,flank+L, 5*nz)
z1 = np.linspace(-flank, 0, 2*nz)
z2 = np.linspace(0, L, nz)
z3 = np.linspace(L, flank+L, 2*nz)

m = np.sqrt(-(k**2/(omega**2))*(1 - N_omega**2))
l = np.sqrt(k**2/(omega**2))

# Coefficients
F = (2j*m*l*np.exp(-1j*m*L))/((m**2-l**2)*np.sinh(l*L) + 2j*m*l*np.cosh(l*L))
C = -(1j*m-l)*F*np.exp(1j*m*L + l*L)/(2*l)
D =  (1j*m+l)*F*np.exp(1j*m*L - l*L)/(2*l)
B = C + D - 1

def psi_1(x):
    return 1*np.exp(1j*m*x) + B*np.exp(-1j*m*x)
def psi_2(x):
    return C*np.exp(-l*x) + D*np.exp(l*x)
def psi_3(x):
    return F*np.exp(1j*m*x)

print("BC1-2:", psi_1(0)-psi_2(0))
print("BC2-3:", psi_2(L)-psi_3(L))

psi1=psi_1(z1)
psi2=psi_2(z2)
psi3=psi_3(z3)

def sq_amp(psi):
    # return (psi*np.conj(psi)).real
    return psi.real

fig,ax = plt.subplots()
ax.plot(z1, sq_amp(psi1), color='w')
ax.plot(z2, sq_amp(psi2), color='w')
ax.plot(z3, sq_amp(psi3), color='w')

ax1 = ax.twinx()
ax1.plot(z1, (z1*0+N_omega*omega))
ax1.plot(z3, (z3*0+N_omega*omega))
ax1.plot(z, (z*0+omega))
# ax1.set_ylim([0,1.3])
ax1.set_ylabel(r'Potential')

plt.show()
