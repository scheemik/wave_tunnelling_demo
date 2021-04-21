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


fig,ax = plt.subplots()
ax.plot(x,P[:,0],'b-')
ax.plot(x,V(x),'r--')
plt.show()
