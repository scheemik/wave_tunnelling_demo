"""
Author: Mikhail Schee
Created: 2021-04-21

This script solves and plots the time-independent Schrodinger equation in 1D

-(hbar^2/2m)*(d^2 P/dx^2) + V(x)*P = E*P

for a given potential V(x). Assuming hbar^2/m = 1,

-(1/2)*(d^2 P/dx^2) + V(x)*P = E*P

"""

import numpy as np
import matplotlib.pyplot as plt

# Enable dark mode plotting
plt.style.use('dark_background')

from scipy.integrate import odeint
import numpy
import matplotlib.pyplot as plt

#Inputs
b=0.05
g=9.81
l=1
m=1

#initial condition
theta_0 = [0,3]

#time plot
t = np.linspace(0,20,240)

#Defining the differential equations

def system(theta,t,b,g,l,m):
	theta1 = theta[0]
	theta2 = theta[1]
	dtheta1_dt = theta2
	dtheta2_dt = -(b/m)*theta2-g*np.sin(theta1)
	dtheta_dt=[dtheta1_dt,dtheta2_dt]

	return dtheta_dt

#Solving ODE
theta = odeint(system,theta_0,t,args = (b,g,l,m))

fig,ax = plt.subplots()
ax.plot(t,theta[:,0],'b-')
ax.plot(t,theta[:,1],'r--')
plt.show()

###################################

#Inputs
hbar = 1
m=1
E=1
def V(x):
    return x**2

#initial condition
P_0 = [0.1,0.1]

# space plot
x = np.linspace(-1,1,100)

#Defining the differential equations

def system(Ps,x,m,hbar,V,E):
	P = Ps[0]
	dPdx = Ps[1]
	d2Pdx2 = (2*m/hbar)*(V(x)-E)*P
	return [dPdx, d2Pdx2]

# Solving ODE
P = odeint(system,P_0,x,args = (m,hbar,V,E))

fig,ax = plt.subplots()
ax.plot(x,P[:,0],'b-')
ax.plot(x,V(x),'r--')
plt.show()
