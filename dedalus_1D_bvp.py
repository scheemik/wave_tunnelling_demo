"""
Dedalus script for solving a 1D differential equation.
Based off the example script for solving the Lane-Enden equation.

This is a 1D script and should be ran serially.  It should converge within
roughly a dozen iterations, and should take under a minute to run.

The equation is of the general form:
    dx(dx(P)) + P = 0
    P(z=0) = 0
    P(z=L) = P_0
where C^2 is some coefficient, and the equation is solved over the interval
z=[0,L], where L is the length of the domain. The equation is second order,
so it requires 2 boundary conditions.

"""

import time
import numpy as np
import matplotlib.pyplot as plt

from dedalus import public as de
from dedalus.extras import flow_tools

import logging
logger = logging.getLogger(__name__)


# Parameters
x_min = 0
L = 1
x_max = L
Nx = 32
ncc_cutoff = 1e-10
tolerance = 1e-5 #1e-10

E = 1
V_0 = 1.5*E

# Build domain
xbasis = de.Chebyshev('x', Nx, interval=(x_min, x_max))
domain = de.Domain([xbasis], np.float64)#np.complex128)

# Setup problem
problem = de.NLBVP(domain, variables=['P1', 'P1x', 'P2', 'P2x', 'P3', 'P3x'], ncc_cutoff=ncc_cutoff)
# Problem parameters
problem.parameters['V_0'] = V_0
problem.parameters['E'] = E

# Section 1
problem.add_equation("dx(P1x) = -E*P1")         # Schrodinger equation for V_0=0
problem.add_equation("P1x - dx(P1) = 0")        # defining first derivative
# Section 2
problem.add_equation("dx(P2x) = (V_0-E)*P2")    # Schrodinger equation
problem.add_equation("P2x - dx(P2) = 0")        # defining first derivative
# Section 3
problem.add_equation("dx(P3x) = -E*P3")         # Schrodinger equation for V_0=0
problem.add_equation("P3x - dx(P3) = 0")        # defining first derivative
# BCs: initial conditions at x=0
problem.add_bc("left(P1) = 1")
problem.add_bc("left(P1x) = 0")
# BCs: continuous across section 1 to section 2
problem.add_bc("right(P1) - left(P2) = 0")
problem.add_bc("right(P1x) - left(P2x) = 0")
# BCs: continuous across section 2 to section 3
problem.add_bc("right(P2) - left(P3) = 0")
problem.add_bc("right(P2x) - left(P3x) = 0")

# Setup initial guess
solver = problem.build_solver()
x = domain.grid(0)

# Section 1
P1 = solver.state['P1']
P1x = solver.state['P1x']
P1['g'] = np.cos(100*x)
P1.differentiate('x', out=P1x)
# Section 2
P2 = solver.state['P2']
P2x = solver.state['P2x']
P2['g'] = np.cos(100*x)#0*x+1
P2.differentiate('x', out=P2x)
# Section 2
P3 = solver.state['P3']
P3x = solver.state['P3x']
P3['g'] = np.cos(100*x)#0*x+1
P3.differentiate('x', out=P3x)

# Iterations
pert = solver.perturbations.data
pert.fill(1+tolerance)
start_time = time.time()
while np.sum(np.abs(pert)) > tolerance:
    solver.newton_iteration()
    logger.info('Perturbation norm: {}'.format(np.sum(np.abs(pert))))
end_time = time.time()

logger.info('-'*20)
logger.info('Iterations: {}'.format(solver.iteration))
logger.info('Run time: %.2f sec' %(end_time-start_time))

# Plotting the output
# print('len(x) =', len(x))
# print('len(P1) =', len(P1['g']))
# What Dedalus solved for
plt.plot(x, P1['g'].real, label=r'$\psi_1$')
plt.plot(x+L, P2['g'].real, label=r'$\psi_2$')
plt.plot(x+2*L, P2['g'].real, label=r'$\psi_3$')
plt.plot(x, P1x['g'], label=r'$\partial_x\psi_1$')
plt.plot(x+L, P1x['g'], label=r'$\partial_x\psi_2$')
plt.plot(x+2*L, P1x['g'], label=r'$\partial_x\psi_3$')
# plt.plot(x, x_masks[0], label=r'$x_1$')
# plt.plot(x, x_masks[1], label=r'$x_2$')
# plt.plot(x, x_masks[2], label=r'$x_3$')
# Analytical solutions
# plt.plot(x, 2*np.sin(x), '-.', label=r'$\psi_{ana}$')
# plt.plot(x, 2*np.cos(x), '-.', label=r'$\psi_{z, ana}$')

plt.legend()
plt.xlabel('x')
plt.ylabel('Amplitude')
plt.show()
