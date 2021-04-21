"""
Dedalus script for solving a 1D differential equation.
Based off the example script for solving the Lane-Enden equation.

This is a 1D script and should be ran serially.  It should converge within
roughly a dozen iterations, and should take under a minute to run.

The equation is of the general form:
    dz(dz(P)) + P = 0
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
z_min = 0
z_max = np.pi/2.0
Nz = 128
ncc_cutoff = 1e-10
tolerance = 1e-5 #1e-10

# Build domain
z_basis = de.Chebyshev('z', Nz, interval=(z_min, z_max), dealias=1)#2)
domain = de.Domain([z_basis], np.float64)

# Setup problem
problem = de.NLBVP(domain, variables=['P', 'Pz'], ncc_cutoff=ncc_cutoff)
problem.add_equation("dz(Pz) = -P")#, tau=False)
problem.add_equation("Pz - dz(P) = 0")
problem.add_bc("left(P) = 0")
problem.add_bc("right(P) = 2")

# Setup initial guess
solver = problem.build_solver()
z = domain.grid(0)
P = solver.state['P']
Pz = solver.state['Pz']
P['g'] = np.cos(z)
P.differentiate('z', out=Pz)

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
print('len(z) =', len(z))
print('len(P) =', len(P['g']))
# What Dedalus solved for
plt.plot(z, P['g'], label=r'$\psi$')
plt.plot(z, Pz['g'], label=r'$\psi_z$')
# Analytical solutions
plt.plot(z, 2*np.sin(z), '-.', label=r'$\psi_{ana}$')
plt.plot(z, 2*np.cos(z), '-.', label=r'$\psi_{z, ana}$')

plt.legend()
plt.xlabel('z')
plt.ylabel('Amplitude')
plt.show()
