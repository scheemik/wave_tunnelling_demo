"""
Created on Sun Dec 28 12:02:59 2014

@author: Pero

1D Schrödinger Equation in a harmonic oscillator.

Program calculates bound states and energies for a quantum harmonic oscillator. It will find eigenvalues
in a given range of energies and plot wave function for each state.

For a given energy vector e, program will calculate 1D wave function using the Schrödinger equation
in the potential V(x). If the wave function diverges on x-axis, the
energy e represents an unstable state and will be discarded. If the wave function converges on x-axis,
energy e is taken as an eigenvalue of the Hamiltonian (i.e. it is alowed energy and wave function
represents allowed state).

Program uses differential equation solver &quot;odeint&quot; to calculate Sch. equation and optimization
tool &quot;brentq&quot; to find the root of the function. Both tools are included in the Scipy module.
The following functions are provided:

    - V(x) is a potential function of the HO. For a given x it returns the value of the potential
    - SE(psi, x) creates the state vector for a Schrödinger differential equation. Arguments are:
        psi - previous state of the wave function
        x - the x-axis
    - Wave_function(energy) calculates wave function using SE and &quot;odeint&quot;. It returns the wave-function
        at the value b far outside of the square well, so we can estimate convergence of the wave function.
    - find_all_zeroes(x,y) finds the x values where y(x) = 0 using &quot;brentq&quot; tool.

Vales of m and L are taken so that h-bar^2/m*L^2 is 1.

v2 adds feature of computational solution of analytical model from the usual textbooks. As a result,
energies computed by the program are printed and compared with those gained by the previous program.
"""
from pylab import *
from scipy.integrate import odeint
from scipy.optimize import brentq
#import matplotlib as plt

def V(x):
    """
    Potential function in the Harmonic oscillator. Returns V = 0.5 k x^2 if |x|&lt;L and 0.5*k*L^2 otherwise
    """
    if abs(x)< L:
        return 0.5*k*x**2
    else:
        return 0.5*k*L**2

def SE(psi, x):
    """
    Returns derivatives for the 1D schrodinger eq.
    Requires global value E to be set somewhere. State0 is first derivative of the
    wave function psi, and state1 is its second derivative.
    """
    state0 = psi[1]
    state1 = (2.0*m/h**2)*(V(x) - E)*psi[0]
    return array([state0, state1])

def Wave_function(energy):
    """
    Calculates wave function psi for the given value
    of energy E and returns value at point b
    """
    global psi
    global E
    E = energy
    psi = odeint(SE, psi_init, x)
    return psi[-1,0]

def find_all_zeroes(x,y):
    """
    Gives all zeroes in y = f(x)
    """
    all_zeroes = []
    s = sign(y)
    for i in range(len(y)-1):
        if s[i]+s[i+1] == 0:
            zero = brentq(Wave_function, x[i], x[i+1])
            all_zeroes.append(zero)
    return all_zeroes



def find_analytic_energies(en):
    """
    Calculates Energy values for the harmonic oscillator using analytical
    model (Griffiths, Introduction to Quantum Mechanics, page 35.)
    """
    E_max = max(en)
    print('Allowed energies of HO:')
    i = 0
    while((i+0.5)*h*w < E_max):
        print('%.2f'%((i+0.5)*h*w))
        i+=1



N = 1000                  # number of points to take on x-axis
psi = np.zeros([N,2])     # Wave function values and its derivative (psi and psi')
psi_init = array([.001,0])# Wave function initial states
E = 0.0                   # global variable Energy  needed for Sch.Eq, changed in function "Wave function"
b = 2                     # point outside of HO where we need to check if the function diverges
x = linspace(-b, b, N)    # x-axis
k = 100                   # spring constant
m = 1                     # mass of the body
w = sqrt(k/m)             # classical HO frequency
h = 1                     # normalized Planck constant
L = 1                     # size of the HO

def main():
    # main program

    en = linspace(0, 0.5*k*L**2, 50)   # vector of energies where we look for the stable states

    psi_end = []      # vector of wave function at x = b for all of the energies in en
    for e1 in en:
        psi_end.append(Wave_function(e1))     # for each energy e1 find the the psi(x) outside of HO

    E_zeroes = find_all_zeroes(en, psi_end)   # now find the energies where psi(b) = 0

    #Plot wave function values at b vs energy vector
    fig,ax = plt.subplots()
    ax.plot(en,psi_end)
    ax.set_title('Values of the $\Psi(b)$ vs. Energy')
    ax.set_xlabel('Energy, $E$')
    ax.set_ylabel('$\Psi(x = b)$', rotation='horizontal')
    for E in E_zeroes:
        ax.plot(E, [0], 'go')
        # annotate("E = %.2f" %E, xy = (E, 0), xytext=(E, 5))
    # grid()
    plt.show()

    # Print energies for the found states
    # print(&quot;Energies for the bound states are: &quot;)
    for En in E_zeroes:
        print("%.2f " %En)

    # Print energies of each bound state from the analytical model
    find_analytic_energies(en)

    # Plot the wave function for 1st 4 eigenstates
    fig,ax = plt.subplots()
    for i in range(4):                                                 # For each of 1st 4 allowed energies
        Wave_function(E_zeroes[i])                                     # find the wave function psi(x)
        ax.plot(x, 100**i*psi[:,0]**2, label="E = %.2f" %E_zeroes[i])      # and plot it scaled for comparison
    plt.legend(loc="upper right")
    ax.set_title('Wave function')
    ax.set_xlabel('x, $x/L$')
    ax.set_ylabel('$|\Psi(x)|^2$', rotation='horizontal', fontsize = 20)
    plt.show()

    # Plot the wave function for the last eigenstate
    figure(3)
    Wave_function(E_zeroes[-1])                                        # Find Wave function for the last allowed energy
    ax.plot(x, psi[:,0]**2, label="E = %.2f" %E_zeroes[-1])
    legend(loc="upper right")
    title('Wave function')
    xlabel('x, $x/L$')
    ylabel('$|\Psi(x)|^2$', rotation='horizontal', fontsize = 20)
    grid()

if __name__ == "__main__":
    main()
