import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
from numpy import linalg as LA


################FUNCTIONS
def vectorfield(w, t, p):
    """
    Defines the differential equations for the coupled spring-mass system.

    Arguments:
        w :  vector of the state variables:
                  w = [x1,y1,x2,y2]
        t :  time
        p :  vector of the parameters:
                  p = [m1,m2,k1,k2,L1,L2,b1,b2]
    """
    x1, y1, x2, y2 = w
    # m1, m2, k1, kc, k2, v1, v2, om1, om2 = p
    m1, m2, k1, kc, k2 = p
    # Create f = (x1',y1',x2',y2'):

    f2 = [
        y1,
        (-k1 - kc) * x1 / m1 + (kc * x2 / m1),
        y2,
        (kc * x1 / m2) + (-k2 - kc) * x2 / m2,
    ]

    return f2


# Use ODEINT to solve the differential equations defined by the vector field
from scipy.integrate import odeint

# Parameter values
# Masses:
m1 = 1.5
m2 = 1.5
# Spring constants
k1 = 2.0
k2 = 2.0
# # res freq
# om1 = 2.0
# om2 = 2.0
# kc
kc = 3.0


# Initial conditions
# x1 and x2 are the initial displacements; y1 and y2 are the initial velocities
x1 = 1.0
y1 = 0.0
x2 = 0.0
y2 = 0.0

# ODE solver parameters
abserr = 1.0e-8
relerr = 1.0e-6
stoptime = 10.0
numpoints = 250

# Create the time samples for the output of the ODE solver.
# I use a large number of points, only because I want to make
# a plot of the solution that looks nice.
t = [stoptime * float(i) / (numpoints - 1) for i in range(numpoints)]

# Pack up the parameters and initial conditions:
# p = [m1, m2, k1, k2, L1, L2, b1, b2]
p = [
    m1,
    m2,
    k1,
    kc,
    k2,
]

w0 = [x1, y1, x2, y2]

# Call the ODE solver.
wsol = odeint(vectorfield, w0, t, args=(p,), atol=abserr, rtol=relerr)

with open("2s_nondamped.dat", "w") as f:
    # Print & save the solution.
    for t1, w1 in zip(t, wsol):
        # print(t1, w1[0], w1[1], file=f)
        print(t1, w1[0], w1[1], w1[2], w1[3], file=f)


# Plot the solution that was generated

from numpy import loadtxt
from pylab import figure, plot, xlabel, grid, legend, title, savefig
from matplotlib.font_manager import FontProperties

t, x1, y1, x2, y2 = loadtxt("2s_nondamped.dat", unpack=True)


figure(1, figsize=(6, 4.5))

xlabel("t")
grid(True)
# hold(True)
lw = 1

plot(t, x1, "b", linewidth=lw)
plot(t, x2, "g", linewidth=lw)

legend((r"$x_1$", r"$x_2$"), prop=FontProperties(size=16))
title("Mass Displacements for the\nCoupled Spring-Mass System")
savefig("2s_nondamped.png", dpi=100)


# from numpy import linalg as LA

# f1 = np.array([[-m1 * om1**2 + k1 + kc, -kc], [-kc, -m2 * om2**2 + k2 + kc]])
# f2 = np.array(
#     [
#         [0, 1, 0, 0],
#         [-m1 * om1**2 + k1 + kc, 0, -kc, 0],
#         [0, 0, 0, 1],
#         [-kc, 0, -m2 * om2**2 + k2 + kc, 0],
#     ]
# )
# eigenvalues, eigenvectors = LA.eig(f1)
# print(eigenvalues)
