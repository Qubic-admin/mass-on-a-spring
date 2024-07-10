import numpy as np
from numpy import linalg as LA

# Parameter values
# Masses:
m1 = 2.0
m2 = 2.0
# Spring constants
k1 = 2.0
k2 = 5.0
# w
w1 = 2.0
w2 = 3.0


i = 0
KC = np.linspace(-10, 10, 1000)
# W = np.linspace(-100, 100, 1000)
Eig = np.zeros((2, 1000))
for kc in KC:
    f2 = np.array(
        [
            [
                k1 + kc - m1 * w1**2,
                -kc,
            ],
            [
                -kc,
                k2 + kc - m2 * w2**2,
            ],
        ]
    )
    eigenvalues, eigenvectors = LA.eig(f2)
    Eig[:, i] = eigenvalues
    i = i + 1

from numpy import loadtxt
from pylab import figure, plot, xlabel, grid, legend, title, savefig
from matplotlib.font_manager import FontProperties

xlabel("kc")

grid(True)
# hold(True)
lw = 1

plot(KC, Eig[0, :], "b", linewidth=lw)
plot(KC, Eig[1, :], "g", linewidth=lw)
# plot(KC, Eig[2, :], "r", linewidth=lw)
# plot(KC, Eig[3, :], "k", linewidth=lw)

legend((r"$E_1$", r"$E_2$"), prop=FontProperties(size=16))
title("eigenvalues vs coupling")
savefig("eigs.png", dpi=100)

print(Eig[:, 10])
