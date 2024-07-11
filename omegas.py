import numpy as np
from numpy import linalg as LA
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

# Parameter values
# Masses:
m = 10.0
# m2 = 2.0
# Spring constants
k1 = 2.0
k2 = 2.0
# w
w2 = 5.0

kc = 40


i = 0
w1 = np.linspace(0, 10, 1000)
# W = np.linspace(-100, 100, 1000)
W = np.zeros((4, 1000))

W[0, :] = w1
W[1 ,:] = w2
W[2 ,:] =(1/2 * (w1**2+w2**2) + (1/4* (w1**2-w2**2)**2+ kc**2/m**2)**(1/2))**(1/2)
W[3 ,:] =(1/2 * (w1**2+w2**2) - (1/4* (w1**2-w2**2)**2+ kc**2/m**2)**(1/2))**(1/2)



fig, ax = plt.subplots()
fig.subplots_adjust(bottom=0.25)
h0 = ax.plot(w1, W[0, :], "b")[0]
h1 = ax.plot(w1, W[1, :], "g")[0]
h2 = ax.plot(w1, W[2, :], "r")[0]
h3 = ax.plot(w1, W[3, :], "k")[0]

# Create subplot
# Import libraries

# Create axes for frequency and amplitude sliders
kc_ax = plt.axes([0.25, 0.15, 0.65, 0.03])
kc_val = Slider(kc_ax, 'kc', 0.0, 200.0, 100)


def update(val):
      
    kc = kc_val.val
        
    w1 = np.linspace(0, 10, 1000)
    W[0, :] = w1
    W[1 ,:] = w2
    W[2 ,:] =np.sqrt(1/2 * (w1**2+w2**2) + np.sqrt(1/4* (w1**2-w2**2)**2+ kc**2/m**2))
    W[3 ,:] =np.sqrt(1/2 * (w1**2+w2**2) - np.sqrt(1/4* (w1**2-w2**2)**2+ kc**2/m**2))

    h0.set_data(w1, W[0, :])
    h1.set_data(w1, W[1, :])
    h2.set_data(w1, W[2, :])
    h3.set_data(w1, W[3, :])

# Call update function when slider value is changed
kc_val.on_changed(update)

# display graph
h0.set_label("w1")
h1.set_label("w2")
h2.set_label("w+")
h3.set_label("w-")
ax.legend(loc='upper right')

plt.show()




# from pylab import figure, plot, xlabel, grid, legend, title, savefig
# from matplotlib.font_manager import FontProperties

# xlabel("w1")

# grid(True)
# # hold(True)
# lw = 1

# plot(w1, W[0, :], "b", linewidth=lw)
# plot(w1, W[1, :], "g", linewidth=lw)
# plot(w1, W[2, :], "r", linewidth=lw)
# plot(w1, W[3, :], "k", linewidth=lw)
# # plot(KC, Eig[2, :], "r", linewidth=lw)
# # plot(KC, Eig[3, :], "k", linewidth=lw)

# legend((r"$w_1$", r"$w_2$", r"$w+$", r"$w-$"), prop=FontProperties(size=16))
# title("avoided crossing")
# savefig("avoided_crossing.png", dpi=100)
# # print(Eig[:, 10])
