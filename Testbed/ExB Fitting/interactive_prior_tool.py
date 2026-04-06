 
import numpy as np
import pandas as pd
import pymc as pm
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import pytensor.tensor as pt

 # Load and plot the raw data
df = pd.read_csv('SampleWienFilterData.txt')
x = df['# Velocity [m/s]']
y = df[' Current [A]']
#plt.plot(x, y, '.')

# Scale the data for numerical stability of MCMC
x_min, x_max = x.min(), x.max()
y_min, y_max = y.min(), y.max()

def minmax_scale(data, data_min, data_max):
    return (data - data_min) / (data_max - data_min)

def minmax_rescale(scaled_data, data_min, data_max):
    return scaled_data * (data_max - data_min) + data_min

# Scale
x = minmax_scale(x, x_min, x_max)
y = minmax_scale(y, y_min, y_max)


def triple_gaussian(x, A1, mu1, s1, A2, mu2, s2, A3, mu3, s3, C):
    return (A1 * np.exp(-0.5 * ((x-mu1)/s1)**2) +
            A2 * np.exp(-0.5 * ((x-mu2)/s2)**2) +
            A3 * np.exp(-0.5 * ((x-mu3)/s3)**2) + C)

fig, ax = plt.subplots(figsize=(12,10))
plt.subplots_adjust(bottom=0.42)
sc_points, = ax.plot(x, y, 'o', alpha=0.5, label="data", color='purple')
(line_tot,) = ax.plot([], [], 'b-', lw=2, label="total (sum)")
(line_g1,) = ax.plot([], [], 'r--', lw=2, label="Gaussian 1")
(line_g2,) = ax.plot([], [], 'orange', ls='--', lw=2, label="Gaussian 2")
(line_g3,) = ax.plot([], [], 'g--', lw=2, label="Gaussian 3")

ax.legend()
ax.set_ylim(np.min(y)-0.1*np.ptp(y), np.max(y)+0.1*np.ptp(y))
ax.set_xlim(np.min(x), np.max(x))
ax.set_xlabel('x')
ax.set_ylabel('y')

# Choose reasonable slider starting points and ranges for scaled data (0-1)
default_params = dict(
    A1=0.1, mu1=0.2, s1=0.08,
    A2=1.0, mu2=0.6, s2=0.09,
    A3=0.1, mu3=0.8, s3=0.10, C=0.0
)

slider_params = [
    #       [label,    min, max,     default, color]
    ['A1',   0.00, 0.3,    default_params['A1'], 'red'],
    ['mu1',  0.00, 0.4,    default_params['mu1'], 'red'],
    ['s1',   0.01, 0.20,   default_params['s1'], 'red'],
    ['A2',   0.02, 1.5,    default_params['A2'], 'orange'],
    ['mu2',  0.45, 0.80,   default_params['mu2'], 'orange'],
    ['s2',   0.01, 0.20,   default_params['s2'], 'orange'],
    ['A3',   0.00, 0.3,    default_params['A3'], 'green'],
    ['mu3',  0.70, 1.0,    default_params['mu3'], 'green'],
    ['s3',   0.01, 0.20,   default_params['s3'], 'green'],
    ['C',   -0.2, 0.2,     default_params['C'], "gray"]
]

sliders = []
slider_axes = []
for i, (label, valmin, valmax, valinit, col) in enumerate(slider_params):
    ax_slider = plt.axes([0.12, 0.39 - i*0.03, 0.73, 0.02], facecolor='lightgray')
    s = Slider(ax_slider, label, valmin, valmax, valinit=valinit, color=col)
    sliders.append(s)
    slider_axes.append(ax_slider)

def update(val=None):
    vals = [s.val for s in sliders]
    yy = triple_gaussian(x, *vals)
    g1 = vals[0] * np.exp(-0.5*((x-vals[1])/vals[2])**2)
    g2 = vals[3] * np.exp(-0.5*((x-vals[4])/vals[5])**2)
    g3 = vals[6] * np.exp(-0.5*((x-vals[7])/vals[8])**2)
    line_tot.set_data(x, yy)
    line_g1.set_data(x, g1 + vals[9])
    line_g2.set_data(x, g2 + vals[9])
    line_g3.set_data(x, g3 + vals[9])
    fig.canvas.draw_idle()

for s in sliders:
    s.on_changed(update)
update()

plt.show()