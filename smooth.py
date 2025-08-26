import numpy as np
import matplotlib.pyplot as plt
import math
import plotly.graph_objects as go 



# generate 3 randomwalks of lenght 200
naidatafile = open("/Users/sciencegenuis2089/DataFiles/Data1655337600nai.csv", "r") 
naidataraw = [i.split(',') for i in naidatafile.read().split('\n')]

naidataraw.remove([""])

naidata = []

for fiveminuteinterval in naidataraw:
    naidata.append(fiveminuteinterval[6:])
    
for i in range(len(naidata)):
    for j in range(len(naidata[i])):
        if (naidata[i][j] != "error_flag"):
            naidata[i][j] = int(naidata[i][j])

energycalibration = lambda x: 3.87589476e-04 * x ** 2 + 3.59950887e+00 * x + 2.62548505e-13

xvaluesinbins = np.array(range(0, 1025))
xvaluesinenergy = energycalibration(xvaluesinbins)

dataset = 1 #1 of 141(?)

yvalues = np.array(naidata[dataset])


naiintegrated = np.zeros(1024)
for lst in naidata:
    for n in range(len(lst)-1):
        naiintegrated[n] += lst[n]

xvalues = np.array(range(0, 1024))

integratedyvalues = np.empty(0)
for item in naiintegrated:
    integratedyvalues = np.append(integratedyvalues, np.log10(item))

import sys
np.set_printoptions(threshold=sys.maxsize)

i = 0 
while i < len(integratedyvalues):
    if ((type(integratedyvalues[i]) != int and type(integratedyvalues[i]) != float and type(integratedyvalues[i]) != np.float64) or math.isinf(integratedyvalues[i])):
        integratedyvalues = np.delete(integratedyvalues, i)
        i -= 1
    i += 1
print(integratedyvalues)


print("Lowess Smoother")

from tsmoothie.utils_func import sim_randomwalk
from tsmoothie.smoother import LowessSmoother
from tsmoothie.smoother import GaussianSmoother
from tsmoothie.smoother import ConvolutionSmoother

# operate smoothing
#smoother = LowessSmoother(smooth_fraction=0.02, iterations=1)
#smoother = ConvolutionSmoother(5, 'blackman')
smoother = GaussianSmoother(sigma=1, n_knots=102)
data = integratedyvalues
smoother.smooth(data)

# generate intervals
#low, up = smoother.get_intervals('prediction_interval')
low, up = smoother.get_intervals('sigma_interval')
 
fig = go.Figure()



fig.update_layout(
    title="Integrated Gamma Spectra",
    xaxis_title="Energy [kev]", 
    yaxis_title="Counts",
    font=dict(
        family="LEMON MILK",
        size=18,
        color="Black"
    ),
    plot_bgcolor='rgba(0,0,0,0)',
    template='plotly_white'
)
fig.update_yaxes(showgrid=True,
                 gridcolor="lightgrey",
                 linecolor="black",
                 tickcolor="black",
                 tickfont=dict(color="black", size=10),
                 title_font = {"size": 22, "color": "black"},
                 title_standoff = 15,
)
fig.update_xaxes(linecolor="black",
                 tickfont=dict(color="black", size=10),
                 title_font = {"size": 22, "color": "black"},
                 title_standoff = 15
)

fig.add_trace(go.Scatter(x=xvalues, y=smoother.smooth_data[0], name = "Integrated Counts"))
fig.add_trace(go.Scatter(x=xvalues, y=smoother.data[0], name = "s Counts"))
fig.show()
"""
print("Spectral Smoother")

# operate smoothing
smoother = SpectralSmoother(smooth_fraction=0.1, pad_len=3)
data = integratedyvalues
smoother.smooth(data)

# generate intervals
low, up = smoother.get_intervals('sigma_interval')

# plot the smoothed timeseries with intervals
plt.figure(figsize=(18,5))

for i in range(1):
    
    plt.subplot(1,3,i+1)
    plt.plot(smoother.smooth_data[i], linewidth=3, color='blue')
    plt.plot(smoother.data[i], '.k')
    plt.title(f"timeseries {i+1}"); plt.xlabel('time')

    plt.fill_between(range(len(smoother.data[i])), low[i], up[i], alpha=0.3)

plt.show()

print("Gaussian Smoother")

# operate smoothing
smoother = GaussianSmoother()
data = data1
smoother.smooth(data)

# generate intervals
low, up = smoother.get_intervals('prediction_interval')

# plot the smoothed timeseries with intervals
plt.figure(figsize=(18,5))

for i in range(3):
    
    plt.subplot(1,3,i+1)
    plt.plot(smoother.smooth_data[i], linewidth=3, color='blue')
    plt.plot(smoother.data[i], '.k')
    plt.title(f"timeseries {i+1}"); plt.xlabel('time')

    plt.fill_between(range(len(smoother.data[i])), low[i], up[i], alpha=0.3)


print("Convolution Smoother")

# operate smoothing
smoother = ConvolutionSmoother()
data = data1
smoother.smooth(data)

# generate intervals
low, up = smoother.get_intervals('prediction_interval')

# plot the smoothed timeseries with intervals
plt.figure(figsize=(18,5))

for i in range(3):
    
    plt.subplot(1,3,i+1)
    plt.plot(smoother.smooth_data[i], linewidth=3, color='blue')
    plt.plot(smoother.data[i], '.k')
    plt.title(f"timeseries {i+1}"); plt.xlabel('time')

    plt.fill_between(range(len(smoother.data[i])), low[i], up[i], alpha=0.3)
"""