import numpy as np
import plotly.graph_objects as go 
from tsmoothie.smoother import *
import plotly.express as px

def getStartAndEnds(upOrDown, energy):
    i = 1

    Areas = []

    changing = False

    area = []
    while i < len(upOrDown):
        if (energy[i] > 400):
            if (upOrDown[i] and not changing):
                area.append(energy[i])
            elif (upOrDown[i] and changing and upOrDown[i]-upOrDown[i-1] < 20):
                continue
            elif(upOrDown[i] and changing and upOrDown[i]-upOrDown[i-1] > 20):
                area.append(energy[i-1])
                Areas.append(area)
                area = []
                area.append(energy[i])
            elif(not upOrDown[i]):
                changing = False
                if (len(area) >= 1):
                    Areas.append(area)
                area = []
        i += 1

    startEndSegment = []
    for item in Areas:
        startEnd = []
        startEnd.append(item[0])
        startEnd.append(item[-1])
        startEndSegment.append(startEnd)
    
    return(startEndSegment)

fileName = input("Enter the file name: ")
naidatafile = open(fileName)

naidataraw = [i.split(',') for i in naidatafile.read().split('\n')]

naidataraw.remove([""])

naidata = []

for fiveminuteinterval in naidataraw:
    naidata.append(fiveminuteinterval[6:]) 

for i in range(len(naidata)):
    for j in range(len(naidata[i])):
        naidata[i][j] = int(naidata[i][j])

energycalibration = lambda x: 3.87589476e-04 * x ** 2 + 3.59950887e+00 * x + 2.62548505e-13

dataset = 1

xvaluesinbins = np.array(range(45, 590))
energy = energycalibration(xvaluesinbins)

counts = np.array(naidata[dataset]) 
counts = np.delete(counts, [i for i in range(45)])
counts = counts[:-429]

smoother = ConvolutionSmoother(window_len=12, window_type='ones')
smoother.smooth(counts)

# generate intervals
low, up = smoother.get_intervals('sigma_interval', n_sigma=2)
smoothed = smoother.smooth_data[0]

up = [False] * len(smoothed)
down = [False] * len(smoothed)

peakLength = int(input("Input peak length: "))
window_size = int(input("Input window size: "))
estimatedSlope = float(input("Input a good number to determine what the slope of the baseline is: "))

background = np.convolve(smoothed, np.ones(window_size)/window_size, mode='same')
i = peakLength
while i < len(background):
    if ((background[i] - background[i-peakLength]) > 0.25):
        up[i] = True
    else:
        up[i] = False
    
    if (abs((-background[i] + background[i-peakLength])/(peakLength)) > estimatedSlope and (not up[i])):
        down[i] = True
    else:
        down[i] = False

    i += 1


up = getStartAndEnds(up, energy)
down = getStartAndEnds(down, energy)

peaks = []

while (len(up) > 0):        
    if(len(down) > 0):
        if (down[0][0] < up[0][0]):
            down.pop(0)

        elif (down[0][0] - up[0][1] <= 50):
            peaks.append([up[0][0], down[0][1]])
            down.pop(0)
            up.pop(0)
        else:
            upLength = up[0][1] - up[0][0]
            peaks.append([up[0][0], up[0][1]+upLength])
            up.pop(0)
    else:
        upLength = up[0][1] - up[0][0]
        peaks.append([up[0][0], up[0][1]+upLength])
        up.pop(0)

        
print("Peaks in the data: ")
print(peaks)
        
fig = go.Figure()

fig.add_trace(go.Scatter(x=energy, y=background))

fig.update_layout(
    title="Energy v background",
    font=dict(
        family="LEMON MILK",
        size=8,
        color="Black"
    ),
    plot_bgcolor='rgba(0,0,0,0)',
    template='plotly_white'
)
fig.update_yaxes(showgrid=True,
                    linecolor="black",
                    tickfont=dict(color="black", size=10),
                    title_font = {"size": 22, "color": "black"},
                    title_standoff = 15,
                    type="log"
)
fig.update_xaxes(linecolor="black",
                    tickfont=dict(color="black", size=10),
                    title_font = {"size": 22, "color": "black"},
                    title_standoff = 15
)
fig.show()
