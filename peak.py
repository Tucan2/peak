import numpy as np
import plotly.graph_objects as go 
from processing import singleROI
import math
from tsmoothie.smoother import *

def appendToUp(up, i, upBefore, upBeforeTwice):
    if (upBeforeTwice and upBefore):
        if (i-2 not in up):
            up.append(i-2)
            upy.append(smoothed[i-2])
        if (i-1 not in up):
            up.append(i-1)
            upy.append(smoothed[i-1])
        if (i not in up):
            up.append(i)
            upy.append(smoothed[i])
    return(up)

def makeRoi(up):
    i = 1
    before = [up[0]]
    finalUp = []
    while i < len(up):
        if (before[-1] +1 != up[i]):
            if (len(before) >= 4):
                finalUp.append(before)
            before = []
        before.append(up[i])
        i += 1
    if (len(before) >= 4):
        finalUp.append(before)
    return(finalUp)

def straighten(roiy):
    i = 0
    while i < len(roiy):
        oneROIy = roiy[i]

        j = 1
        totalSlope = 0
        while j < len(roiy[i]):
            if (roi[i] not in up):
                totalSlope += oneROIy[j]-oneROIy[j-1]
            j += 1
        
        averageSlope = totalSlope/(len(oneROIy)-1)

        j = 0
        k = -len(roi[i])//2
        while j < len(roi[i]):
            oneROIy[j] -= averageSlope*(k)
            j += 1
            k += 1

        roiy[i] = oneROIy

        i += 1
    return roiy

#naidatafile = open("/Users/sciencegenuis2089/DataFiles/Data1655337600nai.csv", "r") 
#naidatafile = open("/Users/sciencegenuis2089/Downloads/etch_roof_d3s.csv", "r")
naidatafile = open("/Users/sciencegenuis2089/DataFiles/Data1655337600nai.csv", "r") 

naidataraw = [i.split(',') for i in naidatafile.read().split('\n')]

naidataraw.remove([""])

naidata = []

for fiveminuteinterval in naidataraw:
    naidata.append(fiveminuteinterval[6:])
    
for i in range(len(naidata)):
    for j in range(len(naidata[i])):
        naidata[i][j] = int(naidata[i][j])

energycalibration = lambda x: 3.87589476e-04 * x ** 2 + 3.59950887e+00 * x + 2.62548505e-13

xvaluesinbins = np.array(range(0, 1024))
xvaluesinenergy = energycalibration(xvaluesinbins)

dataset = 1 #1 of 141(?)

yvalues = np.array(naidata[dataset])

naiintegrated = sum(np.array(naidata))

xvalues = np.array(range(0, 1024))
integratedyvalues = naiintegrated


xvalsList = xvalues.tolist()


fig = go.Figure()

#fig.add_trace(go.Scatter(fillcolor = "RED", x=xvalues, y=integratedyvalues, name = "Integrated Counts"))

smoothed = np.log10(naiintegrated)
graphX = [4*x for x in xvalues]
fig.add_trace(go.Scatter(x=xvalues, y=smoothed, name = "Integrated Counts"))


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


i = 2
upBefore = False
upBeforeTwice = False
up = []
upy = []

while i < len(smoothed):
    upBefore = False
    upBeforeTwice = False
    if (i < 100):
        if (smoothed[i] >= smoothed[i-1] - 0.01):
            upBefore = True

        if (smoothed[i] >= smoothed[i-2] - 0.01):
            upBeforeTwice = True
            
        up = appendToUp(up, i, upBefore, upBeforeTwice)
        
    else:
        if (smoothed[i] >= smoothed[i-1]):
            upBefore = True

        if (smoothed[i] >= smoothed[i-2]):
            upBeforeTwice = True

        up = appendToUp(up, i, upBefore, upBeforeTwice)
    i += 1

roi = makeRoi(up)

i = 0
while i < len(roi):
    lastIndex = len(roi[i]) + roi[i][-1] + 25
    if (roi[i][0] < 100):
        lastIndex -= 15
    
    j = roi[i][-1]+1
    while j < lastIndex:
        roi[i].append(j)
        j += 1

    j = roi[i][0]-25
    if (roi[i][0] < 100):
        j += 15

    if (j < 0):
        j = 0

    while j < roi[i][0]:
        roi[i].insert(0, j)
        j += 1
    
    i += 1

roiy = []
yForGraph = []
xForGraph =[]
for potentialPeak in roi:
    temp = []
    for index in potentialPeak:
        temp.append(smoothed[index])
        xForGraph.append(index)
        yForGraph.append(smoothed[index])
    roiy.append(temp)


peaks = []
roi2 = []

fig.add_trace(go.Scatter(fillcolor = "YELLOW", x=roi, y=yForGraph, name = "Zoomssss", mode = "markers"))

i = 0
while i < 0:
    roiy = straighten(roiy)
    i += 1

i = 0
newROI = []
while i < len(roiy):
    oneROIy = roiy[i]

    #minimum = min(oneROIy) - 0.05
    #oneROIy = [x-minimum for x in oneROIy]

    singlePeak = singleROI(oneROIy)

    peaks.append(singlePeak)
    for item in oneROIy:
        newROI.append(item)

    i += 1



fig.add_trace(go.Scatter(fillcolor = "RED", x=xForGraph, y=newROI, name = "Things", mode = "markers"))

print(peaks)

peaksReal = []
peaksRealy = []
i = 0
while i < len(peaks):
    if (len(peaks[i]) != 0):
        j = peaks[i][0]
        while j <= peaks[i][1]:
            peaksReal.append(j)
            peaksRealy.append(smoothed[j])
            j += 1
    i += 1

print(peaksReal)

fig.add_trace(go.Scatter(fillcolor = "GREEN", x=peaksReal, y=peaksRealy, name = "Zooms", mode = "markers"))

fig.show()
#print(peaksReal)