import numpy as np
import plotly.graph_objects as go 
from processing import singleROI
import math
from tsmoothie.smoother import *
from scipy.signal import detrend
import sys


slope = float(input("What is the slope?"))

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
    print(up)
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

def straighten(temp, xvals):
    from tsmoothie.smoother import GaussianSmoother

    # operate smoothing
    #smoother = LowessSmoother(smooth_fraction=0.02, iterations=1)
    #smoother = ConvolutionSmoother(5, 'blackman')
    smoother = GaussianSmoother(sigma=1, n_knots=4)
    data = temp
    smoother.smooth(data)

    # generate intervals
    #low, up = smoother.get_intervals('prediction_interval')
    low, up = smoother.get_intervals('sigma_interval')

    data = smoother.smooth_data[0]

    """
    averageSlope = 0
    i = 1
    while i < len(data):
        averageSlope += data[i] - data[i-1]
        i += 1

    averageSlope /= len(data) - 1
    """

    slope = 0
    up = []

    i = 1
    while i < len(data):
        up.append(data[i] - data[i-1] > 0)
        i += 1

    i = 0
    index1 = -1
    index2 = -1
    while i < len(up):
        if (up[i] and index1 == -1):
            index1 = i - 1
            break

        i += 1

    i = len(up)-1
    while i >= index1:
        if (not up[i] and index2 == -1):
            index2 = i -1
            break
        
        i -= 1
    
    if (index2 == -1):
        index2 = len(data)-1

    slope = (data[index2]-data[index1])/(index2-index1)

    i = 0
    k = -len(temp)//2
    while i < len(temp):
        if (type(temp[i]) == int or type(temp[i]) == float or type(temp[i]) == np.float64):
            temp[i] -= slope*(k)
        i += 1
        k += 1

    #fig.add_trace(go.Scatter(x=xvals, y=data, name = "Zoomasldfkj"))
    #fig.add_trace(go.Scatter(x=xvals, y=temp, name = "ADIUFOIUW"))
    #fig.add_trace(go.Scatter(x=[index1 + xvals[0], index2 + xvals[0]], y=[ temp[index1],  temp[index2]], name="LDELELELELEL"))


    return temp


naidatafile = open("/Users/sciencegenuis2089/DataFiles/ahs_os_d3s.csv", "r") 
#naidatafile = open("/Users/sciencegenuis2089/DataFiles/etch_roof_d3s.csv", "r")
#naidatafile = open("/Users/sciencegenuis2089/DataFiles/Data1655337600nai.csv", "r") 

naidataraw = [i.split(',') for i in naidatafile.read().split('\n')]
#print(naidataraw[0])

naidataraw.remove([""])

naidata = []

for fiveminuteinterval in naidataraw:
    naidata.append(fiveminuteinterval[6:])
    
energycalibration = lambda x: 3.87589476e-04 * x ** 2 + 3.59950887e+00 * x + 2.62548505e-13

xvaluesinbins = np.array(range(0, 1024))
xvaluesinenergy = energycalibration(xvaluesinbins)

dataset = 76 #1 of 141(?)

yvalues = np.array(naidata[dataset])

naiintegrated = [0] * 1024
for lst in naidata:
    for n in range(len(lst)-1):
        naiintegrated[n] += int(lst[n])

xvalues = np.array(range(0, 1023))
integratedyvalues = naiintegrated

xvalsList = xvalues.tolist()

fig = go.Figure()

#fig.add_trace(go.Scatter(fillcolor = "RED", x=xvalues, y=integratedyvalues, name = "Integrated Counts"))

index = 0
maxi = naiintegrated[0]
i = 0
while i < len(naiintegrated)-2:
    if (naiintegrated[i] > maxi):
        maxi = naiintegrated[i]
        index = i
    i += 1

print(naiintegrated)
print("SDLFJSDLKJ")
print(maxi)


xvalsList = xvalsList[index: len(xvalsList)]

print(index)
for i in range(index):
    naiintegrated = np.delete(naiintegrated, 0)


smoothed = np.log1p(naiintegrated)
smoothed = np.log1p(smoothed)

#smoothed = np.log1p(smoothed)

#smoothed = np.log1p(smoothed)
print(smoothed)

#smoothed = detrend(smoothed, type='linear')
print()

print(smoothed)
print("SDFLJSLFJ")


#smoothed = np.log1p(smoothed)

#smoothed = np.log1p(smoothed)

#smoothed = np.log(naiintegrated)
np.set_printoptions(threshold=sys.maxsize)

fig.add_trace(go.Scatter(x=xvalues, y=smoothed, name = "Integrated Counts"))

#smoother = GaussianSmoother(sigma=0.075, n_knots=50)
smoother = LowessSmoother(smooth_fraction=0.02, iterations=2)
#smoother = ConvolutionSmoother(8, 'blackman')
smoother.smooth(smoothed)

# generate intervals
#low, up = smoother.get_intervals('prediction_interval')
low, up = smoother.get_intervals('sigma_interval')

smoothed = smoother.smooth_data[0]


fig.add_trace(go.Scatter(x=xvalues, y=smoothed, name = "Smoothed Counts"))

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

while i < len(smoothed)-1:
    upBefore = False
    upBeforeTwice = False
    if (i < 150):
        upBefore = smoothed[i] >= smoothed[i-1] - 0.01

        upBeforeTwice = smoothed[i] >= smoothed[i-2] - 0.01

        up = appendToUp(up, i, upBefore, upBeforeTwice)
        
    else:
        upBefore =  smoothed[i] >= smoothed[i-1]
            

        upBeforeTwice = smoothed[i] >= smoothed[i-2]

        up = appendToUp(up, i, upBefore, upBeforeTwice)
    i += 1

roi = makeRoi(up)

i = 0
while i < len(roi)-5:
    lastIndex = len(roi[i]) + roi[i][-1] + 25
    
    j = roi[i][-1]+1
    while j < lastIndex:
        roi[i].append(j)
        j += 1

    firstIndex = roi[i][0]-25

    if (firstIndex < 0):
        firstIndex = 0

    
    j = roi[i][0]-1
    while j >= firstIndex:
        roi[i].insert(0, j)
        j -= 1
    
    i += 1

roiy = []
yForGraph = []
for potentialPeak in roi:
    temp = []
    for index in potentialPeak:
        temp.append(smoothed[index])
        yForGraph.append(smoothed[index])
    roiy.append(temp)


peaks = []
roi2 = []

#fig.add_trace(go.Scatter(fillcolor = "YELLOW", x=roi, y=yForGraph, name = "Zoomssss", mode = "markers"))


for item in roiy:
    i = 0
    while i < len(item):
        if ((type(item[i]) != int and type(item[i]) != float and type(item[i]) != np.float64) or math.isinf(item[i])):
            item.pop(i)
            i -= 1
        i += 1

i = 0
while i < len(roiy):
    temp = roiy[i]

    minimum = temp[0]
    j =0
    while j < len(temp):
        minimum = min(temp[j], minimum)
        j += 1

    toSubtract = minimum - 0.1
    j=0
    while j < len(temp):
        temp[j] = temp[j]- toSubtract
        temp[j] *= 30
        j += 1

    if (563 in roi[i]):
        #temp = straighten(temp, roi[i])
        #temp = straighten(temp, roi[i])
        #temp = straighten(temp, roi[i])
        #print("SDKFHSDKFHKSD")
        #print(temp)
        for item in temp:
            roi2.append(item)
        temp2, label = singleROI(temp)
        #peaks.append(temp2)
        #print(temp2)
        print(label)
        #print()
    else:
        #temp = straighten(temp, roi[i])
        #temp = straighten(temp, roi[i])
        #temp = straighten(temp, roi[i])
        #item = detrend(item, type='linear')
        #item = detrend(item, type='constant')
        for item in temp:
            roi2.append(item)
        temp2, label = singleROI(temp)
        print(label)
        peaks.append(temp2)
    i += 1

roix = []

for item in roi:
    for thing in item:
        roix.append(thing)

#print(roix)
#print(roi2)

fig.add_trace(go.Scatter(fillcolor = "RED", x=roix, y=roi2, name = "Things", mode = "markers"))


i = 0

while i < len(peaks):
    if (len(peaks[i]) > 0):
        peaks[i][0] += roi[i][0]
        peaks[i][1] += roi[i][0]
    i += 1


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


fig.add_trace(go.Scatter(fillcolor = "GREEN", x=peaksReal, y=peaksRealy, name = "Zooms", mode = "markers"))

fig.show()


"""
fig2 = go.Figure()

x = [i for i in range(10)]
y = [2*i for i in x]

fig2.add_trace(go.Scatter(x=x, y=y, name="Before"))

y = straighten(y, x)

fig2.add_trace(go.Scatter(x=x, y=y, name="AFTER"))
fig2.show()
"""