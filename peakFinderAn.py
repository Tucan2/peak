import numpy as np
import plotly.graph_objects as go 
from processing import singleROI
from scipy import integrate
from tsmoothie.smoother import *
from scipy.signal import detrend
import sys

np.set_printoptions(threshold=sys.maxsize)

def fixDetrend(smoothed):
    minimum = min(smoothed)
    if minimum < 0:
        minimum *= -1
    i = 0
    while i < len(smoothed):
        smoothed[i] = smoothed[i] + minimum + 0.01
        i += 1
    return smoothed


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

dataset = 40 #1 of 141(?)

yvalues = np.array(naidata[dataset])

naiintegrated = [0] * 1024

for lst in naidata:
    for n in range(len(lst)-1):
        naiintegrated[n] += int(lst[n])
"""
for i in range (2):
    for n in range(len(naidata[i])-1):
        naiintegrated[n] += int(naidata[i][n])
"""

xvalues = np.array(range(0, 1023))
length = 1023
integratedyvalues = naiintegrated

xvalsList = xvalues.tolist()

fig = go.Figure()

integratedyvalues = np.log1p(integratedyvalues)

fig.add_trace(go.Scatter(fillcolor = "GREEN", x=xvalues, y=integratedyvalues, name = "Original", mode = "lines+markers"))
print(integratedyvalues)


maxIndex = 0
maxNum = naiintegrated[0]

i = 0
while i < len(naiintegrated)-2:
    if (naiintegrated[i] > maxNum):
        maxNum = naiintegrated[i]
        maxIndex = i
    if ((naiintegrated[i] < 4.5 and i > 200) or i == 975): # < 4.5, 975 arbitrary; purpose is to cut off data at the end 
        length = i
        break
    i += 1
    
xvalsList = xvalsList[maxIndex: length]

for i in range(maxIndex):
    naiintegrated = np.delete(naiintegrated, 0)

naiintegrated = naiintegrated[0: length - maxIndex]

linearSectionData = np.log1p(naiintegrated)

endLinearIndex = 110 #110 end index for linear, in energy 
linearSectionData = linearSectionData.tolist()
linearSectionData = linearSectionData[0: endLinearIndex]

linearSectionData = detrend(linearSectionData, type='linear')

#smoother = GaussianSmoother(sigma=0.075, n_knots=50)
smoother = LowessSmoother(smooth_fraction=0.02, iterations=2)
#smoother = ConvolutionSmoother(8, 'blackman')

smoother.smooth(naiintegrated)

low, up = smoother.get_intervals('sigma_interval')

smoothNoLog = smoother.smooth_data[0]

smoothed = np.log1p(naiintegrated)
smoothed = detrend(smoothed, type='linear')

for i in range(len(linearSectionData)):
    smoothed[i] = linearSectionData[i]

np.set_printoptions(threshold=sys.maxsize)

fig.add_trace(go.Scatter(x=xvalues, y=smoothed, name = "Detrended Counts", mode = "lines+markers"))

smoother.smooth(smoothed)

# generate intervals
#low, up = smoother.get_intervals('prediction_interval')
low, up = smoother.get_intervals('sigma_interval')

smoothed = smoother.smooth_data[0]
smoothed = fixDetrend(smoothed)

minYPoints = []
pointsMin = []

i = 1
while i < len(smoothed) - 1:
    if (smoothed[i-1] > smoothed[i] and smoothed[i+1] > smoothed[i]):
        pointsMin.append(i)
        minYPoints.append(smoothed[i])
    i += 1

peaks = []

i = 1
while i < len(minYPoints):
    betweenMax = max(smoothed[pointsMin[i-1]: pointsMin[i]])
    if (not (betweenMax < 1.02 * float((minYPoints[i]+minYPoints[i-1])/2.0))):
        j = pointsMin[i-1]
        betweenMaxIndex = -1
        average = 0
        extra = detrend(smoothed[j:pointsMin[i]])

        extra = fixDetrend(extra)

        index = 0
        while j < pointsMin[i]:
            if (smoothed[j] == betweenMax):
                betweenMaxIndex = j
            average += extra[index]
            j+= 1
            index += 1

        if (pointsMin[i-1] <= 110):
            distance = int((betweenMaxIndex-pointsMin[i-1] + pointsMin[i] - betweenMaxIndex)/2.0)
            peaks.append([max(betweenMaxIndex-distance, pointsMin[i-1]), min(betweenMaxIndex + distance, pointsMin[i])])
        elif (not (float(average/float(pointsMin[i]-pointsMin[i-1])) - float((extra[0]+extra[-1])/2.0) < 0.2 * float((minYPoints[i]+minYPoints[i-1])/2.0))):
            distance = int((betweenMaxIndex-pointsMin[i-1] + pointsMin[i] - betweenMaxIndex)/2.0)
            peaks.append([max(betweenMaxIndex-distance, pointsMin[i-1]), min(betweenMaxIndex + distance, pointsMin[i])])
    
    i += 1


for i in range(len(peaks)):
    firstPoint = peaks[i][0]
    secondPoint = peaks[i][1]

    trapezoid = 1/2.0 * (secondPoint-firstPoint) * (smoothed[firstPoint] + smoothed[secondPoint])

    ydata = smoothed[firstPoint:secondPoint]
    xdata = xvalues[firstPoint:secondPoint]

    counts = integrate.simpson(ydata, x=xdata) - trapezoid

    firstPoint += maxIndex
    secondPoint += maxIndex
    print("Peak: [" + str(firstPoint) + ", " + str(secondPoint) + "], estimated counts: " + str(counts))


peaksReal = []
peaksRealy = []
i = 0
while i < len(peaks):
    if (len(peaks[i]) != 0):
        j = peaks[i][0]
        while j <= peaks[i][1]:
            peaksReal.append(j+maxIndex)
            peaksRealy.append(integratedyvalues[j+maxIndex])
            j += 1
    i += 1

peaksX = []
for i in range(len(peaks)):
    bounds, label = singleROI(peaks[i])
    if (label == 1):
        peaksX.append(bounds)

if(len(peaksX) > 0):
    print(peaksX)

fig.add_trace(go.Scatter(x=xvalues, y=smoothed, name = "Smoothed Counts"))

fig.add_trace(go.Scatter(fillcolor = "RED", x=pointsMin, y=minYPoints, name = "Min", mode = "markers"))

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

fig.add_trace(go.Scatter(fillcolor = "GREEN", x=peaksReal, y=peaksRealy, name = "Peaks", mode = "markers"))

fig.show()



"""REMEMBER TO DO FORCE CHECK FOR PEAK AT 110"""