import random
import csv
import plotly.graph_objects as go 
import numpy as np
from datetime import datetime

def createPeaks(peakLength, type, dataRand):
    middleLength = 0
    if (type[1]):
        middleLength = random.randint(int(peakLength*.2), int(peakLength*.8))
    
    totalPeakLength = peakLength
    peakLength -= middleLength
    if (type[0]):
        upLength = random.randint(int(peakLength*0.4), int(peakLength*0.6))
        downLength = peakLength - upLength

        maximum = max(dataRand)
        maximum *= 1000
        maximum = int(maximum)
        peakHeight = float(random.randint(150+maximum, 1000+maximum))

        peakHeight /= 1000.0

        peak = []

        numSlopes = random.randint(1, 5)
        slopes = []

        if (numSlopes <= -1):
            slopes.append([peakHeight/float(upLength), upLength])
            
            print("HI")
            print(slopes)
        else:
            while (True):
                slopes = []
                
                firstSlopeLength = random.randint(1, int(upLength-1))
                slopes.append([float(random.randint(5, 55))/1000.0, firstSlopeLength])
                if (slopes[0][0]*firstSlopeLength  < peakHeight):
                    break
            
            slopes.append([(peakHeight - slopes[0][0])/(upLength-slopes[0][1])])
            slopes[1].append(upLength-firstSlopeLength)
            print(slopes)

def makeParabola(point1, point2):
    a = (point2[1] - point1[1])/(point2[0]-point1[0])**2
    equation = str(a) + "*((x-" + str(point1[0]) + ")**2) + " + str(point1[1])
    return equation

def checkValidDatarand(dataRand, length):
    while True:
        element = dataRand[0]
        dataRand = dataRand[1:]
        if ((0.1 + element) >= 0.05):
            return(dataRand, element)
        elif (len(dataRand) == 0):
            dataRand = np.random.default_rng().normal(loc=0.1, scale=0.1, size=length)

x = 5
a = 0
exec("a = " + str(makeParabola([2, 0.5], [10, 2])))

type = [True, True, False, False]

for i in range(1):
    (random.randint)
    length = random.randint(25, 125)
    peakLength = random.randint(length//4, length)    
    dataRand = np.random.default_rng().normal(loc=0.1, scale=0.1, size=length)

    peak = createPeaks(peakLength, type, dataRand)

    data = []

    startIndex = random.randint(0, length-peakLength)

    i = 0
    while i < length:
        if (i == startIndex):
            j = 0
            while j < len(peak):
                data.append(peak[j])
                j += 1
                i += 1
        else:
            dataRand, element = checkValidDatarand(dataRand, length)
            data.append(element)
            i += 1
    

a=[]


fig = go.Figure()
fig.add_trace(go.Scatter(x=np.array(range(0, len(a))).tolist(), y=a, name = "Integrated Counts"))

fig.update_layout(title="Integrated Gamma Spectra", xaxis_title="Energy [kev]",  yaxis_title="Counts", font=dict(family="LEMON MILK", size=18, color="Black"), plot_bgcolor='rgba(0,0,0,0)', template='plotly_white')

fig.update_yaxes(showgrid=True, gridcolor="lightgrey", linecolor="black", tickcolor="black", tickfont=dict(color="black", size=10), title_font = {"size": 22, "color": "black"}, title_standoff = 15,)

fig.update_xaxes(linecolor="black", tickfont=dict(color="black", size=10), title_font = {"size": 22, "color": "black"}, title_standoff = 15)

#fig.show()

#need to have lines from straight at whatever min angle is, same for curve
#then need to have 0 to some straight at top
#then straight line to min angle down
#for curve same but note in can be concave up/down