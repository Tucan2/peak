import numpy as np
import plotly.graph_objects as go 
from scipy import integrate
from tsmoothie.smoother import *
from scipy.signal import detrend
import sys
import math

np.set_printoptions(threshold=sys.maxsize)

#ensures all data points positive
def fixDetrend(smoothed):
    minimum = min(smoothed)
    
    if minimum < 0:
        minimum *= -1

    i = 0
    while i < len(smoothed):
        smoothed[i] = smoothed[i] + minimum + 0.01
        i += 1

    return smoothed

def peakFinderAnalytic(fileName, dataset=0, numFiles=-1, energyCalibrated=[False, False]):
    #energyCalibrated: whether the data should be energy calibrated, and whether it is NAI data or not
    #Get data
    naidatafile = open(fileName, "r") 

    naidataraw = [i.split(',') for i in naidatafile.read().split('\n')]

    naidataraw.remove([""])

    naidata = []
    
    xvaluesinbins = np.array(range(0, 1024))

    xvaluesinenergy = xvaluesinbins

    if (energyCalibrated[1]):
        for fiveminuteinterval in naidataraw:
            naidata.append(fiveminuteinterval[6:])
        if (energyCalibrated[0]):
            #Calibrate energy values
            energycalibration = lambda x: 3.87589476e-04 * x ** 2 + 3.59950887e+00 * x + 2.62548505e-13
            xvaluesinenergy = energycalibration(xvaluesinbins)
    else:
        calib_total = 0.0

        #have to ignore first index since naidataraw[0][5] is a string while naidataraw[i][5] is a number
        i = 1
        while i < len(naidataraw):
            calib_total += float(naidataraw[i][5])
            naidata.append(naidataraw[i][6:])
            i += 1

        if (energyCalibrated[0]):
            #Calibrate energy values
            calib_total = float(calib_total/len(naidata))
            for i in range(len(xvaluesinenergy)):
                xvaluesinenergy[i] = calib_total * i

            

    naiintegrated = [0] * 1024

    #sum up relevant number of samples from file
    #the more samples, the less noisy the data and thus the better the program will work
    if(numFiles < 1):
        for lst in naidata:
            for n in range(len(lst)-1):
                naiintegrated[n] += float(lst[n])
    else:
        
        i = dataset
        while i <= (dataset + numFiles):
            for n in range(len(naidata[i])-1):
                naiintegrated[n] += float(naidata[i][n])
            i += 1

    length = 1023

    xvalsList = xvaluesinenergy.tolist()

    fig = go.Figure()

    integratedyvalues = np.log1p(naiintegrated)

    fig.add_trace(go.Scatter(fillcolor = "GREEN", x=xvaluesinenergy, y=integratedyvalues, name = "Original", mode = "lines+markers"))

    #only consider data after the maximum value (so scipy's detrend works properly)
    #also ignore the tail end of the data (not meaningful with regards to peaks)
    maxIndex = 0
    maxNum = naiintegrated[0]

    i = 0
    while i < len(naiintegrated)-2:
        if (naiintegrated[i] > maxNum):
            maxNum = naiintegrated[i]
            maxIndex = i
        if (i == 975 or (i > 200 and naiintegrated[i] > 0 and math.log(naiintegrated[i]) < 0.2 * math.log(maxNum))): # < 0.3, > 200, == 975 arbitrary; purpose is to cut off data at the end 
            length = i
            break
        i += 1

    xvalsList = xvalsList[maxIndex: length]

    for i in range(maxIndex):
        naiintegrated = np.delete(naiintegrated, 0)

    naiintegrated = naiintegrated[0: length - maxIndex]

    #first part of data often is linear with a different slope compared to the rest of the data
    #thus detrended (de-linearized) differently
    linearSectionData = np.log1p(naiintegrated)

    endLinearIndex = 110 #110 considered end index for linear section
    linearSectionData = linearSectionData.tolist()
    linearSectionData = linearSectionData[0: endLinearIndex]

    linearSectionData = detrend(linearSectionData, type='linear')

    smoothed = np.log1p(naiintegrated)
    smoothed = detrend(smoothed, type='linear') #rest of the data also delinearized)

    for i in range(len(linearSectionData)):
        smoothed[i] = linearSectionData[i]

    fig.add_trace(go.Scatter(x=xvaluesinenergy, y=smoothed, name = "Detrended Counts", mode = "lines+markers"))

    smoother = LowessSmoother(smooth_fraction=0.02, iterations=2) #to denoise data
    smoother.smooth(smoothed)

    smoothed = smoother.smooth_data[0]
    smoothed = fixDetrend(smoothed) #ensures no negative data

    #get local minima in the smoothed, delinearized data
    minYPoints = []
    pointsMin = []

    i = 1
    while i < len(smoothed) - 1:
        if (smoothed[i-1] > smoothed[i] and smoothed[i+1] > smoothed[i]):
            pointsMin.append(i)
            minYPoints.append(smoothed[i])
        i += 1

    peaks = []

    #tries to determine whether there is peak between two minimia
    i = 1
    while i < len(minYPoints):
        betweenMax = max(smoothed[pointsMin[i-1]: pointsMin[i]]) #maximum between the two minimia

        minimaAverage = float((minYPoints[i]+minYPoints[i-1])/2.0)

        #the maximum has to be at least 1.02 * the average of the minimia
        #pervents linear sections between minimia from being counted as a peak 
        if (betweenMax >= 1.02 * minimaAverage): #
            j = pointsMin[i-1]
            betweenMaxIndex = -1
            average = 0

            #want detrended smoothed area
            extra = detrend(smoothed[j:pointsMin[i]])
            extra = fixDetrend(extra)

            #find the index of the max and calculates the average of the peak
            index = 0
            while j < pointsMin[i]:
                if (smoothed[j] == betweenMax):
                    betweenMaxIndex = j
                average += extra[index]
                j+= 1
                index += 1

            average = float(average/float(pointsMin[i]-pointsMin[i-1]))

            #the average must be at least 1.05 the average of the detrended section
            peakLargeEnough = average >= 1.05 * float((extra[0]+extra[-1])/2.0)

            #before 110 is relatively linear so peakLargeEnough need not apply
            if (pointsMin[i-1] <= 110 or peakLargeEnough):
                #try to make the peak boundaries more accurate by looking at average distance between minimia and maximia too
                minimiaDistance = int((betweenMaxIndex-pointsMin[i-1] + pointsMin[i] - betweenMaxIndex)/2.0) 
                peaks.append([max(betweenMaxIndex-minimiaDistance, pointsMin[i-1]), min(betweenMaxIndex + minimiaDistance, pointsMin[i])])
        
        i += 1

    for i in range(len(peaks)):
        firstPoint = peaks[i][0]
        secondPoint = peaks[i][1]

        ydata = smoothed[firstPoint:secondPoint]
        xdata = xvaluesinenergy[firstPoint:secondPoint]

        #estimate background with linear approximation
        trapezoid = 1/2.0 * (xdata[-1]-xdata[0]) * (ydata[0] + ydata[-1]) 

        #calculates counts in the relevant area
        counts = integrate.simpson(ydata, x=xdata) - trapezoid

        firstPoint += maxIndex
        secondPoint += maxIndex
        
        firstPoint = xvaluesinenergy[firstPoint]
        secondPoint = xvaluesinenergy[secondPoint]
        print("Peak: [" + str(firstPoint) + ", " + str(secondPoint) + "], estimated counts: " + str(counts))

    peaksReal = []
    peaksRealy = []
    i = 0
    while i < len(peaks):
        if (len(peaks[i]) != 0):
            j = peaks[i][0]
            while j <= peaks[i][1]:
                peaksReal.append(xvaluesinenergy[j+maxIndex])
                peaksRealy.append(integratedyvalues[j+maxIndex])
                j += 1
        i += 1

    pointsMin = [xvaluesinenergy[i] for i in pointsMin]

    fig.add_trace(go.Scatter(x=xvaluesinenergy, y=smoothed, name = "Smoothed Counts"))

    fig.add_trace(go.Scatter(fillcolor = "RED", x=pointsMin, y=minYPoints, name = "Min", mode = "markers"))

    if (energyCalibrated[0]):
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
    else:
        fig.update_layout(
            title="Integrated Gamma Spectra",
            xaxis_title="Bins/4", 
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

#peakFinderAnalytic("/Users/sciencegenuis2089/DataFiles/ahs_os_d3s.csv", numFiles= 36, energyCalibrated=[True, False]) #Taken from website
#peakFinderAnalytic("/Users/sciencegenuis2089/DataFiles/etch_roof_d3s.csv", dataset=3, numFiles=2) #Taken from website
peakFinderAnalytic("Data1655337600nai.csv", numFiles=1, energyCalibrated=[True, True]) 