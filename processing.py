import torch
from rcnn import RecurrentCNN
import numpy as np
import plotly.graph_objects as go 

def singleROI(roi):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') #where to run model
    model = RecurrentCNN().to(device) #saving model
    model.load_state_dict(torch.load("RecurrentCNN.pt", map_location=device)) #getting weights for model

    model.eval() #storing weights

    #preprossesing signal to get it into right format

    signal = torch.tensor(roi / np.max(roi), dtype=torch.float32, device=device)
    signal = signal.view(1, 1, -1)

    classifier_output, segmentator_output = model(signal) #classifier to determine whether there is a peak, segementator to get the bunds of the peak
    
    classifier_output = classifier_output.data.cpu().numpy()
    segmentator_output = segmentator_output.data.sigmoid().cpu().numpy()

    label = np.argmax(classifier_output) #0 is no peak, 1 is peak

    #xvalues = [x for x in range(len(classifier_output))]
    #fig = go.Figure()
    #fig.add_trace(go.Scatter(x=xvalues, y=classifier_output, name = "the outoput thing"))
    #fig.show()

    feature = []

    if label == 1: #if there is a peak
        borders = get_borders(segmentator_output[0, 0, :], segmentator_output[0, 1, :]) #gets peak borders
        for border in borders:
            feature.append(border[0]) #start of peak
            feature.append(border[1]) #end of peak

    return feature, label


def get_borders(integration_mask, intersection_mask):
    domain = integration_mask * (1 - intersection_mask) > .5 #.5 can be changed, threshold; applies a boolean 
    
    borders_roi = []
    begin = 0 if domain[0] else -1
    peakWidth = 1 if domain[0] else 0

    for n in range(len(domain) - 1):
        if domain[n + 1] and not domain[n]:  #peak begins
            begin = n + 1
            peakWidth = 1
        elif domain[n + 1] and begin != -1:  #peak continues
            peakWidth += 1
        elif not domain[n + 1] and begin != -1:  # peak ends
            if peakWidth > 3: #3 = minimium peak length
                b = int(begin //1)
                e = int((n + 2) // 1)
                borders_roi.append([b, e]) #add beginning and end
            begin = -1 #peak ended
            peakWidth = 0
    
    #if peak does not end before end of roi
    if begin != -1 and peakWidth > 3:
        b = int(begin // 1)
        e = int((len(domain)-b-1)// 1)
        borders_roi.append([b, e])
    
    return borders_roi