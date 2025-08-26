import torch
import torch.nn as nn

class EncodingCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoding = nn.Sequential( 
            nn.Dropout(0.15), #better for training
            nn.Conv1d(in_channels=2, out_channels=4, kernel_size=5, padding=2), nn.LazyBatchNorm1d(),
            nn.SiLU(), #activiation function called swish
            nn.Conv1d(in_channels=4, out_channels=8, kernel_size=5, padding=2), nn.LazyBatchNorm1d(),
            nn.SiLU(), 
            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=5, padding=2), nn.LazyBatchNorm1d(),
            nn.SiLU(), 
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, padding=2), nn.LazyBatchNorm1d(),
            nn.SiLU(),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2), nn.LazyBatchNorm1d(),
            nn.SiLU(),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, padding=2), nn.LazyBatchNorm1d(),
            nn.SiLU()
        )

    def forward(self, x):
        return self.encoding(x).transpose(2, 1)
    
    def reset_parameters(self):
        for item in self.encoding:
            if (isinstance(item, nn.Conv1d)): #where there are weights and biases
                #initalization that makes for better training
                nn.init.kaiming_uniform_(item.weight, mode='fan_in', nonlinearity='relu')
                nn.init.zeros_(item.bias)
    
#class ResidualBlock(nn.Module)
    
class Classifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Dropout(0.15), #training better
            nn.Linear(256, 128), nn.LazyBatchNorm1d(), nn.SiLU(), 
            nn.Linear(128, 64), nn.LazyBatchNorm1d(), nn.SiLU(), 
            nn.Linear(64, 32), nn.LazyBatchNorm1d(), nn.SiLU(), 
            nn.Linear(32, 16), nn.LazyBatchNorm1d(), nn.SiLU(), 
            nn.Linear(16, 8), nn.LazyBatchNorm1d(), nn.SiLU(), 
            nn.Linear(8, 4), nn.LazyBatchNorm1d(), nn.SiLU(), 
            nn.Linear(4, 2))

    def forward(self, x):
        return self.classifier(x)
    
    def reset_parameters(self):
        for item in self.classifier:
            if (isinstance(item, nn.Linear)): #where there are weights and biases
                #initalization that makes for better training
                nn.init.kaiming_uniform_(item.weight, mode='fan_in', nonlinearity='relu') 
                nn.init.zeros_(item.bias)

class Intergrator(nn.Module):
    def __init__(self):
        super().__init__()

        self.integrator = nn.Sequential(
            nn.Dropout(0.15), #training better
            nn.Linear(512, 256), nn.LazyBatchNorm1d(), nn.SiLU(), 
            nn.Linear(256, 128), nn.LazyBatchNorm1d(), nn.SiLU(), 
            nn.Linear(128, 64), nn.LazyBatchNorm1d(), nn.SiLU(), 
            nn.Linear(64, 32), nn.LazyBatchNorm1d(), nn.SiLU(), 
            nn.Linear(32, 16), nn.LazyBatchNorm1d(), nn.SiLU(), 
            nn.Linear(16, 8), nn.LazyBatchNorm1d(), nn.SiLU(), 
            nn.Linear(8, 4), nn.LazyBatchNorm1d(), nn.SiLU(), 
            nn.Linear(4, 2))
        
    def forward(self, x):
        return self.classifier(x)
    
    def reset_parameters(self):
        for item in self.classifier:
            if (isinstance(item, nn.Linear)): #where there are weights and biases
                #initalization that makes for better training
                nn.init.kaiming_uniform_(item.weight, mode='fan_in', nonlinearity='relu') 
                nn.init.zeros_(item.bias)

class RecurrentCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoding = EncodingCNN() #only convolves, adds a bunch of output channels

        self.LSTM = nn.LSTM(128, 128, num_layers=5, dropout=0.2, batch_first=True, bidirectional=True)
        self.LSTM2 = nn.LSTM(256, 256, num_layers=5, dropout=0.2, batch_first=True, bidirectional=True)
        
        self.classifier = Classifier() #use cross enthropy loss, adam when training
        self.integrator = Intergrator()

    def _preprocessing(self, batch):
        batch_size, _, n_points = batch.shape
        processed_batch = batch.clone().view(batch_size, n_points)

        for x in processed_batch:
            x[x < 1e-4] = 0
            pos = (x != 0)
            x[pos] = torch.log10(x[pos])
            x[pos] = x[pos] - torch.min(x[pos])
            x[pos] = x[pos] / torch.max(x[pos])
            
        return processed_batch.view(batch_size, 1, n_points) #batch_size, 1, n_points = sizing

    def forward(self, x):
        x = torch.cat((x, self._preprocessing(x)), dim=1) #torch.cat appends the 2 tensors

        x = self.encoding(x)
        x, _ = self.LSTM(x)

        integrator_input, (classifier_input, _) = self.LSTM2(x)

        classifier_output = self.classifier(classifier_input[0])
        integrator_output = self.integrator(integrator_input)

        return classifier_output, integrator_output.transpose(2, 1)
