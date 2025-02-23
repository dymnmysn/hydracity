import torch.nn as nn

class HydraKitti(nn.Module):
    def __init__(self, hydramodel):
        super(HydraKitti, self).__init__()
        self.encoder = hydramodel.salsa.encoder
        self.decoder = hydramodel.salsa.decoder1
    
    def forward(self, x):
        encoder_output, encoder_features = self.encoder(x)
        logits = self.decoder(encoder_output, encoder_features)
        return logits