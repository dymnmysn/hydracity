import torch.nn as nn

class HydraWaymo(nn.Module):
    def __init__(self, hydramodel):
        super(HydraWaymo, self).__init__()
        self.encoder = hydramodel.salsa.encoder
        self.decoder = hydramodel.salsa.decoder2
    
    def forward(self, x):
        encoder_output, encoder_features = self.encoder(x)
        logits = self.decoder(encoder_output, encoder_features)
        return logits