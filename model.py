import torch
import torch.nn as nn

# TEST ARCHITECTURE CHANGING SOON

class VCModel(nn.Module):
    def __init__(self):
        super(VCModel, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv3d(3, 16, kernel_size=(3, 3, 4), stride=(2, 2, 1), padding=(1, 1, 0)),
            nn.ReLU(),
            nn.Conv3d(16, 32, kernel_size=(3, 3, 1), stride=(2, 2, 1), padding=(1, 1, 0)),
            nn.ReLU(),
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(32, 16, kernel_size=(3, 3, 1), stride=(2, 2, 1), padding=(1, 1, 0), output_padding=(1, 1, 0)),
            nn.ReLU(),
            nn.ConvTranspose3d(16, 3, kernel_size=(3, 3, 4), stride=(2, 2, 1), padding=(1, 1, 0), output_padding=(1, 1, 0)),
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded