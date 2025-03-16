import torch.nn as nn
import torchvision.models as models
import torch

class ColorizationAutoencoder(nn.Module):
    def __init__(self):
        super(ColorizationAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential( # [(W+2P-K)/S]+1 = [(W-1)/2] +1
            # 224x224
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # (112x112)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # (56x56)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # (28x28)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # (14x14)
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),  # (7x7)
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        # Decoder
        self.decoder = nn.Sequential(  #(W-1)x2+2
            # 7x7
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),  # (14x14)
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # (28x28)
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # (56x56)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # (112x112)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 2, kernel_size=3, stride=2, padding=1, output_padding=1),  # (224x224)
        )

        #final activation
        self.activation=nn.Tanh()

        # Learned Gamma Correction
        self.gamma_correction = GammaCorrectionModule(init_gamma=1.0)

    def forward(self, x):
        
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.activation(x)
        x = self.gamma_correction(x)
        return x
    
class GammaCorrectionModule(nn.Module):
    def __init__(self, init_gamma=1.0):
        super(GammaCorrectionModule, self).__init__()
        self.gamma = nn.Parameter(torch.tensor(init_gamma, dtype=torch.float32)) #learnable gamma correction parameter

    def forward(self, ab_channels):
            
        ab_normalized = (ab_channels + 1) / 2  # Scale to [0,1]
        ab_corrected = torch.pow(ab_normalized, self.gamma)  # Apply gamma
        ab_corrected = ab_corrected * 2 - 1  # Scale back to [-1,1]
        return ab_corrected
    
class ResnetAutoencoder(nn.Module):
    def __init__(self):
        super(ResnetAutoencoder, self).__init__()
        
        # Load pre-trained VGG16 and remove classification layers
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.encoder = nn.Sequential(*list(resnet.children())[:-2])  # Remove classification layers
        # Freeze encoder weights
        for param in self.encoder.parameters():
            param.requires_grad = False  # Encoder is fixed (pretrained weights)

        # Decoder
        self.decoder = nn.Sequential(     #(W-1)x2+2
            # 7x7
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),  # (14x14)
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # (28x28)
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # (56x56)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # (112x112)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 2, kernel_size=3, stride=2, padding=1, output_padding=1),  # (224x224)
        )

        self.activation=nn.Tanh()

    def forward(self, x):

        x = self.encoder(x) # Extract features from Resnet-18
        x = self.decoder(x)  # Reconstruct image
        x = self.activation(x)

        return x
        