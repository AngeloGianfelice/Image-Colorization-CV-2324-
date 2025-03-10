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
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # (56x56)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # (28x28)
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # (14x14)
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),  # (7x7)
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

        self.activation=nn.Tanh()

    def forward(self, x):
        
        x = self.encoder(x)
        x = self.decoder(x)
        x= self.activation(x)
        
        
        return x
    
class VGGAutoencoder(nn.Module):
    def __init__(self):
        super(VGGAutoencoder, self).__init__()
        
        # Load pre-trained VGG16 and remove classification layers
        vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_FEATURES)
        self.encoder = nn.Sequential(*list(vgg16.features.children()))  # Remove classification layers

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
       
        x=x.repeat(1,3,1,1) #convert grayscale to rgb tensor

        with torch.no_grad():  # Freeze encoder computations
            x = self.encoder(x)  # Extract features from VGG-16

        x = self.decoder(x)  # Reconstruct image
        x = self.activation(x)

        return x
        