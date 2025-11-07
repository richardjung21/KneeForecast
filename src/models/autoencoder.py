import torch
import torch.nn as nn
import torch.nn.functional as F
from medvae import MVAE

class Classifier(nn.Module):
    # Edit in the future
    # Currently too simple and not efficient
    def __init__(self, num_classes=2):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

    def calculate_loss(self, x, targets):
        x = self.forward(x)
        return F.cross_entropy(x, targets)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Prototypes(nn.Module):
    def __init__(self, num_classes=2, latent_dim=128):
        super(Prototypes, self).__init__()
        self.prototypes = nn.Parameter(torch.randn(num_classes, latent_dim))
        self.init_prototypes()

    def init_prototypes(self):
        nn.init.kaiming_uniform_(self.prototypes, a=math.sqrt(5))

    def forward(self, x):
        return self.prototypes

class Autoencoder(nn.Module):
    ### Contains functions for training the Autoencoder portion of the model
    ### Forward method returns total loss
    ### Separate encode and decode methods for usage in other classes

    def __init__(self, num_classes=2, latent_dim=128):
        super(Autoencoder, self).__init__()
        self.mvae = MVAE(model_name="medvae_4_3_2d", modality="xray")
        self.classifier = Classifier(num_classes=num_classes)
        self.prototypes = Prototypes(num_classes=num_classes, latent_dim=latent_dim)

    def encode(self, x):
        x = self.mvae.encode(x)
        return x
    
    def decode(self, x):
        x = self.mvae.decode(x)
        return x

    def calculate_prototype_loss(self, x):
        return F.mse_loss(x, self.prototypes)

    def forward(self, x):
        x = self.encode(x)
        class_loss = self.classifier.calculate_loss(x)
        prototype_loss = self.calculate_prototype_loss(x)
        return x