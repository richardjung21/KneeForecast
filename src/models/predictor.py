import torch
import torch.nn as nn
import torch.nn.functional as F
from medvae import MVAE
import math
from vector_quantize_pytorch import VectorQuantize as VQ
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
from sklearn.metrics import roc_auc_score, mean_absolute_error
import numpy as np

class Classifier(nn.Module):
    # Classifier Model that can be instantiated for many classification tasks
    # num_classes is the number of classes to classify
    def __init__(self, c, h, w, num_classes=2):
        super(Classifier, self).__init__()
        self.fc0 = nn.Linear(c * h * w, 1024)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

    def calculate_loss(self, x, targets):
        x = self.forward(x)
        return F.nll_loss(F.log_softmax(x, dim=1), targets.long())

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.gelu(self.fc0(x))
        x = F.gelu(self.fc1(x))
        x = F.gelu(self.fc2(x))
        x = self.fc3(x)
        return x

class Regression(nn.Module):
    # Regression Model that can be instantiated for many regression tasks
    # num_classes is the number of classes to classify
    def __init__(self, c, h, w, outputs=1):
        super(Regression, self).__init__()
        self.fc0 = nn.Linear(c * h * w, 1024)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, outputs)
    def calculate_loss(self, x, targets):
        x = self.forward(x)
        return F.mse_loss(x.squeeze(1), targets)
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.gelu(self.fc0(x))
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        return x

class Prototypes(nn.Module):
    ### Initializes prototypes for each class in the normalized latent space
    ### Prototypes are used to force separation of classes in the latent space
    ### Still to be implemented
    ### Incomplete and should not yet be used
    def __init__(self, num_classes=2, latent_dim=128):
        super(Prototypes, self).__init__()
        self.prototypes = nn.Parameter(torch.randn(num_classes, latent_dim))
        self.init_prototypes()

    def init_prototypes(self):
        nn.init.kaiming_uniform_(self.prototypes, a=math.sqrt(5))

    def forward(self, x):
        return self.prototypes

class Predictor(nn.Module):
    ### Contains functions for training the Autoencoder portion of the model
    ### Forward method returns total loss
    ### Separate encode and decode methods for usage in other classes

    def __init__(self, num_classes:list[int] = [2], regressions:int = 1, latent_dim=32):
        super(Predictor, self).__init__()
        self.mvae = MVAE(model_name="medvae_4_3_2d", modality="xray")
        for param in self.mvae.model.encoder.parameters():
            param.requires_grad = False
        self.classifiers = nn.ModuleList()
        for num_class in num_classes:
            self.classifiers.append(Classifier(3, 64, 64, num_class))
        self.regressions = Regression(3, 64, 64, regressions)
        
        #self.prototypes = Prototypes(num_classes=2, latent_dim=latent_dim)
        
        self.adapter_head = nn.Sequential(
            nn.Conv2d(3, latent_dim, kernel_size=1, padding=0),
            nn.GELU(),
            nn.Conv2d(latent_dim, latent_dim, kernel_size=1, padding=0),
            nn.GELU(),
            nn.Conv2d(latent_dim, 3, kernel_size=1, padding=0),
        )

    def encode(self, x):
        return self.mvae.encode(x)
    
    def decode(self, x):
        return self.mvae.decode(x)

    def calculate_prototype_loss(self, x):
        return x
    
    def train_concepts(self, x, classifications, regressors):
        # Expects input classifications to be a tensor of shape (batch_size, tasks) for both classification and regression
        latent = self.encode(x)
        latent = latent.detach() # Stop gradients from flowing back to the encoder
        latent = latent + self.adapter_head(latent)
        concept_loss = 0
        for i, classifier in enumerate(self.classifiers):
            class_loss = classifier.calculate_loss(latent, classifications[:, i])
            concept_loss += class_loss
        regression_loss = self.regressions.calculate_loss(latent, regressors)
        concept_loss += regression_loss

        concept_loss = concept_loss/(len(self.classifiers) + 1)
        
        y = self.decode(latent)
        return concept_loss, y

    def validate_concepts(self, x, classifications, regressors):
        latent = self.encode(x)
        latent = latent.detach() # Stop gradients from flowing back to the encoder
        latent = latent + self.adapter_head(latent)
        
        concept_loss = 0
        
        # Store predictions and targets for metric calculation
        all_class_preds = []
        all_class_targets = []
        all_reg_preds = []
        all_reg_targets = []

        for i, classifier in enumerate(self.classifiers):
            class_loss = classifier.calculate_loss(latent, classifications[:, i])
            concept_loss += class_loss
            
            # Get predictions
            logits = classifier(latent)
            probs = F.softmax(logits, dim=1)
            all_class_preds.append(probs)
            all_class_targets.append(classifications[:, i])
            
        regression_loss = self.regressions.calculate_loss(latent, regressors)
        concept_loss += regression_loss
        
        # Get predictions
        preds = self.regressions(latent)
        if preds.shape[1] == 1:
            preds = preds.squeeze(1)
        all_reg_preds.append(preds)
        all_reg_targets.append(regressors)

        concept_loss = concept_loss/(len(self.classifiers) + 1)
        
        y = self.decode(latent)
        return concept_loss, y, all_class_preds, all_class_targets, all_reg_preds, all_reg_targets

    def forward(self, x):
        x = self.encode(x)


if __name__ == "__main__":
    predictor = Predictor(num_classes=[3, 4, 3, 3], regressions=2, latent_dim=32)
    print(predictor)
    # print(predictor.train_concepts(torch.randn(2, 3, 256, 256), torch.tensor([[1,3,2,2],[0,2,2,1]]), torch.randn((2,2))))