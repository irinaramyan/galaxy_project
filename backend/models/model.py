# using a pre-trained ResNet model and applying transfer learning
import torch 
import torch.nn as nn
from torchvision import models 

class GalaxyClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()

        # loading resnet
        self.backbone = models.resnet18(weights="IMAGENET1K_V1")

        # freeze early layers, they dont need training
        for param in list(self.backbone.parameters())[:-20]: # keeping only last 20 params trainable
            param.requires_grad = False 

        # replace final fc layer
        in_features = self.backbone.fc.in_features # taking the in_features from the original model
        self.backbone.fc = nn.Identity() # does nothing and passes unchanged
        self.head = nn.Sequential( # adding custom layers
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes) # 512 -> 256 -> 4 (num classes)
        )

    def forward(self, x):
        embedding = self.backbone(x) # 512-dim vector
        logits = self.head(embedding)
        return logits, embedding