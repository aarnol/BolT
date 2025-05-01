from torch import nn
import torch
# Assuming you have a pre-trained model on fMRI data
class TransferModel(nn.Module):
    def __init__(self, pretrained_model, output_classes):
        super(TransferModel, self).__init__()
        
        # Extract feature extraction layers from pretrained model
        self.feature_extractor = nn.Sequential(*list(pretrained_model.children())[:-2])
        
        # Freeze the feature extractor weights
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
            
        # Create new task-specific layers for fNIRS
        self.adaption_layer = nn.Sequential(
            nn.Linear(pretrained_model.feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        # New classification head
        self.classifier = nn.Linear(256, output_classes)
    
    def forward(self, x):
        # Transform fNIRS data to BOLD-like format if needed
        
        # Extract features using frozen layers
        features = self.feature_extractor(x)
        
        # Adapt features through new trainable layers
        adapted = self.adaption_layer(features)
        
        # Classification
        output = self.classifier(adapted)
        return output

