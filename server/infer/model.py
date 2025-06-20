import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class SignClassifier(nn.Module):
    """Stage 1: Multi-label classification for clinical signs"""
    def __init__(self, num_signs=10, pretrained=True):
        super(SignClassifier, self).__init__()
        # CoAtNet backbone
        self.backbone = timm.create_model('coatnet_2_rw_224.sw_in12k', 
                                        pretrained=pretrained, 
                                        features_only=True)
        
        # Get feature dimension from backbone
        dummy = torch.zeros(1, 3, 224, 224)
        with torch.no_grad():
            out = self.backbone(dummy)
        self.feature_dim = out[-1].shape[1]
        
        # Global average pooling
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_signs)
        )
        
    def forward(self, x):
        # Extract features
        features = self.backbone(x)[-1]  # Get last stage output
        features = self.pool(features).view(features.size(0), -1)
        
        # Classify signs
        sign_logits = self.classifier(features)
        return sign_logits
    
    def freeze_backbone(self):
        """Freeze backbone parameters"""
        for param in self.backbone.parameters():
            param.requires_grad = False

class GlaucomaClassifier(nn.Module):
    """Stage 2: Binary classification for glaucoma using predicted signs"""
    def __init__(self, num_signs=10):
        super(GlaucomaClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(num_signs, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Binary classification
        )
    
    def forward(self, sign_logits):
        return self.classifier(sign_logits)

class TwoStageModel(nn.Module):
    """Combined two-stage model"""
    def __init__(self, num_signs=10, pretrained=True):
        super(TwoStageModel, self).__init__()
        self.sign_classifier = SignClassifier(num_signs, pretrained)
        self.glaucoma_classifier = GlaucomaClassifier(num_signs)
        
    def forward(self, x):
        # Stage 1: Predict signs
        sign_logits = self.sign_classifier(x)
        
        # Stage 2: Predict glaucoma
        glaucoma_logits = self.glaucoma_classifier(sign_logits)
        
        return sign_logits, glaucoma_logits
    
    def freeze_sign_classifier(self):
        """Freeze sign classifier parameters"""
        self.sign_classifier.freeze_backbone()
        for param in self.sign_classifier.classifier.parameters():
            param.requires_grad = False 