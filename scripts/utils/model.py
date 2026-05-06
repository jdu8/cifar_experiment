# scripts/utils/model.py

import torch
import torch.nn as nn
import torchvision.models as models


class ResNetCIFAR(nn.Module):
    def __init__(self, model_name='resnet18', num_classes=10):
        super().__init__()

        base = getattr(models, model_name)(weights=None)

        # Modify for 32x32 CIFAR input
        base.conv1   = nn.Conv2d(3, 64, kernel_size=3,
                                 stride=1, padding=1, bias=False)
        base.maxpool = nn.Identity()

        self.encoder = nn.Sequential(*list(base.children())[:-1])
        self.fc      = nn.Linear(base.fc.in_features, num_classes)

    def forward(self, x, return_embedding=False):
        emb = self.encoder(x)
        emb = emb.view(emb.size(0), -1)
        out = self.fc(emb)
        if return_embedding:
            return out, emb
        return out


def get_model(model_name='resnet18', num_classes=10):
    model = ResNetCIFAR(model_name=model_name, num_classes=num_classes)
    total = sum(p.numel() for p in model.parameters())
    print(f"Model: {model_name} | Parameters: {total:,}")
    return model