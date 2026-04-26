import torch
import torch.nn as nn
import torchvision.models as models


class VisualExtractor(nn.Module):
    def __init__(self, args):
        super(VisualExtractor, self).__init__()

        self.visual_extractor = args.visual_extractor

        # ==============================
        # LOAD BACKBONE (NEW API)
        # ==============================
        if self.visual_extractor == 'resnet18':
            weights = models.ResNet18_Weights.DEFAULT
            backbone = models.resnet18(weights=weights)
            self.out_dim = 512

        elif self.visual_extractor == 'resnet34':
            weights = models.ResNet34_Weights.DEFAULT
            backbone = models.resnet34(weights=weights)
            self.out_dim = 512

        elif self.visual_extractor == 'resnet50':
            weights = models.ResNet50_Weights.DEFAULT
            backbone = models.resnet50(weights=weights)
            self.out_dim = 2048

        else:
            raise ValueError(f"Unsupported backbone: {self.visual_extractor}")

        # Remove avgpool & fc
        self.model = nn.Sequential(*list(backbone.children())[:-2])

        # Global pooling
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # ==============================
        # 🔥 FREEZE BACKBONE (IMPORTANT)
        # ==============================
        for name, p in self.model.named_parameters():
            if "layer4" in name:
                p.requires_grad = True
            else:
                p.requires_grad = False

    def forward(self, images):
        """
        images: (B, 3, H, W)
        returns:
            att_feats: (B, N, C)
            fc_feats:  (B, C)
        """

        # 🔥 NO GRAD → saves a LOT of memory
        with torch.no_grad():
            feat_map = self.model(images)   # (B, C, H, W)

        B, C, H, W = feat_map.shape

        # ==============================
        # ATTENTION FEATURES
        # ==============================
        att_feats = feat_map.view(B, C, -1).permute(0, 2, 1)  # (B, N, C)

        # ==============================
        # GLOBAL FEATURES
        # ==============================
        fc_feats = self.avg_pool(feat_map).view(B, C)  # (B, C)

        return att_feats, fc_feats