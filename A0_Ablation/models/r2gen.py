import torch
import torch.nn as nn

from modules.visual_extractor import VisualExtractor
from modules.encoder_decoder import EncoderDecoder
from modules.problem_encoder import ProblemEncoder


class R2GenModel(nn.Module):
    """
    Clean multimodal R2Gen:
    Image (2 views) + Problem → Report
    """

    def __init__(self, args, tokenizer):
        super().__init__()

        self.args = args
        self.tokenizer = tokenizer

        # =========================
        # VISUAL ENCODER
        # =========================
        self.visual_extractor = VisualExtractor(args)
        self.vis_proj = nn.Linear(self.visual_extractor.out_dim, args.d_model)

        # =========================
        # PROBLEM ENCODER
        # =========================
        self.problem_encoder = ProblemEncoder(
            input_dim=args.problem_dim,
            embed_dim=args.d_model
        )

        # =========================
        # DECODER
        # =========================
        self.encoder_decoder = EncoderDecoder(args, tokenizer)

    # ======================================================
    # REQUIRED FOR PYTORCH
    # ======================================================
    def forward(self, images, problem_vec, targets=None, mode='train'):

        if self.args.dataset_name == 'iu_xray':
            return self.forward_iu_xray(images, problem_vec, targets, mode)
        else:
            return self.forward_mimic(images, problem_vec, targets, mode)

    # ======================================================
    # IU-XRAY
    # ======================================================
    def forward_iu_xray(self, images, problem_vec, targets, mode):

        # ---- IMAGE FEATURES ----
        att1, fc1 = self.visual_extractor(images[:, 0])
        att2, fc2 = self.visual_extractor(images[:, 1])

        att_feats = torch.cat([att1, att2], dim=1)
        att_feats = self.vis_proj(att_feats)

        fc_feats = (fc1 + fc2) / 2  # kept for compatibility (not used in decoder)

        # ---- PROBLEM ----
        problem_embed = self.problem_encoder(problem_vec)

        # ---- FUSION ----
        visual = att_feats.permute(1, 0, 2)  # (N, B, D)

        problem_embed = problem_embed.unsqueeze(0).repeat(
            visual.size(0), 1, 1
        )

        fused = torch.cat([visual, problem_embed], dim=0)
        fused = fused.permute(1, 0, 2)  # (B, T, D)

        # ---- DECODER ----
        return self.encoder_decoder(fused, targets, mode)

    # ======================================================
    # MIMIC-CXR
    # ======================================================
    def forward_mimic(self, images, problem_vec, targets, mode):

        att_feats, fc_feats = self.visual_extractor(images)
        att_feats = self.vis_proj(att_feats)

        problem_embed = self.problem_encoder(problem_vec)

        visual = att_feats.permute(1, 0, 2)

        problem_embed = problem_embed.unsqueeze(0).repeat(
            visual.size(0), 1, 1
        )

        fused = torch.cat([visual, problem_embed], dim=0)
        fused = fused.permute(1, 0, 2)

        return self.encoder_decoder(fused, targets, mode)