import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

out_channels = 2048
gamma = 0.8

# spatial encoder
class SpatialEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Load pre-trained ResNet50
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        # 1. Modify Layer 3: Change stride from 2 to 1, apply dilation of 2
        resnet.layer3[0].conv2.stride = (1, 1)
        resnet.layer3[0].downsample[0].stride = (1, 1)
        for module in resnet.layer3.modules():
            if isinstance(module, nn.Conv2d) and module.kernel_size == (3, 3):
                module.dilation = (2, 2)
                module.padding = (2, 2)

        # 2. Modify Layer 4: Change stride from 2 to 1, apply dilation of 4
        resnet.layer4[0].conv2.stride = (1, 1)
        resnet.layer4[0].downsample[0].stride = (1, 1)
        for module in resnet.layer4.modules():
            if isinstance(module, nn.Conv2d) and module.kernel_size == (3, 3):
                module.dilation = (4, 4)
                module.padding = (4, 4)

        # Remove the final AvgPool and FC layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        
        # Freeze early layers, but leave modified layer3 and layer4 unfrozen
        for name, param in self.backbone.named_parameters():
            if 'layer3' in name or 'layer4' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        
    def forward(self, x):
        # Input: (B, 3, 256, 256)
        # Output: (B, 2048, 32, 32) instead of (B, 2048, 8, 8)
        return self.backbone(x)



class LeGall53DWT2D(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.in_channels = in_channels
        
        # 1D filters
        Fl = torch.tensor([-1/8, 1/4, 3/4, 1/4, -1/8], dtype=torch.float32)
        Fh = torch.tensor([0, -1/2, 1, -1/2, 0], dtype=torch.float32)
        
        # 2D filters via outer product
        LL = Fl.unsqueeze(1) * Fl.unsqueeze(0)
        HL = Fh.unsqueeze(1) * Fl.unsqueeze(0)
        LH = Fl.unsqueeze(1) * Fh.unsqueeze(0)
        HH = Fh.unsqueeze(1) * Fh.unsqueeze(0)
        
        # Stack filters: shape (4, 1, 5, 5)
        filters = torch.stack([LL, HL, LH, HH], dim=0).unsqueeze(1)
        
        # Repeat for each input channel: shape (4 * in_channels, 1, 5, 5)
        filters = filters.repeat(in_channels, 1, 1, 1)
        
        # Register as buffer so it's moved to the correct device but not updated by optimizer
        self.register_buffer('filters', filters)
        
    def forward(self, x):
        # x: (B, C, H, W)
        # groups=in_channels applies the 4 filters to each channel independently
        # Output: (B, C * 4, H/2, W/2)
        return F.conv2d(x, self.filters, stride=2, padding=2, groups=self.in_channels)


class FrequencyEncoder(nn.Module):
    def __init__(self, in_channels=3, out_channels=out_channels):
        super().__init__()

        self.dwt = LeGall53DWT2D(in_channels=in_channels)
        
        # The DWT outputs in_channels * 4 channels, at half the spatial resolution (H/2, W/2)
        # We need to reach H/8, W/8. So we need an overall stride of 4 after the DWT.
        self.conv1 = nn.Conv2d(in_channels * 4, 64, kernel_size=3, padding=1, stride=2)
        self.conv2 = nn.Conv2d(64, out_channels, kernel_size=3, padding=1, stride=2) 
        
    def forward(self, x):
        # x: (B, C, H, W)
        x_dwt = self.dwt(x) # (B, C*4, H/2, W/2)
        
        x_freq = F.relu(self.conv1(x_dwt)) # (B, 64, H/4, W/4)
        x_freq = self.conv2(x_freq)        # (B, 2048, H/8, W/8)
        return x_freq


def masked_pooling(features, mask):
    """
        features: (Batch, Channels, H, W)
        mask: (Batch, 1, H, W)
    """

    # resize mask to the size of the feature maps
    if mask.shape[-2:] != features.shape[-2:]:
        mask = F.interpolate(mask, size=features.shape[-2:], mode='nearest')
    
    masked_features = features * mask
    
    sum_features = torch.sum(masked_features, dim=(-1, -2), keepdim=True) # Shape: (B, C, 1, 1)
    mask_area = torch.sum(mask, dim=(-1, -2), keepdim=True) # Shape: (B, 1, 1, 1)
    prototype = sum_features / (mask_area + 1e-8)

    return prototype


class PrototypeFusion(nn.Module):
    def __init__(self, channels=out_channels):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=channels, num_heads=8, batch_first=True)
        
    def forward(self, p_s, p_f):
        B, C, _, _ = p_s.shape
        p_s = p_s.view(B, 1, C)
        p_f = p_f.view(B, 1, C)
        
        # Q = spatial prototype, KV = frequency prototype
        attn_output, _ = self.attention(query=p_s, key=p_f, value=p_f)
        
        p_fused = attn_output.view(B, C, 1, 1)
        return p_fused


class BandWeighting(nn.Module):
    def __init__(self, channels=out_channels):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // 4),
            nn.ReLU(),
            nn.Linear(channels // 4, channels),
            nn.Sigmoid() 
        )
        
    def forward(self, p_f):
        B, C, _, _ = p_f.shape
        p_f_flattened = p_f.view(B, C)

        weights = self.mlp(p_f_flattened).view(B, C, 1, 1)

        return p_f * weights


def compute_similarity_map(query_features, prototype):
    """
    query_features: (B, C, H, W)
    prototype: (B, C, 1, 1)
    """
    sim_map = F.cosine_similarity(query_features, prototype, dim=1) # Shape: (B, H, W)
    
    return sim_map.unsqueeze(1) # Shape: (B, 1, H, W)


def dice_loss(logits, target):
        # Dice loss on foreground
        probs = F.softmax(logits, dim=1)[:, 1, :, :] # (B, H, W)
        target_float = target.float()
        intersection = (probs * target_float).sum(dim=(1, 2))
        union = probs.sum(dim=(1, 2)) + target_float.sum(dim=(1, 2))
        dice_loss = 1.0 - (2.0 * intersection + 1e-5) / (union + 1e-5)
        dice_loss = dice_loss.mean()

        return dice_loss


class DualBranchPANet(nn.Module):
    def __init__(self):
        super().__init__()
        # Backbone output channels
        channels = 2048 
        
        self.spatial_encoder = SpatialEncoder()
        self.freq_encoder = FrequencyEncoder(in_channels=3, out_channels=channels)
        self.prototype_fusion = PrototypeFusion(channels=channels)
        self.band_weighting = BandWeighting(channels=channels)
        
        self.fusion_weight1 = nn.Parameter(torch.tensor(0.25, requires_grad=True))
        self.fusion_weight2 = nn.Parameter(torch.tensor(0.25, requires_grad=True))
        self.scaler = nn.Parameter(torch.tensor(20.0))
        
    def forward(self, support_img, query_img, fore_mask, back_mask=None, target=None):
        """
        Args:
            support_img: (B, K, 3, H, W) - Support images
            query_img: (B, 3, H, W) - Query image
            fore_mask: (B, K, 1, H, W) - Foreground mask
            back_mask: (B, K, 1, H, W) - Background mask (optional)
            target: (B, H, W) - Target mask (optional)
        Returns:
            logits: (B, 2, H, W) - Logits
            loss: (1, ) - Loss
        """
        B, K, C, H, W = support_img.shape
        support_img = support_img.view(B * K, C, H, W)
        fore_mask = fore_mask.view(B * K, 1, H, W)
        
        if back_mask is None:
            back_mask = 1 - fore_mask
        else:
            back_mask = back_mask.view(B * K, 1, H, W)

        # Extract features for Support Image
        f_s = self.spatial_encoder(support_img)
        f_f = self.freq_encoder(support_img)
        
        # Extract features for Query Image
        q_s = self.spatial_encoder(query_img)
        q_f = self.freq_encoder(query_img)
        
        # spatial prototypes
        p_s_fg = masked_pooling(f_s, fore_mask)   # (B*K, C, 1, 1)
        p_s_bg = masked_pooling(f_s, back_mask)   # (B*K, C, 1, 1)

        # frequency prototypes
        p_f_fg = masked_pooling(f_f, fore_mask)   # (B*K, C, 1, 1)
        p_f_bg = masked_pooling(f_f, back_mask)   # (B*K, C, 1, 1)
        
        # aggregate K shots
        p_s_fg = p_s_fg.view(B, K, -1, 1, 1).mean(dim=1)  # (B, C, 1, 1)
        p_s_bg = p_s_bg.view(B, K, -1, 1, 1).mean(dim=1)
        p_f_fg = p_f_fg.view(B, K, -1, 1, 1).mean(dim=1)
        p_f_bg = p_f_bg.view(B, K, -1, 1, 1).mean(dim=1)
        
        # prototype fusion using MHA
        p_fused_fg = self.prototype_fusion(p_s_fg, p_f_fg)
        p_fused_bg = self.prototype_fusion(p_s_bg, p_f_bg)

        # Bandweighting of frequency prototypes
        p_f_weighted_fg = self.band_weighting(p_f_fg)
        p_f_weighted_bg = self.band_weighting(p_f_bg)

        # Simalarity scores
        spatial_sim_fg = compute_similarity_map(q_s, p_s_fg)
        spatial_sim_bg = compute_similarity_map(q_s, p_s_bg)

        fused_sim_fg = compute_similarity_map(q_s, p_fused_fg)       # (B, 1, H', W')
        fused_sim_bg = compute_similarity_map(q_s, p_fused_bg)       # (B, 1, H', W')

        freq_sim_fg = compute_similarity_map(q_f, p_f_weighted_fg)     # (B, 1, H', W')
        freq_sim_bg = compute_similarity_map(q_f, p_f_weighted_bg)     # (B, 1, H', W')

        
        # Similarity score fusion
        sim_fg = self.fusion_weight2 * spatial_sim_fg + self.fusion_weight1 * freq_sim_fg + (1 - self.fusion_weight1 - self.fusion_weight2) * fused_sim_fg
        sim_bg = self.fusion_weight2 * spatial_sim_bg + self.fusion_weight1 * freq_sim_bg + (1 - self.fusion_weight1 - self.fusion_weight2) * fused_sim_bg  
        
        logits = torch.cat([sim_bg, sim_fg], dim=1) * self.scaler      # (B, 2, H', W')
        
        logits = F.interpolate(
            logits, 
            size=query_img.shape[-2:], 
            mode='bilinear', 
            align_corners=False
        )  # (B, 2, H, W)

        loss = None
        if target is not None:
            ce_loss = F.cross_entropy(logits, target.long(), reduction='mean') # (1, )
            d_loss = dice_loss(logits, target)
            loss_seg = (1 - gamma) * ce_loss + gamma * d_loss
            
            # --- Prototype Alignment Regularization (PAR) ---
            # 1. Obtain predicted mask for query
            pred_mask = F.softmax(logits, dim=1)[:, 1:2, :, :]  # Uses continuous foreground probability            
            # Since q_s is H', W' and logits is H, W, we need to use nearest neighbor interpolation to downsample pred_mask back to H', W' for masked_pooling
            pred_mask_ds = F.interpolate(pred_mask, size=q_s.shape[-2:], mode='nearest')
            pred_back_mask_ds = 1.0 - pred_mask_ds
            
            # 2. Extract query prototypes
            q_s_fg = masked_pooling(q_s, pred_mask_ds)   # (B, C, 1, 1)
            q_s_bg = masked_pooling(q_s, pred_back_mask_ds)
            q_f_fg = masked_pooling(q_f, pred_mask_ds)
            q_f_bg = masked_pooling(q_f, pred_back_mask_ds)
            
            # 3. Prototype fusion for query prototypes
            q_fused_fg = self.prototype_fusion(q_s_fg, q_f_fg)
            q_fused_bg = self.prototype_fusion(q_s_bg, q_f_bg)
            
            # 4. Bandweighting for query frequency prototypes
            q_f_weighted_fg = self.band_weighting(q_f_fg)
            q_f_weighted_bg = self.band_weighting(q_f_bg)
            
            # 5. Calculate similarities for support image using query prototypes
            # Expand query prototypes to match B*K support images
            q_s_fg_exp = q_s_fg.repeat_interleave(K, dim=0)
            q_s_bg_exp = q_s_bg.repeat_interleave(K, dim=0)
            q_fused_fg_exp = q_fused_fg.repeat_interleave(K, dim=0)
            q_fused_bg_exp = q_fused_bg.repeat_interleave(K, dim=0)
            q_f_weighted_fg_exp = q_f_weighted_fg.repeat_interleave(K, dim=0)
            q_f_weighted_bg_exp = q_f_weighted_bg.repeat_interleave(K, dim=0)
            
            par_spatial_sim_fg = compute_similarity_map(f_s, q_s_fg_exp)
            par_spatial_sim_bg = compute_similarity_map(f_s, q_s_bg_exp)
            par_fused_sim_fg = compute_similarity_map(f_s, q_fused_fg_exp)
            par_fused_sim_bg = compute_similarity_map(f_s, q_fused_bg_exp)
            par_freq_sim_fg = compute_similarity_map(f_f, q_f_weighted_fg_exp)
            par_freq_sim_bg = compute_similarity_map(f_f, q_f_weighted_bg_exp)
            
            # 6. Fusion
            par_sim_fg = self.fusion_weight2 * par_spatial_sim_fg + self.fusion_weight1 * par_freq_sim_fg + (1 - self.fusion_weight1 - self.fusion_weight2) * par_fused_sim_fg
            par_sim_bg = self.fusion_weight2 * par_spatial_sim_bg + self.fusion_weight1 * par_freq_sim_bg + (1 - self.fusion_weight1 - self.fusion_weight2) * par_fused_sim_bg
            
            par_logits = torch.cat([par_sim_bg, par_sim_fg], dim=1) * self.scaler # (B*K, 2, H', W')
            par_logits = F.interpolate(par_logits, size=support_img.shape[-2:], mode='bilinear', align_corners=False)
            
            # 7. PAR Loss against support ground truth
            support_target = fore_mask.squeeze(1) # (B*K, H, W)
            par_ce_loss = F.cross_entropy(par_logits, support_target.long(), reduction='mean')
            par_d_loss = dice_loss(par_logits, support_target)
            loss_par = (1 - gamma) * par_ce_loss + gamma * par_d_loss
            
            loss = 0.5 * (loss_seg + loss_par)
    
        return logits, loss


# Usage
if __name__ == "__main__":
    print("Initializing model...")
    model = DualBranchPANet()
    
    # Create dummy tensors representing a batch of 2 RGB images, K=10 shots
    B, K = 2, 10
    dummy_support_img = torch.randn(B, K, 3, 256, 256)
    dummy_query_img = torch.randn(B, 3, 256, 256)
    
    # Create dummy binary masks (foreground and background)
    fore_mask = torch.randint(0, 2, (B, K, 1, 256, 256)).float()
    back_mask = 1.0 - fore_mask
    
    print("Running forward pass...")
    dummy_gt = torch.randint(0, 2, (2, 256, 256)).long()
    logits, loss = model(dummy_support_img, dummy_query_img, fore_mask, back_mask, target=dummy_gt)
    
    print(f"Output shape: {logits.shape}")  # Expected: (2, 2, 256, 256)
    assert logits.shape == (2, 2, 256, 256), f"Unexpected shape: {logits.shape}"
    print(f"Cross-entropy loss: {loss.item():.4f}")
    
    # Verify gradient flow
    loss.backward()
    print("Backward pass successful — gradients computed.")
    print("No errors, yay.")
