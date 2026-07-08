from imports import *

# ── 3.1  Spatial Encoder (ResNet-50 backbone, returns multi-scale features) ───

class SpatialEncoder(nn.Module):
    """
    ResNet-50 encoder that returns 4 hierarchical feature maps for skip
    connections AND the bottleneck feature.

    Output channels:
        s0 : (B, 64,  H/2,  W/2)   after initial conv+BN+ReLU
        s1 : (B, 256, H/4,  W/4)   layer1  (res block, 256ch)
        s2 : (B, 512, H/8,  W/8)   layer2
        s3 : (B, 1024,H/16, W/16)  layer3
        s4 : (B, 2048,H/32, W/32)  layer4  — bottleneck input

    All weights shared between support and query forward passes.
    """
    def __init__(self, pretrained: bool = True):
        super().__init__()
        backbone = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        )
        self.enc0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.pool = backbone.maxpool
        self.enc1 = backbone.layer1   # 256ch
        self.enc2 = backbone.layer2   # 512ch
        self.enc3 = backbone.layer3   # 1024ch
        self.enc4 = backbone.layer4   # 2048ch

    def forward(self, x):
        """Returns (s0, s1, s2, s3, s4) at progressively coarser resolutions."""
        s0 = self.enc0(x)             # 64ch,  H/2
        s1 = self.enc1(self.pool(s0)) # 256ch, H/4
        s2 = self.enc2(s1)            # 512ch, H/8
        s3 = self.enc3(s2)            # 1024ch,H/16
        s4 = self.enc4(s3)            # 2048ch,H/32
        return s0, s1, s2, s3, s4


# ── 3.2  Spectral MLP (maps FFT magnitude+phase → real feature map) ───────────

class SpectralMLP(nn.Module):
    """
    Converts the 2D FFT of an input image into a real-valued feature map with
    the same spatial resolution and channel dim as the spatial encoder output.

    Pipeline per image:
        1. Apply 2D FFT → complex spectrum (H×W)
        2. Split into magnitude log(1+|F|) and phase angle(F)
        3. Concatenate along channel dim → (2C_in, H, W) real tensor
        4. Feed through lightweight conv MLP → (FEAT_DIM, H, W)
        5. Bilinear resize to match spatial encoder resolution at each scale

    Design note: we compute FFT on the raw 3-channel image (C=3), giving 6
    real channels (3 mag + 3 phase) as input to the MLP.  This is computationally
    cheap and avoids coupling with the spatial encoder.
    """
    def __init__(self, in_channels: int = 6, feat_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, 64,       kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128,               kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, feat_dim,         kernel_size=1, bias=False),
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """
        x : (B, 3, H, W)  normalized image tensor
        Returns : (B, feat_dim, H, W)
        """
        # 2D FFT per channel (operates on last two dims)
        x_fft = torch.fft.fft2(x, norm='ortho')          # complex (B,3,H,W)

        # Magnitude: log-compress to stabilize dynamic range
        magnitude = torch.log1p(torch.abs(x_fft))         # (B,3,H,W)

        # Phase: shift FFT origin to center (fftshift analog via roll)
        phase     = torch.angle(x_fft)                    # (B,3,H,W) in [-π,π]

        # Concatenate → (B,6,H,W) real-valued input to MLP
        freq_input = torch.cat([magnitude, phase], dim=1)

        return self.mlp(freq_input)                        # (B, feat_dim, H, W)


# ── 3.3  Adaptive Band Weighting (ABW) ────────────────────────────────────────

class AdaptiveBandWeighting(nn.Module):
    """
    Decomposes frequency features into 3 radial rings (low/mid/high spatial
    frequencies) and predicts per-band scalar weights conditioned on the spatial
    prototype P_s.

    Research motivation:
        Different anatomical structures occupy different frequency bands.
        A global lesion boundary → strong mid-frequency signal.
        Fine texture within the lesion → high frequency.
        Global intensity field → low frequency.
        The model learns per-episode which bands to trust.

    Implementation:
        • Build 3 binary masks in the H×W FFT grid defining radial rings
        • Pool frequency features within each ring → 3 ring descriptors
        • Concatenate with P_s → MLP → softmax → (w_low, w_mid, w_high)
        • Return weighted sum of ring-masked frequency feature map
    """
    def __init__(self, feat_dim: int = 256, img_size: int = 256):
        super().__init__()
        self.feat_dim = feat_dim
        self.img_size = img_size

        # MLP: [P_s (feat_dim) || ring_descriptors (3*feat_dim)] → 3 weights
        self.band_mlp = nn.Sequential(
            nn.Linear(feat_dim + 3 * feat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 3),
        )

        # Build radial ring masks (fixed, registered as buffer)
        self.register_buffer('ring_masks', self._build_ring_masks(img_size))

    @staticmethod
    def _build_ring_masks(img_size: int) -> torch.Tensor:
        """
        Returns (3, img_size, img_size) binary float32 masks defining:
            ring 0 (low freq)  : radius 0 to img_size/6
            ring 1 (mid freq)  : radius img_size/6 to img_size/3
            ring 2 (high freq) : radius img_size/3 to img_size/2
        FFT origin assumed at center (after fftshift).
        """
        H, W = img_size, img_size
        cy, cx = H / 2.0, W / 2.0
        ys = torch.arange(H).float().unsqueeze(1).expand(H, W)
        xs = torch.arange(W).float().unsqueeze(0).expand(H, W)
        r  = torch.sqrt((ys - cy) ** 2 + (xs - cx) ** 2)

        r_max = min(H, W) / 2.0
        masks = torch.zeros(3, H, W)
        masks[0] = (r <  r_max / 3).float()
        masks[1] = ((r >= r_max / 3) & (r < 2 * r_max / 3)).float()
        masks[2] = (r >= 2 * r_max / 3).float()
        return masks   # (3, H, W)

    def forward(self, F_f: torch.Tensor, P_s: torch.Tensor) -> torch.Tensor:
        """
        F_f : (B, feat_dim, H, W)   frequency feature map
        P_s : (B, feat_dim)         spatial prototype vector
        Returns : (B, feat_dim, H, W) band-weighted frequency features
        """
        B, C, H, W = F_f.shape

        # Resize ring masks if feature map size differs from img_size
        ring_masks = self.ring_masks  # (3, H_img, W_img)
        if ring_masks.shape[1] != H or ring_masks.shape[2] != W:
            ring_masks = F.interpolate(
                ring_masks.unsqueeze(0), size=(H, W), mode='nearest'
            ).squeeze(0)

        # Pool each frequency ring → descriptor vector per ring
        ring_descs = []
        for k in range(3):
            m   = ring_masks[k].unsqueeze(0).unsqueeze(0)   # (1,1,H,W)
            cnt = m.sum().clamp(min=1.0)
            desc = (F_f * m).sum(dim=(2, 3)) / cnt           # (B, C)
            ring_descs.append(desc)

        # Concatenate: [P_s || desc_low || desc_mid || desc_high]
        mlp_input = torch.cat([P_s] + ring_descs, dim=1)    # (B, 4C)
        weights   = torch.softmax(self.band_mlp(mlp_input), dim=1)  # (B, 3)

        # Weighted sum of ring-masked feature maps
        out = torch.zeros_like(F_f)
        for k in range(3):
            m   = ring_masks[k].unsqueeze(0).unsqueeze(0)           # (1,1,H,W)
            w_k = weights[:, k].view(B, 1, 1, 1)                    # (B,1,1,1)
            out = out + w_k * F_f * m

        return out   # (B, feat_dim, H, W)


# ── 3.4  Cross-Domain Alignment Module (CDAM) — the core contribution ─────────

class CrossDomainAlignmentModule(nn.Module):
    """
    Bidirectional cross-attention between spatial and frequency prototypes,
    producing a fused prototype and dual similarity maps.

    Given:
        P_s  : (B, C)  spatial prototype
        P_f  : (B, C)  frequency prototype (band-weighted)
        Q_s  : (B, C, H, W)  query spatial features
        Q_f  : (B, C, H, W)  query frequency features

    Steps:
        1. Cross-attention:
               P_s' = CrossAttn(Q=P_s, K=P_f, V=P_f)
               P_f' = CrossAttn(Q=P_f, K=P_s, V=P_s)
               P_fused = LayerNorm(P_s' + P_f')

        2. Cosine similarity maps:
               S_spatial(i,j) = cosine(P_fused, Q_s[:,i,j])   shape (B,1,H,W)
               S_freq(i,j)    = cosine(P_f',   Q_f[:,i,j])   shape (B,1,H,W)

        3. Learnable weighted fusion:
               S = sigmoid(alpha) * S_spatial + sigmoid(beta) * S_freq

    The dual similarity map S is passed to the UNet decoder as additional
    spatial guidance injected at the bottleneck.
    """
    def __init__(self, feat_dim: int = 256, n_heads: int = 4):
        super().__init__()
        self.feat_dim = feat_dim

        # Project prototypes to Q/K/V spaces (multi-head cross-attention)
        self.q_proj_s = nn.Linear(feat_dim, feat_dim)
        self.k_proj_s = nn.Linear(feat_dim, feat_dim)
        self.v_proj_s = nn.Linear(feat_dim, feat_dim)

        self.q_proj_f = nn.Linear(feat_dim, feat_dim)
        self.k_proj_f = nn.Linear(feat_dim, feat_dim)
        self.v_proj_f = nn.Linear(feat_dim, feat_dim)

        self.n_heads  = n_heads
        self.head_dim = feat_dim // n_heads
        self.scale    = math.sqrt(self.head_dim)

        self.norm_s = nn.LayerNorm(feat_dim)
        self.norm_f = nn.LayerNorm(feat_dim)

        # Learnable fusion weights (initialized at 0 → sigmoid(0)=0.5 equal blend)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.beta  = nn.Parameter(torch.zeros(1))

    def _cross_attn(self, q_vec, k_vec, v_vec,
                    q_proj, k_proj, v_proj):
        """
        Single-step cross-attention:
          q_vec : (B, C)  — query prototype
          k_vec : (B, C)  — key prototype
        Returns: attended vector (B, C)
        """
        B  = q_vec.size(0)
        H  = self.n_heads
        Hd = self.head_dim

        Q = q_proj(q_vec).view(B, H, Hd)        # (B, H, Hd)
        K = k_proj(k_vec).view(B, H, Hd)
        V = v_proj(v_vec).view(B, H, Hd)

        # Scaled dot-product attention over heads (B×H sequences of length 1)
        attn = (Q * K).sum(dim=-1, keepdim=True) / self.scale   # (B, H, 1)
        attn = torch.softmax(attn, dim=1)                        # (B, H, 1)
        out  = (attn * V).view(B, -1)                           # (B, C)
        return out

    def forward(self, P_s, P_f, Q_s, Q_f):
        """
        P_s  : (B, C)
        P_f  : (B, C)
        Q_s  : (B, C, H, W)
        Q_f  : (B, C, H, W)
        Returns: S (B, 1, H, W), P_fused (B, C)
        """
        B, C, H, W = Q_s.shape

        # ── Cross-attention ──────────────────────────────────────────────────
        # P_s queries P_f (spatial attends to frequency)
        P_s_prime = self._cross_attn(P_s, P_f, P_f,
                                     self.q_proj_s, self.k_proj_s, self.v_proj_s)
        P_s_prime = self.norm_s(P_s + P_s_prime)        # residual + LN

        # P_f queries P_s (frequency attends to spatial)
        P_f_prime = self._cross_attn(P_f, P_s, P_s,
                                     self.q_proj_f, self.k_proj_f, self.v_proj_f)
        P_f_prime = self.norm_f(P_f + P_f_prime)        # residual + LN

        # Fused prototype
        P_fused = P_s_prime + P_f_prime                  # (B, C)

        # ── Dual similarity maps ─────────────────────────────────────────────
        # Reshape query features to (B, C, H*W) for batched cosine sim
        Q_s_flat = Q_s.view(B, C, -1)                   # (B, C, H*W)
        Q_f_flat = Q_f.view(B, C, -1)

        # Prototype vectors as (B, C, 1) for broadcasting
        P_fused_norm = F.normalize(P_fused.unsqueeze(-1), dim=1)   # (B, C, 1)
        P_f_norm     = F.normalize(P_f_prime.unsqueeze(-1), dim=1)

        Q_s_norm = F.normalize(Q_s_flat, dim=1)                    # (B, C, H*W)
        Q_f_norm = F.normalize(Q_f_flat, dim=1)

        S_spatial = (P_fused_norm * Q_s_norm).sum(dim=1)           # (B, H*W)
        S_freq    = (P_f_norm    * Q_f_norm).sum(dim=1)

        S_spatial = S_spatial.view(B, 1, H, W)
        S_freq    = S_freq.view(B, 1, H, W)

        # ── Learnable weighted fusion ────────────────────────────────────────
        a = torch.sigmoid(self.alpha)
        b = torch.sigmoid(self.beta)
        S = a * S_spatial + b * S_freq                  # (B, 1, H, W)

        return S, P_fused, P_f_prime


# ── 3.5  Prototype construction (masked average pooling over K shots) ──────────

def build_prototypes(F_s_list, F_f_list, mask_list):
    """
    Computes dual prototypes by masked average pooling over K support images.

    Args:
        F_s_list : list of K tensors, each (1, C, H, W)  — spatial features
        F_f_list : list of K tensors, each (1, C, H, W)  — frequency features
        mask_list: list of K tensors, each (1, 1, H, W)  — binary masks

    Returns:
        P_s : (1, C)  spatial prototype
        P_f : (1, C)  frequency prototype
    """
    ps_list, pf_list = [], []
    for F_s, F_f, M in zip(F_s_list, F_f_list, mask_list):
        # Resize mask to match feature map resolution
        H, W   = F_s.shape[2], F_s.shape[3]
        M_down = F.interpolate(M.float(), size=(H, W), mode='bilinear',
                               align_corners=False)
        M_down = (M_down > 0.5).float()

        denom = M_down.sum(dim=(2, 3)).clamp(min=1.0)   # (1,1)

        p_s = (F_s * M_down).sum(dim=(2, 3)) / denom    # (1, C)
        p_f = (F_f * M_down).sum(dim=(2, 3)) / denom

        ps_list.append(p_s)
        pf_list.append(p_f)

    # Average prototypes across K shots
    P_s = torch.stack(ps_list, dim=0).mean(dim=0)       # (1, C)
    P_f = torch.stack(pf_list, dim=0).mean(dim=0)

    return P_s, P_f


# ── 3.6  UNet Decoder ──────────────────────────────────────────────────────────

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel=3, pad=1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel, padding=pad, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            ConvBNReLU(in_ch, out_ch),
            ConvBNReLU(out_ch, out_ch),
        )

    def forward(self, x):
        return self.net(x)


class UpBlock(nn.Module):
    """Upsample + skip-concatenate + DoubleConv."""
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear',
                              align_corners=False)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class BottleneckBlock(nn.Module):
    """
    Bottleneck: receives encoder s4 (2048ch) concatenated with similarity map S
    (1ch) → reduces to 1024ch.

    The similarity map injection at the bottleneck is the key architectural
    decision that conditions the entire decoding path on prototype similarity.
    """
    def __init__(self, in_ch: int = 2049, out_ch: int = 1024):
        super().__init__()
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, s4, S):
        """
        s4 : (B, 2048, H/32, W/32)
        S  : (B, 1,    H/?,  W/?)   — similarity map at any resolution
        """
        # Resize S to match s4 spatial dimensions
        S_resized = F.interpolate(S, size=s4.shape[2:], mode='bilinear',
                                  align_corners=False)
        x = torch.cat([s4, S_resized], dim=1)   # (B, 2049, H/32, W/32)
        return self.conv(x)                      # (B, 1024, H/32, W/32)


class FreqFSSDecoder(nn.Module):
    """
    Prototype-guided UNet decoder.

    Inputs:
        s0  : (B, 64,   H/2,  W/2)   encoder skip E1
        s1  : (B, 256,  H/4,  W/4)   encoder skip E2
        s2  : (B, 512,  H/8,  W/8)   encoder skip E3
        s3  : (B, 1024, H/16, W/16)  encoder skip E4 (used in bottleneck upblock)
        s4  : (B, 2048, H/32, W/32)  bottleneck input
        S   : (B, 1,    H,    W)     similarity map

    Output:
        (B, 1, H, W)  sigmoid probability mask
    """
    def __init__(self):
        super().__init__()
        self.bottleneck = BottleneckBlock(in_ch=2049, out_ch=1024)

        # up4: 1024 → 512, skip s3=1024 → out 512
        self.up4 = UpBlock(1024, 1024, 512)
        # up3: 512 → 256, skip s2=512 → out 256
        self.up3 = UpBlock(512,  512,  256)
        # up2: 256 → 128, skip s1=256 → out 128
        self.up2 = UpBlock(256,  256,  128)
        # up1: 128 → 64,  skip s0=64  → out 64
        self.up1 = UpBlock(128,  64,   64)

        # Final 2× upsample + 1×1 conv → binary logit
        self.final_up   = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.final_conv = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, s0, s1, s2, s3, s4, S):
        b  = self.bottleneck(s4, S)   # (B,1024,H/32,W/32)
        d4 = self.up4(b,  s3)         # (B,512, H/16,W/16)
        d3 = self.up3(d4, s2)         # (B,256, H/8, W/8)
        d2 = self.up2(d3, s1)         # (B,128, H/4, W/4)
        d1 = self.up1(d2, s0)         # (B,64,  H/2, W/2)
        out = self.final_up(d1)       # (B,32,  H,   W)
        return self.final_conv(out)   # (B,1,   H,   W)  logits


# ── 3.7  Channel adapter: spatial encoder → FEAT_DIM ─────────────────────────

class ChannelAdapter(nn.Module):
    """
    Reduces the high-dimensional spatial encoder output (2048ch) to FEAT_DIM
    for use in prototype construction and the CDAM module.
    Used only for prototype/similarity computation — skip connections retain
    their original channels for the decoder.
    """
    def __init__(self, in_ch: int = 2048, feat_dim: int = 256):
        super().__init__()
        self.adapt = nn.Sequential(
            nn.Conv2d(in_ch, feat_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.adapt(x)


# ── 3.8  FreqFSS — Full model ─────────────────────────────────────────────────

class FreqFSS(nn.Module):
    """
    FreqFSS: Frequency-Guided Prototype Alignment Few-Shot Segmentation.

    Forward pass API:
        model(support_imgs, support_masks, query_img)
        → predicted_mask logits (B, 1, H, W)
           + P_fused (B, C)  [for alignment loss]
           + P_f_prime (B, C) [for freq consistency loss]
           + Q_f (B, C, H, W) [for freq consistency loss]
           + query_mask_pred_probs [for freq consistency loss target region]

    All support and query images are encoded by the SAME encoder (shared weights).
    """
    def __init__(self, config: 'Config'):
        super().__init__()
        C = config.FEAT_DIM

        self.spatial_enc = SpatialEncoder(pretrained=config.PRETRAINED)
        self.spectral_mlp = SpectralMLP(in_channels=6, feat_dim=C)
        self.channel_adapt = ChannelAdapter(in_ch=2048, feat_dim=C)
        self.abw   = AdaptiveBandWeighting(feat_dim=C, img_size=config.IMG_SIZE)
        self.cdam  = CrossDomainAlignmentModule(feat_dim=C, n_heads=config.ATTN_HEADS)
        self.decoder = FreqFSSDecoder()

    def encode_image(self, x):
        """
        Encodes a single image through both pathways.
        Returns:
            (s0, s1, s2, s3, s4)  spatial skip features
            F_s_adapted            (B, C, H/32, W/32)  for prototype
            F_f                    (B, C, H, W)         frequency features at full res
        """
        s0, s1, s2, s3, s4 = self.spatial_enc(x)
        F_s = self.channel_adapt(s4)          # (B, C, H/32, W/32)

        # Frequency encoder operates on original resolution image
        F_f = self.spectral_mlp(x)            # (B, C, H, W)

        return (s0, s1, s2, s3, s4), F_s, F_f

    def forward(self, support_imgs, support_masks, query_img):
        """
        support_imgs  : (B, K, 3, H, W)
        support_masks : (B, K, 1, H, W)
        query_img     : (B, 3, H, W)

        Returns dict with keys:
            'logits'     : (B, 1, H, W)
            'P_fused'    : (B, C)
            'P_f_prime'  : (B, C)
            'Q_f'        : (B, C, H, W)
            'Q_s'        : (B, C, H, W)   (adapted, for alignment loss)
        """
        B, K, _, H, W = support_imgs.shape

        # ── Encode K support images (batch-wise for efficiency) ──────────────
        sup_F_s_list, sup_F_f_list, sup_mask_list = [], [], []

        for k in range(K):
            sup_k   = support_imgs[:, k]         # (B, 3, H, W)
            mask_k  = support_masks[:, k]        # (B, 1, H, W)
            _, F_s_k, F_f_k = self.encode_image(sup_k)
            # Resize frequency features to match spatial feature map resolution
            H_s, W_s = F_s_k.shape[2], F_s_k.shape[3]
            F_f_k_down = F.interpolate(F_f_k, size=(H_s, W_s),
                                        mode='bilinear', align_corners=False)
            sup_F_s_list.append(F_s_k)
            sup_F_f_list.append(F_f_k_down)
            sup_mask_list.append(mask_k)

        # ── Build dual prototypes (averaged over K shots) ────────────────────
        P_s, P_f_raw = build_prototypes(sup_F_s_list, sup_F_f_list, sup_mask_list)
        # shape: (B, C)

        # ── Encode query image ───────────────────────────────────────────────
        (q_s0, q_s1, q_s2, q_s3, q_s4), Q_s, Q_f_full = self.encode_image(query_img)

        # Resize Q_f to match Q_s spatial resolution for CDAM
        H_q, W_q = Q_s.shape[2], Q_s.shape[3]
        Q_f = F.interpolate(Q_f_full, size=(H_q, W_q),
                             mode='bilinear', align_corners=False)

        # ── Adaptive Band Weighting on frequency prototype features ──────────
        # We apply ABW to the query frequency feature map conditioned on P_s
        Q_f_weighted = self.abw(Q_f, P_s)           # (B, C, H_q, W_q)

        # Also weight the frequency prototype using its own band decomposition
        # by globally pooling the weighted map
        P_f = (Q_f_weighted * F.adaptive_avg_pool2d(
            Q_f_weighted, 1)).sum(dim=(2,3)).unsqueeze(-1).unsqueeze(-1)
        P_f = P_f.squeeze(-1).squeeze(-1)
        # Fallback: use raw frequency prototype if weighting collapses
        # (safe blending: 0.7 weighted + 0.3 raw)
        P_f = 0.7 * P_f_raw + 0.3 * F.normalize(P_f_raw, dim=1)

        # ── Cross-Domain Alignment Module ────────────────────────────────────
        S, P_fused, P_f_prime = self.cdam(P_s, P_f, Q_s, Q_f_weighted)
        # S : (B, 1, H_q, W_q)

        # ── Prototype-guided decoder ─────────────────────────────────────────
        logits = self.decoder(q_s0, q_s1, q_s2, q_s3, q_s4, S)

        return {
            'logits':    logits,
            'P_fused':   P_fused,
            'P_f_prime': P_f_prime,
            'Q_f':       Q_f_weighted,
            'Q_s':       Q_s,
            'P_s':       P_s,
            'S':         S,
        }