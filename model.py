"""
Vanilla autoencoder for GoL frames — designed for PIXEL-PERFECT reconstruction.

Architecture intent:
  - The 128×128 input grid is implicitly partitioned into a 16×16 grid of
    non-overlapping 8×8 tiles.
  - Each 8×8 tile is encoded into one of the 16×16 latent feature positions
    (with 128 channels at that position).
  - All 32,768 latent-feature values are then fully connected to a flat
    z ∈ ℝ^d, mediating *global* information between tiles.
  - The decoder mirrors: z → 16×16×128 → each tile decoded back via
    per-pixel channel mixing + PixelShuffle 2× upsamples to 128×128.

Conceptual symmetry — critical for pixel-perfect reconstruction:
  - Encoder uses kernel=2 stride=2 convs for downsampling: each output pixel
    is computed from a disjoint 2×2 patch of input. No overlap between
    adjacent latent positions' receptive fields. Each 16×16 latent position
    sees exactly one 8×8 input tile.
  - Encoder uses 1×1 convs between strides: per-pixel channel mixing only,
    no spatial blur within tiles.
  - Decoder uses 1×1 convs inside PixelShuffleUp: the per-position channels
    are rearranged spatially via PixelShuffle but never mixed across
    neighbours by a 3×3 conv. Each output pixel is computed from exactly
    one latent feature position's channels.
  - The Linear z → 16×16×128 projection is the SOLE point where tile
    information mixes globally. It's fully-connected, so any tile's
    decoder features can be informed by any aspect of any other tile.

This eliminates spatial blur across pixels in BOTH encoder and decoder,
so the model cannot produce a "halo" of probability around true alive
cells: there is no architectural layer that mixes signal between adjacent
output pixels except via the global z bottleneck.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


LATENT_DIM = 1024
GRID_HW    = 128


# -----------------------------------------------------------------------------
# Building blocks
# -----------------------------------------------------------------------------

class ChannelMixRefine(nn.Module):
    """Two convs with a residual skip. Kernel size parametrised so the same
    module serves both stages of two-stage training:
      - kernel_size=3: spatial mixing for trainability (Stage A bootstrap)
      - kernel_size=1: per-pixel channel mixing only, halo-free (Stage B)

    Encoder always uses kernel_size=1 (tile-disjoint by design).
    """
    def __init__(self, c: int, kernel_size: int = 1):
        super().__init__()
        pad = kernel_size // 2
        self.c1 = nn.Conv2d(c, c, kernel_size, padding=pad)
        self.c2 = nn.Conv2d(c, c, kernel_size, padding=pad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x + self.c2(F.gelu(self.c1(x))))


class TileDownsample(nn.Module):
    """Non-overlapping 2×2 → 1×1 downsample via kernel=2 stride=2 conv.

    Each output pixel comes from a disjoint 2×2 input patch — adjacent output
    pixels do NOT see overlapping input regions. Used in the encoder so that
    each 16×16 latent position has a receptive field of exactly one 8×8 input
    tile after three stages of this operation.
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=2, stride=2, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.conv(x))


class PixelShuffleUp(nn.Module):
    """Sub-pixel 2× upsample. Kernel size parametrised:
      - kernel_size=3: spatial mixing at the lower resolution before the shuffle
        (gives the model spatial cooperation, but adjacent output pixels
        from different source positions share overlapping receptive fields
        → introduces ~1-pixel halo). Used in Stage A.
      - kernel_size=1: per-pixel channel mixing only, then PixelShuffle
        rearranges channels into the upper-resolution grid. No spatial
        information shared between source positions → halo-free. Used in
        Stage B after weight surgery transfers learned features.
    """
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 1):
        super().__init__()
        pad = kernel_size // 2
        self.conv    = nn.Conv2d(in_ch, out_ch * 4, kernel_size, padding=pad)
        self.shuffle = nn.PixelShuffle(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.shuffle(self.conv(x)))


# -----------------------------------------------------------------------------
# Encoder
# -----------------------------------------------------------------------------

class Encoder(nn.Module):
    """128×128 binary grid → latent z ∈ ℝ^d.

    Three TileDownsample stages (128→64→32→16), each followed by a
    ChannelMixRefine block at the new resolution. Final flatten + linear
    projection to d. No L2-normalisation; magnitude and direction are both
    free, soft-bounded by the L3 norm loss.
    """
    def __init__(self, d: int = LATENT_DIM):
        super().__init__()
        self.conv = nn.Sequential(
            TileDownsample(1,   32),   ChannelMixRefine(32),   # 128 → 64
            TileDownsample(32,  64),   ChannelMixRefine(64),   # 64  → 32
            TileDownsample(64, 128),   ChannelMixRefine(128),  # 32  → 16
        )
        self.project = nn.Linear(128 * 16 * 16, d, bias=True)

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        h = self.conv(grid)
        h = h.flatten(1)
        return self.project(h)


# -----------------------------------------------------------------------------
# Decoder
# -----------------------------------------------------------------------------

class Decoder(nn.Module):
    """Latent z ∈ ℝ^d → 128×128 logits.

    Architecture is parametrised by `kernel_size`:
      - kernel_size=3 (Stage A): spatial cooperation in PixelShuffleUp +
        ChannelMixRefine. Trainable but introduces ~1-pixel halo.
      - kernel_size=1 (Stage B, after weight surgery): pure per-pixel decoding,
        no spatial smoothing → halo-free → bit-perfect possible.

    Linear projection (z → 16×16×128) is always full-connected: every latent
    feature position is informed by all d latent dimensions. That's the global
    routing layer; everything downstream is local.
    """
    def __init__(self, d: int = LATENT_DIM, kernel_size: int = 1):
        super().__init__()
        ks = kernel_size
        self.kernel_size = ks
        self.project = nn.Linear(d, 128 * 16 * 16)
        self.deconv = nn.Sequential(
            PixelShuffleUp(128, 64, ks), ChannelMixRefine(64, ks),     # 16 → 32
            PixelShuffleUp(64,  32, ks), ChannelMixRefine(32, ks),     # 32 → 64
            PixelShuffleUp(32,  16, ks), ChannelMixRefine(16, ks),     # 64 → 128
            nn.Conv2d(16, 1, kernel_size=1),                            # final always 1×1
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.project(z).view(-1, 128, 16, 16)
        return self.deconv(h)


def transfer_decoder_weights_3x3_to_1x1(decoder_3x3: "Decoder",
                                        decoder_1x1: "Decoder") -> None:
    """Surgery: copy weights from a kernel_size=3 decoder into a kernel_size=1
    decoder. For each 3×3 conv, take the center weight (position (1, 1)) as
    the 1×1 kernel — this preserves the dominant "I see myself" signal of
    each 3×3 kernel, dropping the neighbour contributions. The linear
    projection (full-connected) and final 1×1 conv transfer unchanged.

    The result is a decoder that approximates the 3×3 decoder's behaviour
    on spatially-constant feature maps and diverges from it elsewhere — but
    has a sensible initialization (not random) and can fine-tune to a
    halo-free version using the encoder's already-learned latent structure.
    """
    sd_3x3 = decoder_3x3.state_dict()
    sd_1x1 = decoder_1x1.state_dict()
    for k in sd_1x1.keys():
        if k not in sd_3x3:
            raise KeyError(f"missing key in source decoder: {k}")
        src = sd_3x3[k]
        dst = sd_1x1[k]
        if src.shape == dst.shape:
            sd_1x1[k] = src.clone()
        elif src.dim() == 4 and dst.dim() == 4 and src.shape[2:] == (3, 3) and dst.shape[2:] == (1, 1):
            # Extract centre weight from 3×3 kernel as the 1×1 equivalent
            sd_1x1[k] = src[:, :, 1:2, 1:2].clone()
        else:
            raise ValueError(f"shape mismatch for {k}: src {src.shape} vs dst {dst.shape}")
    decoder_1x1.load_state_dict(sd_1x1)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    enc = Encoder()
    dec = Decoder()
    n_enc = count_params(enc)
    n_dec = count_params(dec)
    print(f"Encoder params: {n_enc:>12,}")
    print(f"Decoder params: {n_dec:>12,}")
    print(f"Total:          {n_enc + n_dec:>12,}")

    g = torch.zeros(2, 1, GRID_HW, GRID_HW)
    g[0, 0, 30, 30] = 1.0
    z = enc(g)
    out = dec(z)
    print(f"\nencoder input:  {g.shape}")
    print(f"encoder output: {z.shape}, ‖z‖ per sample = {z.norm(dim=-1).tolist()}")
    print(f"decoder output: {out.shape}")

    # Conceptual receptive-field check:
    # Each 16×16 latent position should see exactly one 8×8 input region.
    test_in = torch.zeros(1, 1, GRID_HW, GRID_HW)
    test_in[0, 0, 0, 0] = 1.0  # single alive pixel at top-left corner
    h = enc.conv(test_in).abs().sum(dim=1).squeeze(0)  # (16, 16)
    nonzero = (h > 1e-6).nonzero()
    print(f"\nRF sanity: single pixel at (0,0) activates {len(nonzero)} latent positions")
    if len(nonzero) > 0:
        print(f"  activated positions: {nonzero.tolist()}")
    print(f"  ✓ should be exactly 1 position at (0,0) for tile-disjoint encoder")
