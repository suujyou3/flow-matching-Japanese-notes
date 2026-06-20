import torch
from torch import nn
import torch.nn.functional as F

from .models import SinusoidalTimeEmbedding


def _group_count(channels: int) -> int:
    """Choose a GroupNorm group count that divides the channel count."""

    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class TimeConditionedResidualBlock(nn.Module):
    """Small residual block that injects a time embedding as a bias."""

    def __init__(self, channels: int, time_dim: int) -> None:
        super().__init__()
        groups = _group_count(channels)
        self.norm1 = nn.GroupNorm(num_groups=groups, num_channels=channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(time_dim, channels)
        self.norm2 = nn.GroupNorm(num_groups=groups, num_channels=channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        # Add time as a channel-wise bias. This keeps the spatial shape intact.
        # 時刻埋め込みをchannelごとのbiasとして足す。空間サイズは変えない。
        h = h + self.time_proj(time_emb)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return x + h


class TinyUNetVelocity(nn.Module):
    """Tiny image velocity field v_theta(t, x) with U-Net style skip connection.

    This model is intentionally small. It is for reading shape flow and the
    equation/code correspondence, not for high-quality image generation.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        time_dim: int = 128,
    ) -> None:
        super().__init__()
        self.time_embedding = SinusoidalTimeEmbedding(time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.input = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        self.down_block = TimeConditionedResidualBlock(base_channels, time_dim)
        self.downsample = nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1)
        self.middle = TimeConditionedResidualBlock(base_channels * 2, time_dim)
        self.upsample = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1)
        self.up_block = TimeConditionedResidualBlock(base_channels, time_dim)
        self.output = nn.Conv2d(base_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if t.ndim == 1:
            t = t[:, None]
        time_emb = self.time_mlp(self.time_embedding(t))
        h0 = self.input(x)
        # Down path: preserve local image structure while injecting time.
        # down pathでは、局所的な画像構造を保ちながら時刻情報を入れる。
        h1 = self.down_block(h0, time_emb)
        h2 = self.downsample(h1)
        h2 = self.middle(h2, time_emb)
        h3 = self.upsample(h2)
        if h3.shape[-2:] != h1.shape[-2:]:
            h3 = F.interpolate(h3, size=h1.shape[-2:], mode="nearest")
        # Skip connection lets the output keep fine spatial detail.
        # skip connectionにより、出力が細かい空間情報を保ちやすくなる。
        h = self.up_block(h3 + h1, time_emb)
        return self.output(h)


def patchify(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Convert images (B, C, H, W) into flattened patches (B, N, P*P*C)."""

    if x.shape[-1] % patch_size != 0 or x.shape[-2] % patch_size != 0:
        raise ValueError("image height and width must be divisible by patch_size")
    # unfold returns (B, patch_dim, num_patches); transpose to transformer style.
    # unfoldは(B, patch_dim, num_patches)を返すので、Transformer風に転置する。
    patches = F.unfold(x, kernel_size=patch_size, stride=patch_size)
    return patches.transpose(1, 2)


def unpatchify(
    patches: torch.Tensor,
    image_size: tuple[int, int],
    patch_size: int,
    channels: int,
) -> torch.Tensor:
    """Convert flattened patches (B, N, P*P*C) back to images."""

    expected_dim = patch_size * patch_size * channels
    if patches.shape[-1] != expected_dim:
        raise ValueError(f"patch dim must be {expected_dim}, got {patches.shape[-1]}")
    # fold is the inverse layout operation of unfold when patches do not overlap.
    # patchが重ならない設定では、foldはunfoldのレイアウトを戻す操作になる。
    patches = patches.transpose(1, 2)
    return F.fold(patches, output_size=image_size, kernel_size=patch_size, stride=patch_size)


class TinyDiTVelocity(nn.Module):
    """Tiny DiT-style image velocity model for shape-reading exercises.

    This is not a production DiT. It shows the essential path:
        image -> patch tokens -> time-conditioned Transformer -> image velocity

    日本語:
        教材用の最小DiT風速度場モデル。画像をpatch列にし、時刻埋め込みを足し、
        Transformerで処理してから画像shapeのvelocityへ戻す。
    """

    def __init__(
        self,
        image_size: tuple[int, int] = (32, 32),
        in_channels: int = 3,
        patch_size: int = 4,
        embed_dim: int = 128,
        depth: int = 2,
        num_heads: int = 4,
        time_dim: int = 128,
    ) -> None:
        super().__init__()
        height, width = image_size
        if height % patch_size != 0 or width % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")

        self.image_size = image_size
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.patch_dim = patch_size * patch_size * in_channels
        self.num_patches = (height // patch_size) * (width // patch_size)

        self.patch_proj = nn.Linear(self.patch_dim, embed_dim)
        self.time_embedding = SinusoidalTimeEmbedding(time_dim)
        self.time_proj = nn.Linear(time_dim, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=4 * embed_dim,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.out_proj = nn.Linear(embed_dim, self.patch_dim)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if t.ndim == 1:
            t = t[:, None]
        patches = patchify(x, self.patch_size)
        # Convert each flattened patch into a token.
        # 平坦化した各patchをTransformer用tokenへ変換する。
        tokens = self.patch_proj(patches)
        # Add time conditioning and learned position information to every token.
        # 各tokenに時刻条件と位置埋め込みを足す。
        time_token = self.time_proj(self.time_embedding(t))[:, None, :]
        tokens = tokens + time_token + self.pos_embed
        tokens = self.transformer(tokens)
        out_patches = self.out_proj(tokens)
        # The model output must have the same image shape as the target velocity.
        # 出力は教師速度u_tと同じ画像shapeに戻す必要がある。
        return unpatchify(out_patches, self.image_size, self.patch_size, self.in_channels)
