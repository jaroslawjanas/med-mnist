import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

###########################################
###### Simple Convolutional Network #######
###########################################

class SimpleCNN(nn.Module):
    
    def __init__(self, in_channels, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, in_channels, padding=1)
        self.bn1 = nn.BatchNorm2d(16)          # Batch Normalization after first conv layer
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)          # Batch Normalization after second conv layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(32 * 8 * 8, 128)  # 32 channels * 8x8 feature map
        self.dropout = nn.Dropout(0.5)         # Dropout before first fully connected layer
        self.fc2 = nn.Linear(128, num_classes)          # Output layer for 10 classes

    def forward(self, x):
        # Convolutional layers with ReLU, BatchNorm, and MaxPooling
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        
        # Flatten the feature maps
        x = x.view(-1, 32 * 8 * 8)
        
        # Fully connected layers with ReLU and Dropout
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)  # Apply dropout only during training
        x = self.fc2(x)
        
        return x
    
###########################################
########## Convolutional Network ##########
###########################################

class ConvNet(nn.Module):
    """ConvNet implementation in PyTorch."""
    def __init__(self, in_channels, num_classes, scales, filters, filters_max, pooling=F.max_pool2d):
        """
        Args:
            in_channels: Number of channels in the input image.
            num_classes: Number of output classes.
            scales: Number of pooling layers, each reducing spatial dimensions by 2.
            filters: Base number of convolution filters.
                     Number of filters doubles at each scale, capped at filters_max.
            filters_max: Maximum number of filters.
            pooling: Type of pooling function (default: max pooling).
        """
        super().__init__()
        self.pooling = pooling

        def nf(scale):
            """Helper function to calculate the number of filters at a given scale."""
            return min(filters_max, filters * (2 ** scale))

        layers = []
        # Initial convolution layer
        layers.append(nn.Conv2d(in_channels, nf(0), kernel_size=3, padding=1))
        layers.append(nn.LeakyReLU(inplace=True))

        # Add scales with increasing filters
        for i in range(scales):
            layers.append(nn.Conv2d(nf(i), nf(i), kernel_size=3, padding=1))
            layers.append(nn.LeakyReLU(inplace=True))
            layers.append(nn.Conv2d(nf(i), nf(i + 1), kernel_size=3, padding=1))
            layers.append(nn.LeakyReLU(inplace=True))
            layers.append(PoolingLayer(self.pooling, kernel_size=2, stride=2))  # Custom pooling layer

        # Final convolution layer and mean reduction
        layers.append(nn.Conv2d(nf(scales), num_classes, kernel_size=3, padding=1))
        layers.append(MeanReduceLayer())  # Custom mean reduction layer

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass through the network."""
        return self.layers(x)


class PoolingLayer(nn.Module):
    """Wrapper for pooling operations to integrate with nn.Sequential."""
    def __init__(self, pooling_fn, kernel_size, stride):
        super().__init__()
        self.pooling_fn = pooling_fn
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        return self.pooling_fn(x, kernel_size=self.kernel_size, stride=self.stride)


class MeanReduceLayer(nn.Module):
    """Custom layer for mean reduction over spatial dimensions."""
    def forward(self, x):
        return x.mean(dim=(2, 3))

###########################################
########## Wide Residual Network ##########
###########################################

# This implementation is AI conversion of what JAX's default WRN is

class WRNBlock(nn.Module):
    """WideResNet Block"""
    def __init__(self, in_channels, out_channels, stride=1, bn_momentum=0.9, bn_eps=1e-5):
        super().__init__()
        self.proj_conv = None
        if in_channels != out_channels or stride > 1:
            self.proj_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        
        self.bn1 = nn.BatchNorm2d(in_channels, momentum=bn_momentum, eps=bn_eps)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels, momentum=bn_momentum, eps=bn_eps)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        shortcut = self.proj_conv(x) if self.proj_conv else x
        out = F.relu(self.bn1(x))
        out = self.conv1(out)
        out = F.relu(self.bn2(out))
        out = self.conv2(out)
        return out + shortcut


class WideResNet(nn.Module):
    """WideResNet"""
    def __init__(self, in_channels, num_classes, depth=28, width=2, bn_momentum=0.9, bn_eps=1e-5):
        super().__init__()
        assert (depth - 4) % 6 == 0, "Depth must be 6n+4"
        n = (depth - 4) // 6
        block = WRNBlock

        widths = [16, 16 * width, 32 * width, 64 * width]

        self.conv1 = nn.Conv2d(in_channels, widths[0], kernel_size=3, stride=1, padding=1, bias=False)

        self.group1 = self._make_group(block, widths[0], widths[1], n, stride=1, bn_momentum=bn_momentum, bn_eps=bn_eps)
        self.group2 = self._make_group(block, widths[1], widths[2], n, stride=2, bn_momentum=bn_momentum, bn_eps=bn_eps)
        self.group3 = self._make_group(block, widths[2], widths[3], n, stride=2, bn_momentum=bn_momentum, bn_eps=bn_eps)

        self.bn = nn.BatchNorm2d(widths[3], momentum=bn_momentum, eps=bn_eps)
        self.fc = nn.Linear(widths[3], num_classes)

    def _make_group(self, block, in_channels, out_channels, num_blocks, stride, bn_momentum, bn_eps):
        layers = [block(in_channels, out_channels, stride, bn_momentum, bn_eps)]
        for _ in range(1, num_blocks):
            layers.append(block(out_channels, out_channels, 1, bn_momentum, bn_eps))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.group1(out)
        out = self.group2(out)
        out = self.group3(out)
        out = F.relu(self.bn(out))
        out = F.adaptive_avg_pool2d(out, 1)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out
    
###########################################
########### Vision Transformer ############
###########################################

class Patchify(nn.Module):
    def __init__(self, img_size, patch_size):
        super(Patchify, self).__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

    def forward(self, x):
        B, C, H, W = x.shape
        patch_dim = C * self.patch_size * self.patch_size
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x = x.permute(0, 2, 3, 1, 4, 5).reshape(B, -1, patch_dim)
        return x

class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, projection_dim):
        super(PatchEmbedding, self).__init__()
        self.patchify = Patchify(img_size, patch_size)
        self.projection = nn.Linear(in_channels * patch_size * patch_size, projection_dim)

    def forward(self, x):
        x = self.patchify(x)  # (B, num_patches, patch_dim)
        x = self.projection(x)  # (B, num_patches, projection_dim)
        return x

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, projection_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = projection_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(projection_dim, projection_dim * 3)
        self.fc_out = nn.Linear(projection_dim, projection_dim)

    def forward(self, x):
        B, N, D = x.shape
        qkv = self.qkv(x).chunk(3, dim=-1)  # (B, N, projection_dim) -> 3 * (B, N, projection_dim)
        queries, keys, values = map(
            lambda t: t.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2),
            qkv
        )  # (B, num_heads, N, head_dim)

        attention = (queries @ keys.transpose(-2, -1)) * self.scale  # (B, num_heads, N, N)
        attention = attention.softmax(dim=-1)

        out = (attention @ values).transpose(1, 2).reshape(B, N, D)  # (B, N, projection_dim)
        return self.fc_out(out)

class TransformerBlock(nn.Module):
    def __init__(self, projection_dim, num_heads, mlp_ratio, dropout):
        super(TransformerBlock, self).__init__()
        self.attn = MultiHeadSelfAttention(projection_dim, num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(projection_dim, int(projection_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(projection_dim * mlp_ratio), projection_dim)
        )
        self.norm1 = nn.LayerNorm(projection_dim)
        self.norm2 = nn.LayerNorm(projection_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout(self.attn(self.norm1(x)))
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x

class VisionTransformer(nn.Module):
    def __init__(
            self, 
            image_size: int,
            patch_size: int,
            in_channels: int,
            num_classes: int,
            num_datasets:  int,
            projection_dim: int,
            depth: int,
            num_heads: int,
            mlp_ratio: int,
            dropout: float
    ):
        super(VisionTransformer, self).__init__()
        self.patch_embed = PatchEmbedding(image_size, patch_size, in_channels, projection_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, projection_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, (image_size // patch_size) ** 2 + 1, projection_dim))
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.Sequential(
            *[TransformerBlock(projection_dim, num_heads, mlp_ratio, dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(projection_dim)
        self.fc = nn.Linear(projection_dim, num_classes)

        # Two separate heads:
        # 1. Head for predicting the global class (multi-class classification)
        self.fc_class = nn.Linear(projection_dim, num_classes)
        # 2. Head for predicting the dataset ID (for multi-task learning)
        self.fc_dataset = nn.Linear(projection_dim, num_datasets)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        # Expand the class token to batch size and concatenate with patch embeddings
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, projection_dim)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, num_patches+1, projection_dim)
        x = x + self.pos_embed
        x = self.dropout(x)

        x = self.blocks(x)
        x = self.norm(x)
        # Use the class token representation for prediction (first token)
        rep = x[:, 0]  # shape: (B, projection_dim)
        out_class = self.fc_class(rep)      #  Class prediction
        out_dataset = self.fc_dataset(rep)    # Dataset prediction
        
        return out_class, out_dataset
    
###########################################
######### ARCHITECTURE SELECTOR ###########
###########################################
    
def architecture(arch: str):
    if arch == "simple-cnn":
        return partial(SimpleCNN)
    elif arch == "cnn16-3-max":
        return partial(ConvNet, scales=3, filters=16, filters_max=1024, pooling=F.max_pool2d)
    elif arch == "cnn16-3-mean":
        return partial(ConvNet, scales=3, filters=16, filters_max=1024, pooling=F.avg_pool2d)
    elif arch == "cnn32-3-max":
        return partial(ConvNet, scales=3, filters=32, filters_max=1024, pooling=F.max_pool2d)
    elif arch == "cnn32-3-mean":
        return partial(ConvNet, scales=3, filters=32, filters_max=1024, pooling=F.avg_pool2d)
    elif arch == "cnn64-3-max":
        return partial(ConvNet, scales=3, filters=64, filters_max=1024, pooling=F.max_pool2d)
    elif arch == "cnn64-3-mean":
        return partial(ConvNet, scales=3, filters=64, filters_max=1024, pooling=F.avg_pool2d)
    elif arch == "wrn28-1":
        return partial(WideResNet, depth=28, width=1)
    elif arch == "wrn28-2":
        return partial(WideResNet, depth=28, width=2)
    elif arch == "wrn28-10":
        return partial(WideResNet, depth=28, width=10)
    elif arch == "vit":
        return partial(VisionTransformer)
    else:
        raise ValueError(f"Architecture '{arch}' not recognized.")
