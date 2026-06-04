import torch 
import torch.nn as nn
import torch.nn.functional as F
import math

class PatchCreation(nn.Module):
    def __init__(self,
                 input_color_channel: int,
                 patch_size: int,
                 embedding_dimensions : int):
        super().__init__()
        self.patch_size = patch_size
        # use Conv2d to simultaneously extract and project patches
        self.patching_conv = nn.Conv2d(
            input_color_channel,        # in_channels  = 3
            embedding_dimensions,       # out_channels = D = 256
            kernel_size=patch_size,     # 16×16 window
            stride=patch_size,          # non-overlapping
            padding=0
        )
        self.flatten = nn.Flatten(start_dim=2)

    def forward(self, x):
        # x: (B, C, H, W)
        image_dimension = x.shape[-1]
        assert image_dimension % self.patch_size == 0, \
            f"Given image dimension {image_dimension} is not divisible by patch size {self.patch_size}"
        if len(x.shape) == 3:  # allow input in (C, H, W) by adding batch dim
            x = x.unsqueeze(0)
        # output: (B, N, D)  where N = (H/P)*(W/P)
        return self.patching_conv(x).flatten(2).permute(0, 2, 1)



class ViTInputLayer(nn.Module):
    def __init__(self, in_channels: int,
                 patch_size: int,
                 image_size: int,
                 embedding_dimensions: int,
                 input_dropout_rate: float = 0.0):
        super().__init__()
        # Patch embedding module
        self.patch_embeddings = PatchCreation(in_channels,
                                              patch_size,
                                              embedding_dimensions)

        # Learnable CLS token
        # CLS token acts as a global summary of the image. After the encoder, its final state is used for classification.
        self.cls_token = nn.Parameter(
            torch.randn(1, 1, embedding_dimensions),
            requires_grad=True
        )

        # Number of patches
        num_patches = (image_size // patch_size) ** 2
        num_tokens = num_patches + 1  # +1 for CLS token

        # Learnable positional embeddings
        self.positional_embeddings = nn.Parameter(
            torch.randn(1, num_tokens, embedding_dimensions),
            requires_grad=True
        )

        # Optional dropout (helps regularization)
        self.dropout = nn.Dropout(p=input_dropout_rate)

    def forward(self, x):
        batch_size = x.shape[0]

        # Step 1: Get patch embeddings
        patch_embeddings = self.patch_embeddings(x)

        # Step 2: Expand CLS token for batch size
        cls_token = self.cls_token.expand(batch_size, -1, -1)

        # Step 3: Concatenate CLS token + patch embeddings
        tokens = torch.concat((cls_token, patch_embeddings), dim=1)

        # Step 4: Add positional embeddings + apply dropout
        return self.dropout(tokens + self.positional_embeddings)


class LayerNormalisation(nn.Module):
    def __init__(self, embed_dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(embed_dim)) # Scale factor
        self.bias = nn.Parameter(torch.zeros(embed_dim)) # Shift factor

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dimension: int, head: int, dropout_rate: float = 0.0):
        super().__init__()
        assert embedding_dimension % head == 0, "Embedding dimension must be divisible by number of heads."
        self.w_q = nn.Linear(embedding_dimension, embedding_dimension)
        self.w_k = nn.Linear(embedding_dimension, embedding_dimension)
        self.w_v = nn.Linear(embedding_dimension, embedding_dimension)
        self.head = head
        self.d_k = embedding_dimension // head
        self.w_o = nn.Linear(embedding_dimension, embedding_dimension)

        self.attention_dropout = nn.Dropout(p=dropout_rate)
        self.proj_dropout = nn.Dropout(p=dropout_rate)


    @staticmethod
    def attention(q, k, v, dropout: nn.Dropout = None):
        d_k = q.shape[-1]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        scores = scores.softmax(dim=-1)
        if dropout is not None:
            scores = dropout(scores)
        return torch.matmul(scores, v), scores

    def forward(self, q, k, v):
        batch_size, num_tokens, _ = q.shape

        # 1. Linear projections + split into heads
        query = self.w_q(q).view(batch_size, num_tokens, self.head, self.d_k).transpose(1, 2)
        key = self.w_k(k).view(batch_size, num_tokens, self.head, self.d_k).transpose(1, 2)
        value = self.w_v(v).view(batch_size, num_tokens, self.head, self.d_k).transpose(1, 2)

        # 2. Scaled dot-product attention
        x, self.attention_score = MultiHeadAttention.attention(query, key, value, self.attention_dropout)

        # 3. Concatenate heads
        x = x.transpose(1, 2).contiguous().view(batch_size, num_tokens, self.head * self.d_k)

        # 4. Output projection
        return self.proj_dropout(self.w_o(x))

class FeedForwardLayer(nn.Module):
    # Applied token-wise after attention. It is a simple 2-layer MLP that increases then decreases dimensionality.
    def __init__(self, d_model: int, d_ff_scale: int = 2, dropout_rate: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff_scale * d_model), # 256 → 1024
            nn.GELU(), # GELU (Gaussian Error Linear Unit) is smoother than ReLU and is used in most modern Transformers
            nn.Dropout(dropout_rate),
            nn.Linear(d_ff_scale * d_model, d_model), # 1024 → 256
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        return self.mlp(x)

class EncoderBlock(nn.Module):
    def __init__(self, embedding_dimensions, heads, attention_dropout_rate, feed_forward_dropout_rate, dff_scale: int = 2):
        super().__init__()
        self.normalisation_stage1 = LayerNormalisation(embedding_dimensions)
        self.mhsa = MultiHeadAttention(embedding_dimensions, heads, attention_dropout_rate)
        self.normalisation_stage2 = LayerNormalisation(embedding_dimensions)
        self.feed_forward_layer = FeedForwardLayer(embedding_dimensions, d_ff_scale=dff_scale, dropout_rate=feed_forward_dropout_rate)

    def forward(self, x):
        residual1 = x
        x = self.normalisation_stage1(x)
        x = self.mhsa(x, x, x) + residual1
        residual2 = x
        return self.feed_forward_layer(self.normalisation_stage2(x)) + residual2



class Encoder(nn.Module):
    # Each encoder block processes the full sequence. After L=6 blocks, every token has "seen" every other token multiple times through the attention mechanism.
    def __init__(self, num_of_encoders, embeddings, dff_scale, heads, attention_dropout_rate, feed_forward_dropout_rate):
        super().__init__()
        self.encoder_stack = nn.ModuleList(
            EncoderBlock(
                embedding_dimensions=embeddings,
                heads=heads,
                dff_scale=dff_scale,
                attention_dropout_rate=attention_dropout_rate,
                feed_forward_dropout_rate=feed_forward_dropout_rate
            ) for _ in range(num_of_encoders)
        )

    def forward(self, x):
        for module in self.encoder_stack:
            x = module(x)
        return x

class ViT(nn.Module):
    def __init__(self, in_channels, image_size, patch_size, number_of_encoder,
                 embeddings, d_ff_scale, heads, input_dropout_rate,
                 attention_dropout_rate, feed_forward_dropout_rate, number_of_classes):
        super().__init__()
        self.input_layer = ViTInputLayer(in_channels, patch_size, image_size, embeddings, input_dropout_rate)
        self.encoder_stack = Encoder(number_of_encoder, embeddings, d_ff_scale, heads,
                                     attention_dropout_rate, feed_forward_dropout_rate)

        # Uses only the CLS token from the encoder output
        self.classification_head = nn.Sequential(
            nn.LayerNorm([embeddings]), # normalise CLS embedding
            nn.Linear(embeddings, number_of_classes) # 256 → 3 (num classes)
        )

    def forward(self, x):
        x = self.input_layer(x)            # Patch embed + CLS + pos embed
        x = self.encoder_stack(x)          # L encoder blocks
        x = x[:, 0, :]                     # Extract CLS token: (B, D)
        return self.classification_head(x) # → (B, num_classes)

        #return self.classification_head(self.encoder_stack(self.input_layer(x))[:, 0, :])






