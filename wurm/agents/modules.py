import torch
from torch import nn
import torch.nn.functional as F
import math
from torch.nn import functional as F


class AddCoords(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        """
        Args:
            input_tensor: shape(batch, channel, x_dim, y_dim)
        """
        batch_size, _, x_dim, y_dim = input_tensor.size()

        xx_channel = torch.arange(x_dim).repeat(1, y_dim, 1)
        yy_channel = torch.arange(y_dim).repeat(1, x_dim, 1).transpose(1, 2)

        xx_channel = xx_channel.float() / (x_dim - 1)
        yy_channel = yy_channel.float() / (y_dim - 1)

        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        xx_channel = xx_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
        yy_channel = yy_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)

        ret = torch.cat([
            input_tensor,
            xx_channel.type_as(input_tensor),
            yy_channel.type_as(input_tensor)], dim=1)

        return ret


class CoordConv2D(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.addcoords = AddCoords()
        in_size = in_channels+2
        self.conv = nn.Conv2d(in_size, out_channels, **kwargs)

    def forward(self, x):
        ret = self.addcoords(x)
        ret = self.conv(ret)
        return ret


class MultiHeadDotProductAttention(nn.Module):

    def __init__(self, num_heads: int, input_dim: int, output_dim: int):
        super().__init__()

        if input_dim % num_heads != 0:
            raise ValueError('Number of num_heads must divide')

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.per_head_dim = output_dim // num_heads
        self.num_heads = num_heads

        self.q_linear = nn.Linear(input_dim, output_dim)
        self.v_linear = nn.Linear(input_dim, output_dim)
        self.k_linear = nn.Linear(input_dim, output_dim)
        self.out = nn.Linear(output_dim, output_dim)

        self.layer_norm = nn.LayerNorm(normalized_shape=self.per_head_dim, elementwise_affine=False)

    def attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, d_k: int):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

        scores = F.softmax(scores, dim=-1)

        output = torch.matmul(scores, v)
        return output

    def forward(self, x: torch.Tensor):
        batch_size = x.size(0)

        # Calculate queries, keys, values and split into num_heads
        k = self.layer_norm(self.k_linear(x).view(batch_size, -1, self.num_heads, self.per_head_dim))
        q = self.layer_norm(self.q_linear(x).view(batch_size, -1, self.num_heads, self.per_head_dim))
        v = self.layer_norm(self.v_linear(x).view(batch_size, -1, self.num_heads, self.per_head_dim))

        # Transpose to get dimensions batch_size * num_heads * sequence_length * input_dim
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = self.attention(q, k, v, self.per_head_dim)

        # Concatenate num_heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous() \
            .view(batch_size, -1, self.output_dim)

        output = self.out(concat)

        return output


class RelationalModule2D(nn.Module):
    """Implements the relational module from https://arxiv.org/pdf/1806.01830.pdf"""
    def __init__(self,
                 num_heads: int,
                 input_dim: int,
                 output_dim: int,
                 residual: bool,
                 add_coords: bool = True):
        super().__init__()
        if add_coords:
            self.addcoords = AddCoords()
            input_dim = input_dim + 2
        self.attention = MultiHeadDotProductAttention(num_heads, input_dim, output_dim)
        self.residual = residual

    def forward(self, x: torch.Tensor):
        identity = x
        n, c, h, w = x.size()

        if hasattr(self, 'addcoords'):
            x = self.addcoords(x)
            c += 2

        # Unroll the 2D image tensor to a sequence so it can be fed to
        # the attention module then return to original shape
        out = x.view(n, c, h*w).transpose(1, 2)  # n, h*w, c
        out = self.attention(out)
        out = out.transpose(2, 1).view(n, self.attention.output_dim, h, w)

        if self.residual:
            out += identity

        return out


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, residual: bool, add_coords: bool = True):
        super(ConvBlock, self).__init__()
        self.residual = residual
        if residual:
            assert in_channels == out_channels
        self.conv = CoordConv2D(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv(x)
        out = F.relu(out)

        if self.residual:
            out += identity

        return out


def feedforward_block(input_dim: int, output_dim: int):
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.ReLU()
    )
