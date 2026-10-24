"""
ViT / Transformer 编码器模块
ViTBlock   单个 Transformer block
ViT        多层 ViTBlock 堆叠后的完整模型

by Wang Guozheng
2026.07.02
"""
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from .utils.attention import Attention                  # 自定义注意力模块
from .utils.modules import FeedForward                  # Transformer 里的 MLP / FFN
from .utils.modules import PE_wrapper                   # 位置编码封装


class ViTBlock(nn.Module):
    """ViTBlock: 单个 Transformer block"""
    def __init__(self, dim, context_dim=None,
                 num_heads=8, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, qk_norm='rmsnorm',
                 norm_layer='rmsnorm',
                 rope_mode='none',
                 context_norm=True,
                 use_checkpoint=False):

        super().__init__()
        if norm_layer == 'rmsnorm':
            norm_layer = nn.RMSNorm         # 使用 nn.RMSNorm，MLP 不带 bias
            mlp_bias = False
        elif norm_layer == 'layernorm':
            norm_layer = nn.LayerNorm       # 使用 nn.LayerNorm，MLP 带 bias
            mlp_bias = True
        else:
            raise NotImplementedError

        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim=dim,
                              num_heads=num_heads,
                              qkv_bias=qkv_bias, qk_scale=qk_scale,
                              qk_norm=qk_norm,
                              rope_mode=rope_mode)

        if context_dim is not None:
            self.use_context = True
            self.cross_attn = Attention(dim=dim,
                                        num_heads=num_heads,
                                        context_dim=context_dim,
                                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                                        qk_norm=qk_norm,
                                        rope_mode='none')
            self.norm2 = norm_layer(dim)
            if context_norm:
                self.norm_context = norm_layer(context_dim)
            else:
                self.norm_context = nn.Identity()
        else:
            self.use_context = False

        self.norm3 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = FeedForward(in_features=dim,
                               hidden_size=mlp_hidden_dim,
                               multiple_of=256, bias=mlp_bias)

        self.use_checkpoint = use_checkpoint

    def forward(self, x, context=None,
                x_mask=None, context_mask=None, extras=None):
        if self.use_checkpoint:
            return checkpoint(self._forward, x,
                              context,
                              x_mask, context_mask, extras,
                              use_reentrant=False)
        else:
            return self._forward(x, context,
                                 x_mask, context_mask, extras)

    def _forward(self, x, context=None,
                 x_mask=None, context_mask=None, extras=None):
        # self attention
        x_norm = self.norm1(x)
        x_attn = self.attn(x_norm, context=None,
                           context_mask=x_mask,
                           extras=extras)
        x = x + x_attn

        # cross attention
        if self.use_context:
            assert context is not None
            x = x + self.cross_attn(x=self.norm2(x),
                                    context=self.norm_context(context),
                                    context_mask=context_mask, extras=extras)

        # mlp
        x_norm = self.norm3(x)
        x_mlp = self.mlp(x_norm)
        x = x + x_mlp

        return x


class ViT(nn.Module):
    def __init__(self,
                 in_chans=128,
                 out_chans=None,
                 seq_len=128,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 qk_norm='layernorm',
                 norm_layer='layernorm',
                 context_norm=False,
                 use_checkpoint=False,
                 context_dim=None,
                 context_fusion=None,
                 context_max_length=None,
                 context_pe_method='none',
                 pe_method='none',
                 rope_mode='shared'):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        # input
        self.in_chans = in_chans
        self.in_proj = nn.Linear(in_chans, embed_dim)
        out_chans = in_chans if out_chans is None else out_chans
        self.out_chans = out_chans

        # position embedding
        self.rope = rope_mode
        self.x_pe = PE_wrapper(dim=embed_dim, method=pe_method,
                               length=seq_len)

        print(f'x position embedding: {pe_method}')             # 当前使用的位置编码方式
        print(f'rope mode: {self.rope}')                        # RoPE 模式

        # context
        self.extras = 0
        self.use_context = False
        self.context_cross = False
        self.context_max_length = context_max_length
        self.context_fusion = 'none'
        if context_dim is not None:
            self.use_context = True
            self.context_embed = nn.Sequential(
                nn.Linear(context_dim, embed_dim, bias=True),
                nn.SiLU(),
                nn.Linear(embed_dim, embed_dim, bias=True),)
            self.context_fusion = context_fusion
            self.context_pe = PE_wrapper(dim=embed_dim,
                                         method=context_pe_method,
                                         length=context_max_length)
            if context_fusion == 'concat':                      # 把 context token 拼到 x 前面
                self.extras += context_max_length
                # no cross attention layers
                context_dim = None                              # ViTBlock 就不会创建 cross-attention
            elif context_fusion == 'cross':                     # 在 ViTBlock 里用 cross-attention
                self.context_cross = True
                context_dim = embed_dim                         # ViTBlock 创建 cross-attention
            else:
                raise NotImplementedError
        print(f'context fusion mode: {context_fusion}')
        print(f'context position embedding: {context_pe_method}')

        block_list = []
        for i in range(depth):
            block_list.append(ViTBlock(
                dim=embed_dim, context_dim=context_dim, num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, qk_norm=qk_norm,
                norm_layer=norm_layer,
                rope_mode=self.rope,
                context_norm=context_norm,
                use_checkpoint=use_checkpoint))
        self.blocks = nn.ModuleList(block_list)

        # FinalLayer block
        self.norm = nn.LayerNorm(embed_dim)
        self.final_layer = nn.Linear(embed_dim, out_chans)

    def _concat_x_context(self, x, context, x_mask=None, context_mask=None):
        """
        原 x:      [x1, x2, x3, ...]
        context:   [c1, c2, ...]
        拼接后:    [c1, c2, ..., x1, x2, x3, ...]
        """
        assert context.shape[-2] == self.context_max_length
        # Check if either x_mask or context_mask is provided
        B = x.shape[0]
        # Create default masks if they are not provided
        if x_mask is None:
            x_mask = torch.ones(B, x.shape[-2], device=x.device).bool()
        if context_mask is None:
            context_mask = torch.ones(B, context.shape[-2],
                                      device=context.device).bool()
        # Concatenate the masks along the second dimension (dim=1)
        x_mask = torch.cat([context_mask, x_mask], dim=1).bool()
        # Concatenate context and x along the second dimension (dim=1)
        x = torch.cat([context, x], dim=1)
        return x, x_mask

    def forward(self, x, x_mask=None, context=None, context_mask=None):
        x = self.in_proj(x)                                                         # [B, T, in_chans] -> [B, T, embed_dim]
        x = self.x_pe(x)                                                            # 添加位置编码 [B, T, embed_dim]

        # context 处理                                                              # 此代码不使用context，只是将VIT作为参考新信号的特征提取模块 
        if self.use_context:
            context_token = self.context_embed(context)
            context_token = self.context_pe(context_token)
            if self.context_fusion == 'concat':
                x, x_mask = self._concat_x_context(x=x, context=context_token,
                                                   x_mask=x_mask,
                                                   context_mask=context_mask)
                context_token, context_mask = None, None
        else:
            context_token, context_mask = None, None

        # Transformer blocks
        for block in self.blocks:
            x = block(x, context=context_token, x_mask=x_mask, context_mask=context_mask)

        # 输出层
        x = self.norm(x)                    # [B, T, embed_dim] -> [B, T, embed_dim]
        x = self.final_layer(x)             # [B, T, embed_dim] -> [B, T, out_chans]

        return x


# example
if __name__ == '__main__':
    x = torch.rand(4, 128, 128)

    model = ViT(in_chans=128,       # 输入通道数。代表输入序列中每个元素（token）的特征维度是128
                out_chans=None,     # 输出通道数。模型最终输出的特征维度。如果设为 None，通常会默认等于 in_chans
                seq_len=128,        # 序列长度。输入序列包含的元素（token）数量，这里是128个
                embed_dim=768,      # 嵌入维度。这是 Transformer 模型内部最主要的工作维度。输入数据会首先被投影到这个维度进行后续处理。它决定了模型的大小和容量
                depth=12,           # 模型深度。代表 Transformer 模块（Encoder Block）堆叠的数量
                num_heads=12,       # 注意力头数。在多头自注意力机制中，将计算分解到多少个“头”上并行处理
                mlp_ratio=4.,       # MLP 扩展比例。在每个 Transformer 模块的前馈网络（MLP）部分，中间隐藏层的维度会扩展为 embed_dim * mlp_ratio
                qkv_bias=False,     # QKV 偏置。决定在生成 Query, Key, Value 的线性层中是否使用偏置项（bias）。False 表示不使用
                qk_scale=None,      # QK 缩放因子。在计算注意力分数时，对 Q 和 K 的点积结果进行缩放的系数。如果为 None，通常会默认使用 1 / sqrt(head_dim) 来防止梯度过小
                qk_norm='layernorm',    # QK 归一化。在 Q 和 K 向量计算点积之前，对它们使用的归一化方法。这里指定为层归一化（LayerNorm）
                norm_layer='layernorm', # 归一化层类型。在整个模型中使用的归一化层类型，通常是 LayerNorm
                context_norm=False,
                use_checkpoint=True,    # 使用梯度检查点
                context_dim=None,       # 上下文维度。上下文序列中每个元素的特征维度。None 表示不使用上下文输入
                context_fusion=None,    # 上下文融合方式。指定如何将上下文序列与主序列融合，例如 'concat' (拼接) 或 'cross' (交叉注意力)。None 表示不融合
                context_max_length=None,    # 上下文最大长度。上下文序列的最大长度
                context_pe_method='none',   # 上下文归一化。是否在交叉注意力中对上下文信息进行额外的归一化
                pe_method='none',       # 位置嵌入方法。为主输入序列指定位置编码的方法。'none' 表示不使用标准的位置嵌入，这通常意味着模型将依赖 RoPE
                rope_mode='shared')     # RoPE 模式。RoPE (旋转位置嵌入) 是一种高效的位置编码方式。'shared' 可能表示 Query 和 Key 共享相同的旋转矩阵

    y = model(x)

    total_params = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (total_params / 1e6))