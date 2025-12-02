import torch.nn as nn
from models.attention import Attention
class TransformerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config['hidden_size']

        ff_hidden_dim = 4 * self.hidden_size

        # Layer norm for attention block
        self.att_norm = nn.LayerNorm(self.hidden_size, elementwise_affine=False, eps=1E-6)
        self.attn_block = Attention(config)
        # Layer norm for mlp block
        self.ff_norm = nn.LayerNorm(self.hidden_size, elementwise_affine=False, eps=1E-6)
        self.mlp_block = nn.Sequential(
            nn.Linear(self.hidden_size, ff_hidden_dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(ff_hidden_dim, self.hidden_size),
        )
        # *****************************************************************************
        # New: Cross-attention for masks
        self.cross_attn = Attention(config)  # Reuse same Attention class
        self.cross_attn_norm = nn.LayerNorm(self.hidden_size, eps=1e-6)
        # *****************************************************************************
        self.adaptive_norm_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.hidden_size, 6 * self.hidden_size, bias=True) # replace 6 with 9 if using cross attention for mask
        )

        ############################
        # DiT Layer Initialization #
        ############################
        nn.init.xavier_uniform_(self.mlp_block[0].weight)
        nn.init.constant_(self.mlp_block[0].bias, 0)
        nn.init.xavier_uniform_(self.mlp_block[-1].weight)
        nn.init.constant_(self.mlp_block[-1].bias, 0)

        nn.init.constant_(self.adaptive_norm_layer[-1].weight, 0)
        nn.init.constant_(self.adaptive_norm_layer[-1].bias, 0)



    def forward(self, x, tv_emb_adaln, volt_cross_emb=None):

        # Split scale/shift params (now 8 chunks)
        params = self.adaptive_norm_layer(tv_emb_adaln).chunk(9, dim=1)
        (pre_self_attn_shift, pre_self_attn_scale,
         post_self_attn_scale, pre_cross_attn_shift,
         pre_cross_attn_scale, post_cross_attn_scale,
         pre_mlp_shift, pre_mlp_scale,post_mlp_scale) = params

        # Self-attention block (unchanged)
        out = x
        attn_norm = (self.att_norm(out)* (1 + pre_self_attn_scale.unsqueeze(1)) + pre_self_attn_shift.unsqueeze(1))
        out = out + post_self_attn_scale.unsqueeze(1) * self.attn_block(attn_norm)

        # New: Cross-attention with volt embeddings
        if volt_cross_emb is not None:
            cross_norm = self.cross_attn_norm(out) * (
                        1 + pre_cross_attn_scale.unsqueeze(1)) + pre_cross_attn_shift.unsqueeze(1)
            volt_cross_emb = volt_cross_emb.unsqueeze(1)
            out = out + post_cross_attn_scale.unsqueeze(1) * self.cross_attn(
                cross_norm,  # Query
                volt_cross_emb,  # Key
                volt_cross_emb  # Value
            )

        # MLP block
        mlp_norm = self.ff_norm(out) * (1 + pre_mlp_scale.unsqueeze(1)) + pre_mlp_shift.unsqueeze(1)
        out = out + post_mlp_scale.unsqueeze(1) * self.mlp_block(mlp_norm)
        out = out + self.mlp_block(mlp_norm)
        return out


    def forward(self, x, tv_emb_adaln, volt_cross_emb=None):

        if volt_cross_emb is not None:
            # Split scale/shift params (now 9 chunks)
            params = self.adaptive_norm_layer(tv_emb_adaln).chunk(9, dim=1)
            (pre_self_attn_shift, pre_self_attn_scale,
             post_self_attn_scale, pre_cross_attn_shift,
             pre_cross_attn_scale, post_cross_attn_scale,
             pre_mlp_shift, pre_mlp_scale,post_mlp_scale) = params

            # Self-attention block
            out = x
            attn_norm = (self.att_norm(out)* (1 + pre_self_attn_scale.unsqueeze(1)) + pre_self_attn_shift.unsqueeze(1))
            out = out + post_self_attn_scale.unsqueeze(1) * self.attn_block(attn_norm)
            # cross attn with volt emb
            cross_norm = self.cross_attn_norm(out) * (
                        1 + pre_cross_attn_scale.unsqueeze(1)) + pre_cross_attn_shift.unsqueeze(1)
            volt_cross_emb = volt_cross_emb.unsqueeze(1)
            out = out + post_cross_attn_scale.unsqueeze(1) * self.cross_attn(
                cross_norm,  # Query
                volt_cross_emb,  # Key
                volt_cross_emb  # Value
            )
        else:
            scale_shift_params = self.adaptive_norm_layer(tv_emb_adaln).chunk(6, dim=1)
            (pre_self_attn_shift, pre_self_attn_scale, post_self_attn_scale,
             pre_mlp_shift, pre_mlp_scale, post_mlp_scale) = scale_shift_params
            # Self-attention block
            out = x
            attn_norm = (self.att_norm(out) * (1 + pre_self_attn_scale.unsqueeze(1)) + pre_self_attn_shift.unsqueeze(1))
            out = out + post_self_attn_scale.unsqueeze(1) * self.attn_block(attn_norm)

        # MLP block
        mlp_norm = self.ff_norm(out) * (1 + pre_mlp_scale.unsqueeze(1)) + pre_mlp_shift.unsqueeze(1)
        out = out + post_mlp_scale.unsqueeze(1) * self.mlp_block(mlp_norm)
        #out = out + self.mlp_block(mlp_norm)
        return out













