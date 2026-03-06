import torch
import torch.nn as nn
from einops import rearrange, repeat
from layers.SelfAttention_Family import TwoStageAttentionLayer


class SegMerging(nn.Module):
    def __init__(self, d_model, win_size, norm_layer=nn.LayerNorm):
        super().__init__()
        self.d_model = d_model
        self.win_size = win_size
        self.linear_trans = nn.Linear(win_size * d_model, d_model)
        self.norm = norm_layer(win_size * d_model)

    def forward(self, x):
        batch_size, ts_d, seg_num, d_model = x.shape
        pad_num = seg_num % self.win_size
        if pad_num != 0:
            pad_num = self.win_size - pad_num
            x = torch.cat((x, x[:, :, -pad_num:, :]), dim=-2)

        seg_to_merge = []
        for i in range(self.win_size):
            seg_to_merge.append(x[:, :, i::self.win_size, :])
        x = torch.cat(seg_to_merge, -1)

        x = self.norm(x)
        x = self.linear_trans(x)

        return x


class scale_block(nn.Module):
    def __init__(self, configs, win_size, d_model, n_heads, d_ff, depth, dropout, \
                 seg_num=10, factor=10, text_embedding_dim=None, injection_mode='last'):
        super(scale_block, self).__init__()

        if win_size > 1:
            self.merge_layer = SegMerging(d_model, win_size, nn.LayerNorm)
        else:
            self.merge_layer = None

        self.encode_layers = nn.ModuleList()

        for i in range(depth):
            self.encode_layers.append(TwoStageAttentionLayer(configs, seg_num, factor, d_model, n_heads, \
                                                             d_ff, dropout, text_embedding_dim=text_embedding_dim,
                                                             injection_mode=injection_mode))

    def forward(self, x, attn_mask=None, tau=None, delta=None, text_context=None):
        _, ts_dim, _, _ = x.shape

        if self.merge_layer is not None:
            x = self.merge_layer(x)

        for layer in self.encode_layers:
            x = layer(x, attn_mask=attn_mask, tau=tau, delta=delta, text_context=text_context)

        return x, None


class Encoder(nn.Module):
    def __init__(self, attn_layers, injection_mode='last'):
        super(Encoder, self).__init__()
        self.encode_blocks = nn.ModuleList(attn_layers)
        self.injection_mode = injection_mode

    def forward(self, x, text_context=None):
        encode_x = []
        encode_x.append(x)
        

        for block in self.encode_blocks:
            x, attns = block(x, text_context=text_context)
            encode_x.append(x)

        return encode_x, None


class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, seg_len, d_model, d_ff=None, dropout=0.1,
                 text_embedding_dim=None, injection_mode='last', n_heads=None, cfa_reduction=8):
        super(DecoderLayer, self).__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.MLP1 = nn.Sequential(nn.Linear(d_model, d_model),
                                  nn.GELU(),
                                  nn.Linear(d_model, d_model))
        self.linear_pred = nn.Linear(d_model, seg_len)
        
        # Text injection for middle-layer normalization
        self.has_text = text_embedding_dim is not None
        self.injection_mode = injection_mode
        d_ff = d_ff or 4 * d_model
        
        # CMAF: Cross-Modal Attention Fusion
        if self.has_text and (injection_mode == 'cmaf' or injection_mode == 'first-cmaf' or 
                             injection_mode == 'cmaf-last' or injection_mode == 'first-cmaf-last'):
            from layers.SelfAttention_Family import CrossModalAttentionLayer, FullAttention
            if n_heads is None:
                if hasattr(cross_attention, 'n_heads'):
                    n_heads = cross_attention.n_heads
                else:
                    n_heads = 8
            cross_modal_attn = FullAttention(mask_flag=False, factor=5, attention_dropout=dropout, output_attention=False)
            self.cross_modal_attention = CrossModalAttentionLayer(
                cross_modal_attn, d_model, text_embedding_dim, n_heads
            )
            self.norm_cross = nn.LayerNorm(d_model)
        else:
            self.cross_modal_attention = None
            self.norm_cross = None
            
        if self.has_text and (injection_mode == 'middle-layer' or injection_mode == 'middle-inst-layer' or 
                             injection_mode == 'first-middle-layer' or injection_mode == 'middle-layer-last' or 
                             injection_mode == 'first-middle-layer-last'):
            self.text_proj_alpha1 = nn.Linear(text_embedding_dim, d_model, bias=True)
            self.text_proj_beta1 = nn.Linear(text_embedding_dim, d_model, bias=True)
            self.text_proj_alpha2 = nn.Linear(text_embedding_dim, d_model, bias=True)
            self.text_proj_beta2 = nn.Linear(text_embedding_dim, d_model, bias=True)
            # Initialize biases to produce default values (alpha=1, beta=0)
            nn.init.ones_(self.text_proj_alpha1.bias)
            nn.init.zeros_(self.text_proj_alpha2.bias)
            nn.init.zeros_(self.text_proj_beta1.bias)
            nn.init.zeros_(self.text_proj_beta2.bias)
        
        # TAN: Text-Guided Adaptive Normalization
        if self.has_text and (injection_mode == 'tan' or injection_mode == 'first-tan' or 
                             injection_mode == 'tan-last' or injection_mode == 'first-tan-last'):
            self.text_proj_gamma1 = nn.Linear(text_embedding_dim, d_model, bias=True)
            self.text_proj_beta1_tan = nn.Linear(text_embedding_dim, d_model, bias=True)
            self.text_proj_gamma2 = nn.Linear(text_embedding_dim, d_model, bias=True)
            self.text_proj_beta2_tan = nn.Linear(text_embedding_dim, d_model, bias=True)
            nn.init.ones_(self.text_proj_gamma1.bias)
            nn.init.ones_(self.text_proj_gamma2.bias)
            nn.init.zeros_(self.text_proj_beta1_tan.bias)
            nn.init.zeros_(self.text_proj_beta2_tan.bias)
        
        # GCMF: Gated Cross-Modal Fusion
        if self.has_text and (injection_mode == 'gcmf' or injection_mode == 'first-gcmf' or 
                             injection_mode == 'gcmf-last' or injection_mode == 'first-gcmf-last'):
            self.text_proj_gcmf = nn.Linear(text_embedding_dim, d_model, bias=False)
            self.gate_mlp = nn.Sequential(
                nn.Linear(d_model * 2, d_ff, bias=True),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model, bias=True),
                nn.Sigmoid()
            )
            self.fuse_mlp = nn.Sequential(
                nn.Linear(d_model * 2, d_ff, bias=True),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model, bias=True)
            )
        
        # TCMoE: Text-Conditioned Mixture of Experts
        if self.has_text and (injection_mode == 'tcmoe' or injection_mode == 'first-tcmoe' or 
                             injection_mode == 'tcmoe-last' or injection_mode == 'first-tcmoe-last'):
            self.num_experts = 4
            activation_module = nn.GELU()
            self.experts = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(d_model, d_ff),
                    activation_module,
                    nn.Dropout(dropout),
                    nn.Linear(d_ff, d_model),
                    nn.Dropout(dropout)
                ) for _ in range(self.num_experts)
            ])
            self.router = nn.Sequential(
                nn.Linear(text_embedding_dim, d_model, bias=True),
                nn.ReLU(),
                nn.Linear(d_model, self.num_experts, bias=True),
                nn.Softmax(dim=-1)
            )
        
        # Gated Additive
        if injection_mode == 'gating':                    
            self.text_proj_gated = nn.Linear(text_embedding_dim, d_model, bias=False)
            self.gate_network = nn.Sequential(
                nn.Linear(d_model, d_ff, bias=True),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model, bias=True),
                nn.Sigmoid()
            )
        
        # Residual Adapter
        if injection_mode == 'cfa':        
            adapter_bottleneck = max(1, d_model // cfa_reduction)
            self.adapter_down = nn.Linear(text_embedding_dim, adapter_bottleneck, bias=False)
            self.adapter_norm = nn.LayerNorm(adapter_bottleneck)
            self.adapter_activation = nn.ReLU()
            self.adapter_up = nn.Linear(adapter_bottleneck, d_model, bias=False)
        
        # FiLM
        if injection_mode == 'film':            
            self.text_proj_gamma_film = nn.Linear(text_embedding_dim, d_model, bias=True)
            self.text_proj_beta_film = nn.Linear(text_embedding_dim, d_model, bias=True)
            nn.init.ones_(self.text_proj_gamma_film.bias)
            nn.init.zeros_(self.text_proj_beta_film.bias)
        
        # Orthogonal
        if injection_mode == 'orthogonal':
            self.text_proj_ortho = nn.Linear(text_embedding_dim, d_model, bias=False)

        # Middle-additive / middle-concat
        if self.has_text and injection_mode in ('middle-additive', 'middle-concat'):
            self.text_proj_middle = nn.Linear(text_embedding_dim, d_model, bias=False)
        else:
            self.text_proj_middle = None
        if injection_mode == 'middle-concat' and self.has_text:
            self.middle_concat_proj = nn.Linear(2 * d_model, d_model, bias=True)
        else:
            self.middle_concat_proj = None

    def forward(self, x, cross, text_context=None):
        batch = x.shape[0]
        x = self.self_attention(x)
        x = rearrange(x, 'b ts_d out_seg_num d_model -> (b ts_d) out_seg_num d_model')

        cross = rearrange(cross, 'b ts_d in_seg_num d_model -> (b ts_d) in_seg_num d_model')
        tmp, attn = self.cross_attention(x, cross, cross, None, None, None,)
        x = x + self.dropout(tmp)
        y = x = self.norm1(x)
        

        y = self.MLP1(y)
        
        dec_output = self.norm2(x + y)
        
        if text_context is not None:
            def _project_text(projector):
                if projector is None:
                    return None
                if len(text_context.shape) == 2:
                    projected = projector(text_context)
                    return projected.unsqueeze(1).expand(-1, dec_output.shape[1], -1)
                elif len(text_context.shape) == 3:
                    projected_seq = projector(text_context)
                    return projected_seq.mean(dim=1, keepdim=True).expand(-1, dec_output.shape[1], -1)
                else:
                    return None

            if self.injection_mode == 'middle-additive' and self.text_proj_middle is not None:
                text_emb = _project_text(self.text_proj_middle)
                if text_emb is not None:
                    dec_output = dec_output + text_emb
            elif self.injection_mode == 'middle-concat' and self.middle_concat_proj is not None:
                text_emb = _project_text(self.text_proj_middle)
                if text_emb is not None:
                    dec_output = self.middle_concat_proj(torch.cat([dec_output, text_emb], dim=-1))
            elif self.injection_mode == 'gating':
                text_cond = _project_text(self.text_proj_gated)
                if text_cond is not None:
                    gate = self.gate_network(dec_output)
                    dec_output = dec_output + gate * text_cond
            elif self.injection_mode == 'cfa':
                pooled_text = text_context if len(text_context.shape) == 2 else text_context.mean(dim=1)
                adapter_hidden = self.adapter_down(pooled_text)
                adapter_hidden = self.adapter_norm(adapter_hidden)
                adapter_hidden = self.adapter_activation(adapter_hidden)
                adapter_out = self.adapter_up(adapter_hidden).unsqueeze(1).expand(-1, dec_output.shape[1], -1)
                dec_output = dec_output + adapter_out
            elif self.injection_mode == 'film':
                gamma = _project_text(self.text_proj_gamma_film)
                beta = _project_text(self.text_proj_beta_film)
                if gamma is not None and beta is not None:
                    dec_output = gamma * dec_output + beta
            elif self.injection_mode == 'orthogonal':
                ortho_text = _project_text(self.text_proj_ortho)
                if ortho_text is not None:
                    proj_coeff = (ortho_text * dec_output).sum(dim=-1, keepdim=True) / (dec_output.pow(2).sum(dim=-1, keepdim=True) + 1e-6)
                    ortho_component = ortho_text - proj_coeff * dec_output
                    dec_output = dec_output + ortho_component

        dec_output = rearrange(dec_output, '(b ts_d) seg_dec_num d_model -> b ts_d seg_dec_num d_model', b=batch)
        layer_predict = self.linear_pred(dec_output)
        layer_predict = rearrange(layer_predict, 'b out_d seg_num seg_len -> b (out_d seg_num) seg_len')

        return dec_output, layer_predict


class Decoder(nn.Module):
    def __init__(self, layers, injection_mode='last'):
        super(Decoder, self).__init__()
        self.decode_layers = nn.ModuleList(layers)
        self.injection_mode = injection_mode

    def forward(self, x, cross, text_context=None):
        final_predict = None
        i = 0
        
        ts_d = x.shape[1]
        for layer in self.decode_layers:
            cross_enc = cross[i]
            x, layer_predict = layer(x, cross_enc, text_context=text_context)
            if final_predict is None:
                final_predict = layer_predict
            else:
                final_predict = final_predict + layer_predict
            i += 1

        final_predict = rearrange(final_predict, 'b (out_d seg_num) seg_len -> b (seg_num seg_len) out_d', out_d=ts_d)

        return final_predict
