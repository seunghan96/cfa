import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.SelfAttention_Family import CrossModalAttentionLayer, FullAttention


class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=2,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1, 2)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu",
                 text_embedding_dim=None, injection_mode='last', n_heads=None, cfa_reduction=8):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        
        # Text injection support for middle injection
        self.has_text = text_embedding_dim is not None
        self.injection_mode = injection_mode

        

        if injection_mode == 'gating':
            self.text_proj_gated = nn.Linear(text_embedding_dim, d_model, bias=False)
            self.gate_network = nn.Sequential(
                nn.Linear(d_model, d_ff, bias=True),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model, bias=True),
                nn.Sigmoid()  # Gate values between 0 and 1
            )
        

        if injection_mode == 'cfa':
            adapter_bottleneck = max(1, d_model // cfa_reduction)
            
            # Text -> bottleneck -> d_model (residual adapter)
            self.adapter_down = nn.Linear(text_embedding_dim, adapter_bottleneck, bias=False)
            self.adapter_up = nn.Linear(adapter_bottleneck, d_model, bias=False)
            
            # Optional: Layer norm for adapter output
            self.adapter_norm = nn.LayerNorm(adapter_bottleneck)
            self.adapter_activation = nn.ReLU()
        
        if injection_mode == 'film':
            self.text_proj_gamma_film = nn.Linear(text_embedding_dim, d_model, bias=True)
            self.text_proj_beta_film = nn.Linear(text_embedding_dim, d_model, bias=True)
            # Initialize bias: gamma=1 (scale), beta=0 (bias)
            nn.init.ones_(self.text_proj_gamma_film.bias)
            nn.init.zeros_(self.text_proj_beta_film.bias)

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

    def forward(self, x, attn_mask=None, tau=None, delta=None, text_context=None):
        # Self-attention
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        


        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        x = self.norm2(x + y)
        
        if text_context is not None:
            def _project_text(projector):
                if projector is None:
                    return None
                if len(text_context.shape) == 2:
                    projected = projector(text_context)
                    return projected.unsqueeze(1).expand(-1, x.shape[1], -1)
                elif len(text_context.shape) == 3:
                    projected_seq = projector(text_context)
                    return projected_seq.mean(dim=1, keepdim=True).expand(-1, x.shape[1], -1)
                else:
                    return None
            
            if self.injection_mode == 'middle-additive' and self.text_proj_middle is not None:
                text_emb = _project_text(self.text_proj_middle)
                if text_emb is not None:
                    x = x + text_emb
            elif self.injection_mode == 'middle-concat' and self.middle_concat_proj is not None:
                text_emb = _project_text(self.text_proj_middle)
                if text_emb is not None:
                    x = self.middle_concat_proj(torch.cat([x, text_emb], dim=-1))
            elif self.injection_mode == 'gating':
                text_cond = _project_text(self.text_proj_gated)
                if text_cond is not None:
                    gate = self.gate_network(x)
                    x = x + gate * text_cond
            elif self.injection_mode == 'cfa':
                pooled_text = text_context if len(text_context.shape) == 2 else text_context.mean(dim=1)
                adapter_hidden = self.adapter_down(pooled_text)
                adapter_hidden = self.adapter_norm(adapter_hidden)
                adapter_hidden = self.adapter_activation(adapter_hidden)
                adapter_out = self.adapter_up(adapter_hidden).unsqueeze(1).expand(-1, x.shape[1], -1)
                x = x + adapter_out
            elif self.injection_mode == 'film':
                gamma = _project_text(self.text_proj_gamma_film)
                beta = _project_text(self.text_proj_beta_film)
                if gamma is not None and beta is not None:
                    x = gamma * x + beta
            elif self.injection_mode == 'orthogonal':
                ortho_text = _project_text(self.text_proj_ortho)
                if ortho_text is not None:
                    proj_coeff = (ortho_text * x).sum(dim=-1, keepdim=True) / (x.pow(2).sum(dim=-1, keepdim=True) + 1e-6)
                    ortho_component = ortho_text - proj_coeff * x
                    x = x + ortho_component
        
        return x, attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None, 
                 injection_mode='last', text_context=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer
        self.injection_mode = injection_mode

    def forward(self, x, attn_mask=None, tau=None, delta=None, text_context=None):
        # x [B, L, D]
        attns = []

        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                delta = delta if i == 0 else None
                # Pass text_context for middle injection and other text-aware modes
                layer_text_ctx = text_context
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta, text_context=layer_text_ctx)
                x = conv_layer(x)
                attns.append(attn)
            layer_text_ctx = text_context
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None, text_context=layer_text_ctx)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                # Pass text_context for middle injection and other text-aware modes
                layer_text_ctx = text_context
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta, text_context=layer_text_ctx)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        # Handle last injection if needed (injection happens after all layers)
        # This will be handled in the model's forward method
        return x, attns  # Return tuple for compatibility


class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu", text_embedding_dim=None, injection_mode='last', n_heads=None):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        



    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None, text_context=None):
        # Self-attention
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask,
            tau=tau, delta=None
        )[0])
        x = self.norm1(x)
        

        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask,
            tau=tau, delta=delta
        )[0])

        y = x = self.norm2(x)
        


        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        output = self.norm3(x + y)
        

        
        return output


class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None, injection_mode='last', text_embedding_dim=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection
        self.injection_mode = injection_mode
        self.has_text = text_embedding_dim is not None
        d_model = norm_layer.normalized_shape[0] if norm_layer is not None else None
        self.text_proj_middle = None
        self.middle_concat_proj = None
        if self.has_text and d_model is not None and injection_mode != 'unimodal':
            if injection_mode in ('middle-additive', 'middle-concat'):
                self.text_proj_middle = nn.Linear(text_embedding_dim, d_model, bias=False)
            if injection_mode == 'middle-concat':
                self.middle_concat_proj = nn.Linear(2 * d_model, d_model, bias=True)

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None, text_context=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta, text_context=text_context)

        if self.norm is not None:
            x = self.norm(x)

        # Middle fusion after norm
        if text_context is not None and (self.text_proj_middle is not None or self.middle_concat_proj is not None):
            if len(text_context.shape) == 2:
                text_emb = self.text_proj_middle(text_context)
            else:
                text_emb = self.text_proj_middle(text_context.mean(dim=1))
            text_emb = text_emb.unsqueeze(1).expand(-1, x.shape[1], -1)
            if self.injection_mode == 'middle-additive':
                x = x + text_emb
            else:
                x = self.middle_concat_proj(torch.cat([x, text_emb], dim=-1))

        if self.projection is not None:
            x = self.projection(x)
        return x
