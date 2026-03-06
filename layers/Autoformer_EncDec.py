import torch
import torch.nn as nn
import torch.nn.functional as F


class my_Layernorm(nn.Module):
    """
    Special designed layernorm for the seasonal part
    """

    def __init__(self, channels):
        super(my_Layernorm, self).__init__()
        self.layernorm = nn.LayerNorm(channels)

    def forward(self, x):
        x_hat = self.layernorm(x)
        bias = torch.mean(x_hat, dim=1).unsqueeze(1).repeat(1, x.shape[1], 1)
        return x_hat - bias


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class series_decomp_multi(nn.Module):
    """
    Multiple Series decomposition block from FEDformer
    """

    def __init__(self, kernel_size):
        super(series_decomp_multi, self).__init__()
        self.kernel_size = kernel_size
        self.series_decomp = [series_decomp(kernel) for kernel in kernel_size]

    def forward(self, x):
        moving_mean = []
        res = []
        for func in self.series_decomp:
            sea, moving_avg = func(x)
            moving_mean.append(moving_avg)
            res.append(sea)

        sea = sum(res) / len(res)
        moving_mean = sum(moving_mean) / len(moving_mean)
        return sea, moving_mean


class EncoderLayer(nn.Module):
    """
    Autoformer encoder layer with the progressive decomposition architecture
    """

    def __init__(self, attention, d_model, d_ff=None, moving_avg=25, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
        self.decomp1 = series_decomp(moving_avg)
        self.decomp2 = series_decomp(moving_avg)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)
        x, _ = self.decomp1(x)
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        res, _ = self.decomp2(x + y)
        return res, attn


class Encoder(nn.Module):
    """
    Autoformer encoder
    """

    def __init__(self, attn_layers, conv_layers=None, norm_layer=None, injection_mode='last',
                 text_embedding_dim=None, cfa_reduction=8):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer
        self.injection_mode = injection_mode
        
        # Text injection for middle-layer normalization
        self.has_text = text_embedding_dim is not None
        # Determine feature size from norm
        feature_size = None
        if self.has_text and self.norm is not None:
            if hasattr(norm_layer, 'layernorm') and hasattr(norm_layer.layernorm, 'normalized_shape'):
                feature_size = norm_layer.layernorm.normalized_shape[0]
            elif hasattr(norm_layer, 'normalized_shape'):
                feature_size = norm_layer.normalized_shape[0]

        self.text_proj_alpha = None
        self.text_proj_beta = None

        # Add middle-layer fusion options (gated, adapter, FiLM, orthogonal, middle-additive, middle-concat)
        self.text_proj_gated = None
        self.gate_network = None
        self.adapter_down = None
        self.adapter_norm = None
        self.adapter_activation = None
        self.adapter_up = None
        self.text_proj_gamma_film = None
        self.text_proj_beta_film = None
        self.text_proj_ortho = None
        self.text_proj_middle = None
        self.middle_concat_proj = None

        if self.has_text and feature_size is not None and injection_mode != 'unimodal':
            if injection_mode in ('middle-additive', 'middle-concat'):
                self.text_proj_middle = nn.Linear(text_embedding_dim, feature_size, bias=False)
            if injection_mode == 'middle-concat':
                self.middle_concat_proj = nn.Linear(2 * feature_size, feature_size, bias=True)
        if self.has_text and feature_size is not None and (injection_mode == 'gating' or
                                                           injection_mode == 'cfa' or
                                                           injection_mode == 'film' or
                                                           injection_mode == 'orthogonal'):
            if injection_mode == 'gating':
                self.text_proj_gated = nn.Linear(text_embedding_dim, feature_size, bias=False)
                self.gate_network = nn.Sequential(
                    nn.Linear(feature_size, feature_size, bias=True),
                    nn.ReLU(),
                    nn.Linear(feature_size, feature_size, bias=True),
                    nn.Sigmoid()
                )
            if injection_mode == 'cfa':
                adapter_bottleneck = max(1, feature_size // cfa_reduction)
                self.adapter_down = nn.Linear(text_embedding_dim, adapter_bottleneck, bias=False)
                self.adapter_norm = nn.LayerNorm(adapter_bottleneck)
                self.adapter_activation = nn.ReLU()
                self.adapter_up = nn.Linear(adapter_bottleneck, feature_size, bias=False)
            if injection_mode == 'film':
                self.text_proj_gamma_film = nn.Linear(text_embedding_dim, feature_size, bias=True)
                self.text_proj_beta_film = nn.Linear(text_embedding_dim, feature_size, bias=True)
                nn.init.ones_(self.text_proj_gamma_film.bias)
                nn.init.zeros_(self.text_proj_beta_film.bias)
            if injection_mode == 'orthogonal':
                self.text_proj_ortho = nn.Linear(text_embedding_dim, feature_size, bias=False)

    def forward(self, x, attn_mask=None, text_context=None):
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

            # Middle fusion: skip when unimodal or no text_context
            if text_context is not None and (self.text_proj_middle is not None or self.text_proj_gated is not None or
                                              self.adapter_down is not None or self.text_proj_gamma_film is not None or
                                              self.text_proj_ortho is not None):
                def _project_text(projector):
                    if projector is None:
                        return None
                    if len(text_context.shape) == 2:
                        projected = projector(text_context)
                        return projected.unsqueeze(1).expand(-1, x.shape[1], -1)
                    elif len(text_context.shape) == 3:
                        projected_seq = projector(text_context)
                        return projected_seq.mean(dim=1, keepdim=True).expand(-1, x.shape[1], -1)
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

        return x, attns


class DecoderLayer(nn.Module):
    """
    Autoformer decoder layer with the progressive decomposition architecture
    """

    def __init__(self, self_attention, cross_attention, d_model, c_out, d_ff=None,
                 moving_avg=25, dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
        self.decomp1 = series_decomp(moving_avg)
        self.decomp2 = series_decomp(moving_avg)
        self.decomp3 = series_decomp(moving_avg)
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=3, stride=1, padding=1,
                                    padding_mode='circular', bias=False)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask
        )[0])
        x, trend1 = self.decomp1(x)
        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask
        )[0])
        x, trend2 = self.decomp2(x)
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        x, trend3 = self.decomp3(x + y)

        residual_trend = trend1 + trend2 + trend3
        residual_trend = self.projection(residual_trend.permute(0, 2, 1)).transpose(1, 2)
        return x, residual_trend


class Decoder(nn.Module):
    """
    Autoformer encoder
    """

    def __init__(self, layers, norm_layer=None, projection=None, injection_mode='last',
                 text_embedding_dim=None, cfa_reduction=8):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection
        self.injection_mode = injection_mode
        
        # Text injection for middle-layer normalization
        self.has_text = text_embedding_dim is not None
        feature_size = None
        if self.has_text and self.norm is not None:
            if hasattr(norm_layer, 'layernorm') and hasattr(norm_layer.layernorm, 'normalized_shape'):
                feature_size = norm_layer.layernorm.normalized_shape[0]
            elif hasattr(norm_layer, 'normalized_shape'):
                feature_size = norm_layer.normalized_shape[0]

        self.text_proj_alpha = None
        self.text_proj_beta = None

        # Add middle-layer fusion options (gated, adapter, FiLM, orthogonal, middle-additive, middle-concat)
        self.text_proj_gated = None
        self.gate_network = None
        self.adapter_down = None
        self.adapter_norm = None
        self.adapter_activation = None
        self.adapter_up = None
        self.text_proj_gamma_film = None
        self.text_proj_beta_film = None
        self.text_proj_ortho = None
        self.text_proj_middle = None
        self.middle_concat_proj = None

        if self.has_text and feature_size is not None and injection_mode != 'unimodal':
            if injection_mode in ('middle-additive', 'middle-concat'):
                self.text_proj_middle = nn.Linear(text_embedding_dim, feature_size, bias=False)
            if injection_mode == 'middle-concat':
                self.middle_concat_proj = nn.Linear(2 * feature_size, feature_size, bias=True)
        if self.has_text and feature_size is not None and (injection_mode == 'gating' or
                                                           injection_mode == 'cfa' or
                                                           injection_mode == 'film' or
                                                           injection_mode == 'orthogonal'):
            if injection_mode == 'gating':
                self.text_proj_gated = nn.Linear(text_embedding_dim, feature_size, bias=False)
                self.gate_network = nn.Sequential(
                    nn.Linear(feature_size, feature_size, bias=True),
                    nn.ReLU(),
                    nn.Linear(feature_size, feature_size, bias=True),
                    nn.Sigmoid()
                )
            if injection_mode == 'cfa':
                adapter_bottleneck = max(1, feature_size // cfa_reduction)
                self.adapter_down = nn.Linear(text_embedding_dim, adapter_bottleneck, bias=False)
                self.adapter_norm = nn.LayerNorm(adapter_bottleneck)
                self.adapter_activation = nn.ReLU()
                self.adapter_up = nn.Linear(adapter_bottleneck, feature_size, bias=False)
            if injection_mode == 'film':
                self.text_proj_gamma_film = nn.Linear(text_embedding_dim, feature_size, bias=True)
                self.text_proj_beta_film = nn.Linear(text_embedding_dim, feature_size, bias=True)
                nn.init.ones_(self.text_proj_gamma_film.bias)
                nn.init.zeros_(self.text_proj_beta_film.bias)
            if injection_mode == 'orthogonal':
                self.text_proj_ortho = nn.Linear(text_embedding_dim, feature_size, bias=False)

    def forward(self, x, cross, x_mask=None, cross_mask=None, trend=None, text_context=None):
        for layer in self.layers:
            x, residual_trend = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)
            trend = trend + residual_trend

        if self.norm is not None:
            x = self.norm(x)

            # Middle fusion: skip when unimodal or no text_context
            if text_context is not None and (self.text_proj_middle is not None or self.text_proj_gated is not None or
                                              self.adapter_down is not None or self.text_proj_gamma_film is not None or
                                              self.text_proj_ortho is not None):
                def _project_text(projector):
                    if projector is None:
                        return None
                    if len(text_context.shape) == 2:
                        projected = projector(text_context)
                        return projected.unsqueeze(1).expand(-1, x.shape[1], -1)
                    elif len(text_context.shape) == 3:
                        projected_seq = projector(text_context)
                        return projected_seq.mean(dim=1, keepdim=True).expand(-1, x.shape[1], -1)
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

        if self.projection is not None:
            x = self.projection(x)
        return x, trend
