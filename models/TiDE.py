import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)





class ResBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1, bias=True,
                 text_embedding_dim=None, injection_mode='last'):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=bias)
        self.fc2 = nn.Linear(hidden_dim, output_dim, bias=bias)
        self.fc3 = nn.Linear(input_dim, output_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.ln = LayerNorm(output_dim, bias=bias)
        self.injection_mode = injection_mode
        feature_size = output_dim

        if injection_mode != 'unimodal' and text_embedding_dim is not None:
            self.text_proj = nn.Linear(text_embedding_dim, feature_size, bias=False)
        else:
            self.text_proj = None

        if injection_mode == 'middle-concat':
            self.middle_concat_proj = nn.Linear(2 * feature_size, feature_size, bias=True)
        else:
            self.middle_concat_proj = None

        if injection_mode == 'gating':
            self.gate_network = nn.Sequential(
                nn.Linear(feature_size, feature_size, bias=True),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(feature_size, feature_size, bias=True),
                nn.Sigmoid()
            )
        else:
            self.gate_network = None

        if injection_mode == 'cfa':
            adapter_bottleneck = max(1, feature_size // configs.cfa_reduction)
            self.adapter_down = nn.Linear(text_embedding_dim, adapter_bottleneck, bias=False)
            self.adapter_norm = nn.LayerNorm(adapter_bottleneck)
            self.adapter_activation = nn.ReLU()
            self.adapter_up = nn.Linear(adapter_bottleneck, feature_size, bias=False)
        else:
            self.adapter_down = None
            self.adapter_norm = None
            self.adapter_activation = None
            self.adapter_up = None

        if injection_mode == 'film':
            self.text_proj_gamma_film = nn.Linear(text_embedding_dim, feature_size, bias=True)
            self.text_proj_beta_film = nn.Linear(text_embedding_dim, feature_size, bias=True)
            nn.init.ones_(self.text_proj_gamma_film.bias)
            nn.init.zeros_(self.text_proj_beta_film.bias)
        else:
            self.text_proj_gamma_film = None
            self.text_proj_beta_film = None

        if injection_mode == 'orthogonal':
            self.text_proj_ortho = nn.Linear(text_embedding_dim, feature_size, bias=False)
        else:
            self.text_proj_ortho = None

    def _apply_injection(self, out, text_context):
        """Apply text injection to out [B, D] or [B, L, D]. unimodal: text_context is None or text_proj is None, return out unchanged."""
        if text_context is None or self.text_proj is None:
            return out
        if len(out.shape) == 2:
            B, D = out.shape
            if len(text_context.shape) == 2:
                pooled_text = text_context
            else:
                pooled_text = text_context.mean(dim=1)
            need_expand = False
        else:
            B, L, D = out.shape
            if len(text_context.shape) == 2:
                pooled_text = text_context
                need_expand = True
            else:
                pooled_text = text_context.mean(dim=1)
                need_expand = True

        def expand_if(t):
            return t.unsqueeze(1).expand(-1, L, -1) if need_expand and len(out.shape) == 3 else t

        text_emb = self.text_proj(pooled_text)
        text_emb = expand_if(text_emb)

        if self.injection_mode == 'middle-additive':
            return out + text_emb
        if self.middle_concat_proj is not None:
            concat = torch.cat([out, text_emb], dim=-1)
            return self.middle_concat_proj(concat)

        if self.gate_network is not None:
            gate = self.gate_network(out)
            return out + gate * text_emb

        if self.adapter_down is not None:
            adapter_hidden = self.adapter_down(pooled_text)
            adapter_hidden = self.adapter_norm(adapter_hidden)
            adapter_hidden = self.adapter_activation(adapter_hidden)
            adapter_out = self.adapter_up(adapter_hidden)
            adapter_out = expand_if(adapter_out)
            return out + adapter_out

        if self.text_proj_gamma_film is not None:
            gamma = self.text_proj_gamma_film(pooled_text)
            beta = self.text_proj_beta_film(pooled_text)
            gamma = expand_if(gamma)
            beta = expand_if(beta)
            return gamma * out + beta

        if self.text_proj_ortho is not None:
            ortho_text = self.text_proj_ortho(pooled_text)
            ortho_text = expand_if(ortho_text)
            proj_coeff = (ortho_text * out).sum(dim=-1, keepdim=True) / (out.pow(2).sum(dim=-1, keepdim=True) + 1e-6)
            ortho_component = ortho_text - proj_coeff * out
            return out + ortho_component

        return out

    def forward(self, x, text_context=None):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = out + self.fc3(x)
        out = self._apply_injection(out, text_context)
        out = self.ln(out)
        return out


#TiDE
class Model(nn.Module):  
    """
    paper: https://arxiv.org/pdf/2304.08424.pdf 
    """
    def __init__(self, configs, bias=True, feature_encode_dim=2): 
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len  #L 
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len  #H 
        self.hidden_dim=configs.d_model
        self.res_hidden=configs.d_model 
        self.encoder_num=configs.e_layers
        self.decoder_num=configs.d_layers
        self.freq=configs.freq
        self.feature_encode_dim=feature_encode_dim
        self.decode_dim = configs.c_out
        self.temporalDecoderHidden=configs.d_ff
        dropout=configs.dropout

        
        freq_map = {'h': 4, 't': 5, 's': 6,
                    'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        
        self.feature_dim=freq_map[self.freq]

        # Text injection configuration (need to define early for ResBlock initialization)
        text_embedding_dim = getattr(configs, 'text_emb', None)
        injection_mode = getattr(configs, 'text_injection_mode')
        self.injection_mode = injection_mode
        # Per-feature forecast uses x_enc [B, L, 1] and dec_out [B, pred_len, 1]
        self.channels = 1
        if injection_mode in ('first-additive', 'first-concat', 'last-additive', 'last-concat') and text_embedding_dim is not None:
            self.text_projection = nn.Linear(text_embedding_dim, self.channels, bias=False)
        else:
            self.text_projection = None
        if injection_mode == 'first-concat':
            self.first_concat_proj = nn.Linear(2 * self.channels, self.channels, bias=True)
        else:
            self.first_concat_proj = None
        if injection_mode == 'last-concat':
            self.last_fusion_concat_proj = nn.Linear(2 * self.channels, self.channels, bias=True)
        else:
            self.last_fusion_concat_proj = None

        flatten_dim = self.seq_len + (self.seq_len + self.pred_len) * self.feature_encode_dim

        self.feature_encoder = ResBlock(self.feature_dim, self.res_hidden, self.feature_encode_dim, dropout, bias,
                                       text_embedding_dim=text_embedding_dim, injection_mode=injection_mode)
        # Use ModuleList instead of Sequential to pass text_context
        encoder_blocks = [ResBlock(flatten_dim, self.res_hidden, self.hidden_dim, dropout, bias,
                                  text_embedding_dim=text_embedding_dim, injection_mode=injection_mode)]
        encoder_blocks.extend([ResBlock(self.hidden_dim, self.res_hidden, self.hidden_dim, dropout, bias,
                                       text_embedding_dim=text_embedding_dim, injection_mode=injection_mode)
                              for _ in range(self.encoder_num-1)])
        self.encoders = nn.ModuleList(encoder_blocks)
        
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            decoder_blocks = [ResBlock(self.hidden_dim, self.res_hidden, self.hidden_dim, dropout, bias,
                                      text_embedding_dim=text_embedding_dim, injection_mode=injection_mode)
                             for _ in range(self.decoder_num-1)]
            decoder_blocks.append(ResBlock(self.hidden_dim, self.res_hidden, self.decode_dim * self.pred_len, dropout, bias,
                                          text_embedding_dim=text_embedding_dim, injection_mode=injection_mode))
            self.decoders = nn.ModuleList(decoder_blocks)
            self.temporalDecoder = ResBlock(self.decode_dim + self.feature_encode_dim, self.temporalDecoderHidden, 1, dropout, bias,
                                           text_embedding_dim=text_embedding_dim, injection_mode=injection_mode)
            self.residual_proj = nn.Linear(self.seq_len, self.pred_len, bias=bias)
        if self.task_name == 'imputation':
            decoder_blocks = [ResBlock(self.hidden_dim, self.res_hidden, self.hidden_dim, dropout, bias,
                                      text_embedding_dim=text_embedding_dim, injection_mode=injection_mode)
                             for _ in range(self.decoder_num-1)]
            decoder_blocks.append(ResBlock(self.hidden_dim, self.res_hidden, self.decode_dim * self.seq_len, dropout, bias,
                                          text_embedding_dim=text_embedding_dim, injection_mode=injection_mode))
            self.decoders = nn.ModuleList(decoder_blocks)
            self.temporalDecoder = ResBlock(self.decode_dim + self.feature_encode_dim, self.temporalDecoderHidden, 1, dropout, bias,
                                           text_embedding_dim=text_embedding_dim, injection_mode=injection_mode)
            self.residual_proj = nn.Linear(self.seq_len, self.seq_len, bias=bias)
        if self.task_name == 'anomaly_detection':
            decoder_blocks = [ResBlock(self.hidden_dim, self.res_hidden, self.hidden_dim, dropout, bias,
                                      text_embedding_dim=text_embedding_dim, injection_mode=injection_mode)
                             for _ in range(self.decoder_num-1)]
            decoder_blocks.append(ResBlock(self.hidden_dim, self.res_hidden, self.decode_dim * self.seq_len, dropout, bias,
                                          text_embedding_dim=text_embedding_dim, injection_mode=injection_mode))
            self.decoders = nn.ModuleList(decoder_blocks)
            self.temporalDecoder = ResBlock(self.decode_dim + self.feature_encode_dim, self.temporalDecoderHidden, 1, dropout, bias,
                                           text_embedding_dim=text_embedding_dim, injection_mode=injection_mode)
            self.residual_proj = nn.Linear(self.seq_len, self.seq_len, bias=bias)
        self.injection_mode = injection_mode
        
    def forecast(self, x_enc, x_mark_enc, x_dec, batch_y_mark, text_context=None):
        if self.injection_mode == 'unimodal':
            text_context = None

        # Same batch_y_mark preprocessing as forward() so feature_encoder gets consistent input
        # (required when forecast() is called directly, e.g. from experiment / layer-similarity)
        batch_y_mark = torch.cat([x_mark_enc, batch_y_mark[:, -self.pred_len:, :]], dim=1)

        # When called with full 3D input [B, L, D] (e.g. from layer-similarity / analysis), run per-feature and stack
        if x_enc.dim() == 3:
            dec_out = torch.stack([
                self.forecast(x_enc[:, :, f], x_mark_enc, x_dec, batch_y_mark, text_context=text_context)
                for f in range(x_enc.shape[-1])
            ], dim=-1)
            return dec_out

        # First fusion: x_enc is [B, L] (per feature); use 3D for fusion then squeeze back (no inplace)
        if self.injection_mode in ('first-additive', 'first-concat') and self.text_projection is not None:
            x_enc_3d = x_enc.unsqueeze(-1)  # [B, L, 1]
            if len(text_context.shape) == 2:
                text_emb = self.text_projection(text_context).unsqueeze(1).expand(-1, x_enc_3d.shape[1], -1)
            else:
                text_emb = self.text_projection(text_context.mean(dim=1)).unsqueeze(1).expand(-1, x_enc_3d.shape[1], -1)
            if self.injection_mode == 'first-additive':
                x_enc_3d = x_enc_3d + text_emb
            else:
                x_enc_3d = self.first_concat_proj(torch.cat([x_enc_3d, text_emb], dim=-1))
            x_enc = x_enc_3d.squeeze(-1)  # [B, L]

        # Normalization (non-inplace to avoid autograd error)
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev

        enc_text = None if self.injection_mode in ('first-additive', 'first-concat', 'last-additive', 'last-concat') else text_context
        feature = self.feature_encoder(batch_y_mark, text_context=enc_text)
        hidden_input = torch.cat([x_enc, feature.reshape(feature.shape[0], -1)], dim=-1)
        hidden = hidden_input
        for encoder_block in self.encoders:
            hidden = encoder_block(hidden, text_context=enc_text)
        decoded = hidden
        for decoder_block in self.decoders:
            decoded = decoder_block(decoded, text_context=None)
        decoded = decoded.reshape(hidden.shape[0], self.pred_len, self.decode_dim)

        dec_out = self.temporalDecoder(torch.cat([feature[:, self.seq_len:], decoded], dim=-1),
                                       text_context=None).squeeze(-1) + self.residual_proj(x_enc)

        # De-Normalization
        dec_out = dec_out * (stdev[:, 0].unsqueeze(1).repeat(1, self.pred_len))
        dec_out = dec_out + (means[:, 0].unsqueeze(1).repeat(1, self.pred_len))

        # Last fusion: dec_out is [B, pred_len]; use 3D for fusion then squeeze back (no inplace)
        if self.injection_mode in ('last-additive', 'last-concat') and self.text_projection is not None:
            dec_out_3d = dec_out.unsqueeze(-1)  # [B, pred_len, 1]
            if len(text_context.shape) == 2:
                text_emb = self.text_projection(text_context).unsqueeze(1).expand(-1, dec_out_3d.shape[1], -1)
            else:
                text_emb = self.text_projection(text_context.mean(dim=1)).unsqueeze(1).expand(-1, dec_out_3d.shape[1], -1)
            if self.injection_mode == 'last-additive':
                dec_out_3d = dec_out_3d + text_emb
            else:
                dec_out_3d = self.last_fusion_concat_proj(torch.cat([dec_out_3d, text_emb], dim=-1))
            dec_out = dec_out_3d.squeeze(-1)  # [B, pred_len]

        return dec_out
    
    
    def forward(self, x_enc, x_mark_enc, x_dec, batch_y_mark, mask=None, text_context=None):
        '''x_mark_enc is the exogenous dynamic feature described in the original paper'''
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            batch_y_mark=torch.concat([x_mark_enc, batch_y_mark[:, -self.pred_len:, :]],dim=1)
            dec_out = torch.stack([self.forecast(x_enc[:, :, feature], x_mark_enc, x_dec, batch_y_mark, text_context=text_context) for feature in range(x_enc.shape[-1])],dim=-1)
            return dec_out # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = torch.stack([self.imputation(x_enc[:, :, feature], x_mark_enc, x_dec, batch_y_mark, mask, text_context=text_context) for feature in range(x_enc.shape[-1])],dim=-1)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            raise NotImplementedError("Task anomaly_detection for Tide is temporarily not supported")
        if self.task_name == 'classification':
            raise NotImplementedError("Task classification for Tide is temporarily not supported")
        return None

    



