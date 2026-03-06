import torch.nn as nn
import torch

class ResBlock(nn.Module):
    def __init__(self, configs, text_embedding_dim=None, injection_mode='last'):
        super(ResBlock, self).__init__()

        self.temporal = nn.Sequential(
            nn.Linear(configs.seq_len, configs.d_model),
            nn.ReLU(),
            nn.Linear(configs.d_model, configs.seq_len),
            nn.Dropout(configs.dropout)
        )

        self.channel = nn.Sequential(
            nn.Linear(configs.enc_in, configs.d_model),
            nn.ReLU(),
            nn.Linear(configs.d_model, configs.enc_in),
            nn.Dropout(configs.dropout)
        )

        self.injection_mode = injection_mode
        feature_size = configs.enc_in

        if injection_mode != 'unimodal' and text_embedding_dim is not None:
            self.text_proj = nn.Linear(text_embedding_dim, feature_size, bias=False)
        else:
            self.text_proj = None
        if injection_mode == 'gating':
            self.gate_network = nn.Sequential(
                nn.Linear(feature_size, feature_size, bias=True),
                nn.ReLU(),
                nn.Dropout(configs.dropout),
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

        if injection_mode == 'middle-concat':
            self.middle_concat_proj = nn.Linear(2 * feature_size, feature_size, bias=True)
        else:
            self.middle_concat_proj = None

        self.has_text = True

    def _apply_injection(self, x, text_context):
        """Apply injection on [B, L, D]. unimodal: text_context is None or text_proj is None, return x unchanged."""
        if text_context is None or self.text_proj is None:
            return x
        if len(text_context.shape) == 2:
            text_emb = self.text_proj(text_context).unsqueeze(1).expand(-1, x.shape[1], -1)
        else:
            text_emb = self.text_proj(text_context.mean(dim=1)).unsqueeze(1).expand(-1, x.shape[1], -1)

        if self.gate_network is not None:
            gate = self.gate_network(x)
            return x + gate * text_emb

        if self.adapter_down is not None:
            pooled_text = text_context if len(text_context.shape) == 2 else text_context.mean(dim=1)
            adapter_hidden = self.adapter_down(pooled_text)
            adapter_hidden = self.adapter_norm(adapter_hidden)
            adapter_hidden = self.adapter_activation(adapter_hidden)
            adapter_out = self.adapter_up(adapter_hidden).unsqueeze(1).expand(-1, x.shape[1], -1)
            return x + adapter_out

        if self.text_proj_gamma_film is not None:
            pooled_text = text_context if len(text_context.shape) == 2 else text_context.mean(dim=1)
            gamma = self.text_proj_gamma_film(pooled_text).unsqueeze(1).expand(-1, x.shape[1], -1)
            beta = self.text_proj_beta_film(pooled_text).unsqueeze(1).expand(-1, x.shape[1], -1)
            return gamma * x + beta

        if self.text_proj_ortho is not None:
            pooled_text = text_context if len(text_context.shape) == 2 else text_context.mean(dim=1)
            ortho_text = self.text_proj_ortho(pooled_text).unsqueeze(1).expand(-1, x.shape[1], -1)
            proj_coeff = (ortho_text * x).sum(dim=-1, keepdim=True) / (x.pow(2).sum(dim=-1, keepdim=True) + 1e-6)
            ortho_component = ortho_text - proj_coeff * x
            return x + ortho_component

        if self.middle_concat_proj is not None:
            concat = torch.cat([x, text_emb], dim=-1)  # [B, L, 2*D]
            return self.middle_concat_proj(concat)

        if self.injection_mode == 'middle-additive':
            return x + text_emb

        return x

    def forward(self, x, text_context=None):
        # x: [B, L, D]
        x = x + self.temporal(x.transpose(1, 2)).transpose(1, 2)
        x = x + self.channel(x)
        if self.has_text and text_context is not None:
            x = self._apply_injection(x, text_context)
        return x


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.layer = configs.e_layers
        self.pred_len = configs.pred_len
        self.projection = nn.Linear(configs.seq_len, configs.pred_len)
        
        # Text injection configuration (need to define early for ResBlock initialization)
        text_embedding_dim = getattr(configs, 'text_emb', None)
        injection_mode = getattr(configs, 'text_injection_mode')


        self.injection_mode = injection_mode
        self.channels = configs.enc_in

        if injection_mode in ('first-additive', 'first-concat', 'last-additive', 'last-concat'):
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

        self.model = nn.ModuleList([ResBlock(configs,
                                            text_embedding_dim=text_embedding_dim,
                                            injection_mode=injection_mode,
                                            )
                                   for _ in range(configs.e_layers)])
        


    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None, text_context=None):
        if self.injection_mode == 'unimodal':
            text_context = None

        # First fusion: inject into x_enc [B, L, D] before layers (shape preserved)
        if self.injection_mode in ('first-additive', 'first-concat'):
            if len(text_context.shape) == 2:
                text_emb = self.text_projection(text_context)
                text_emb = text_emb.unsqueeze(1).expand(-1, x_enc.shape[1], -1)
            else:
                text_emb = self.text_projection(text_context).mean(dim=1)
                text_emb = text_emb.unsqueeze(1).expand(-1, x_enc.shape[1], -1)

            if self.injection_mode == 'first-additive':
                x_enc = x_enc + text_emb
            else:
                concat = torch.cat([x_enc, text_emb], dim=-1)
                x_enc = self.first_concat_proj(concat)

        enc_text = None if self.injection_mode in ('first-additive', 'first-concat', 'last-additive', 'last-concat') else text_context
        for i in range(self.layer):
            x_enc = self.model[i](x_enc, text_context=enc_text)
        enc_out = self.projection(x_enc.transpose(1, 2)).transpose(1, 2)  # [B, pred_len, D]

        # Last fusion: inject into output [B, pred_len, D] (shape preserved)
        if self.injection_mode in ('last-additive', 'last-concat'):
            if len(text_context.shape) == 2:
                text_emb = self.text_projection(text_context)
                text_emb = text_emb.unsqueeze(1).expand(-1, enc_out.shape[1], -1)
            else:
                text_emb = self.text_projection(text_context).mean(dim=1)
                text_emb = text_emb.unsqueeze(1).expand(-1, enc_out.shape[1], -1)

            if self.injection_mode == 'last-additive':
                enc_out = enc_out + text_emb
            else:
                concat = torch.cat([enc_out, text_emb], dim=-1)
                enc_out = self.last_fusion_concat_proj(concat)

        return enc_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None, text_context=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, mask=mask, text_context=text_context)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        else:
            raise ValueError('Only forecast tasks implemented yet')
