import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Autoformer_EncDec import series_decomp


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2205.13504.pdf
    """

    def __init__(self, configs, individual=False):
        """
        individual: Bool, whether shared model among different variates.
        """
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        if self.task_name == 'classification' or self.task_name == 'anomaly_detection' or self.task_name == 'imputation':
            self.pred_len = configs.seq_len
        else:
            self.pred_len = configs.pred_len
        
        # Text injection configuration
        text_embedding_dim = getattr(configs, 'text_emb', None)
        injection_mode = getattr(configs, 'text_injection_mode')
        
        # Series decomposition block from Autoformer
        self.decompsition = series_decomp(configs.moving_avg)
        self.individual = individual
        self.channels = configs.enc_in
        self.injection_mode = injection_mode
        
        self.has_text = True

        # Shared text -> channels projection (middle injection and first fusion)
        if text_embedding_dim is not None and injection_mode != 'unimodal':
            self.text_projection = nn.Linear(text_embedding_dim, self.channels, bias=False)
        else:
            self.text_projection = None

        if injection_mode == 'first-concat':
            self.first_concat_proj = nn.Linear(2 * self.channels, self.channels, bias=True)

        if injection_mode == 'middle-concat':
            self.middle_concat_proj = nn.Linear(2 * self.channels, self.channels, bias=True)
            
        if injection_mode == 'last-concat':
            self.last_fusion_concat_proj = nn.Linear(2 * self.channels, self.channels, bias=True)

        if injection_mode == 'gating':
            self.last_gate_network = nn.Sequential(
                nn.Linear(self.channels, self.channels, bias=True),
                nn.ReLU(),
                nn.Dropout(configs.dropout),
                nn.Linear(self.channels, self.channels, bias=True),
                nn.Sigmoid()
            )
        else:
            self.last_gate_network = None

        if injection_mode == 'cfa':
            adapter_bottleneck = max(1, self.channels // configs.cfa_reduction)
            self.last_adapter_down = nn.Linear(text_embedding_dim, adapter_bottleneck, bias=False)
            self.last_adapter_norm = nn.LayerNorm(adapter_bottleneck)
            self.last_adapter_activation = nn.ReLU()
            self.last_adapter_up = nn.Linear(adapter_bottleneck, self.channels, bias=False)
        else:
            self.last_adapter_down = None
            self.last_adapter_norm = None
            self.last_adapter_activation = None
            self.last_adapter_up = None

        if injection_mode == 'film':
            self.last_text_proj_gamma_film = nn.Linear(text_embedding_dim, self.channels, bias=True)
            self.last_text_proj_beta_film = nn.Linear(text_embedding_dim, self.channels, bias=True)
            nn.init.ones_(self.last_text_proj_gamma_film.bias)
            nn.init.zeros_(self.last_text_proj_beta_film.bias)
        else:
            self.last_text_proj_gamma_film = None
            self.last_text_proj_beta_film = None

        if injection_mode == 'orthogonal':
            self.last_text_proj_ortho = nn.Linear(text_embedding_dim, self.channels, bias=False)
        else:
            self.last_text_proj_ortho = None

        if injection_mode == 'concat':
            self.last_concat_proj = nn.Conv1d(2 * self.channels, self.channels, kernel_size=1, bias=True)
        else:
            self.last_concat_proj = None

        
        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()

            for i in range(self.channels):
                self.Linear_Seasonal.append(
                    nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Trend.append(
                    nn.Linear(self.seq_len, self.pred_len))

                self.Linear_Seasonal[i].weight = nn.Parameter(
                    (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
                self.Linear_Trend[i].weight = nn.Parameter(
                    (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

            self.Linear_Seasonal.weight = nn.Parameter(
                (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
            self.Linear_Trend.weight = nn.Parameter(
                (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))

        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                configs.enc_in * configs.seq_len, configs.num_class)


    def _apply_middle_injection(self, x, text_context):
        """Apply middle fusion on [B, C, L]. unimodal: text_context is None, return x unchanged."""
        if text_context is None:
            return x
        if len(text_context.shape) == 2:  # [B, text_embedding_dim]
            text_emb = self.text_projection(text_context)  # [B, channels]
            text_emb = text_emb.unsqueeze(-1).expand(-1, -1, x.shape[-1])
        elif len(text_context.shape) == 3:  # [B, text_seq_len, text_embedding_dim]
            text_emb_seq = self.text_projection(text_context)  # [B, text_seq_len, channels]
            text_emb = text_emb_seq.mean(dim=1, keepdim=True)  # [B, 1, channels]
            text_emb = text_emb.permute(0, 2, 1).expand(-1, -1, x.shape[-1])  # [B, channels, L]

        if self.injection_mode == 'gating':
            gate = self.last_gate_network(x.transpose(1, 2)).transpose(1, 2) if x.dim() == 3 else self.last_gate_network(x)
            return x + gate * text_emb

        if self.injection_mode == 'cfa':        
            pooled_text = text_context if len(text_context.shape) == 2 else text_context.mean(dim=1)
            adapter_hidden = self.last_adapter_down(pooled_text)
            adapter_hidden = self.last_adapter_norm(adapter_hidden)
            adapter_hidden = self.last_adapter_activation(adapter_hidden)
            adapter_out = self.last_adapter_up(adapter_hidden).unsqueeze(-1).expand(-1, x.shape[1], x.shape[2])
            return x + adapter_out

        if self.injection_mode == 'film':                
            pooled_text = text_context if len(text_context.shape) == 2 else text_context.mean(dim=1)
            gamma = self.last_text_proj_gamma_film(pooled_text).unsqueeze(-1).expand(-1, x.shape[1], x.shape[2])
            beta = self.last_text_proj_beta_film(pooled_text).unsqueeze(-1).expand(-1, x.shape[1], x.shape[2])
            return gamma * x + beta

        if self.injection_mode == 'orthogonal':                
            pooled_text = text_context if len(text_context.shape) == 2 else text_context.mean(dim=1)
            ortho_text = self.last_text_proj_ortho(pooled_text).unsqueeze(-1).expand(-1, x.shape[1], x.shape[2])
            proj_coeff = (ortho_text * x).sum(dim=1, keepdim=True) / (x.pow(2).sum(dim=1, keepdim=True) + 1e-6)
            ortho_component = ortho_text - proj_coeff * x
            return x + ortho_component

        if self.injection_mode == 'middle-concat':
            # x, text_emb: [B, C, L] -> concat on channel -> [B, 2*C, L] -> project to [B, C, L]
            concat = torch.cat([x, text_emb], dim=1)
            return self.middle_concat_proj(concat.permute(0,2,1)).permute(0,2,1)

        if self.injection_mode == 'middle-additive':                
            return x + text_emb       
        
        else:
            return x


    def encoder(self, x, text_context=None):
        
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(
            0, 2, 1), trend_init.permute(0, 2, 1)
        
        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0), seasonal_init.size(1), self.pred_len],
                                          dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0), trend_init.size(1), self.pred_len],
                                       dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](
                    seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](
                    trend_init[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)
        
        
        # Middle fusion: apply to seasonal/trend components after linear projection
        seasonal_output = self._apply_middle_injection(seasonal_output, text_context)
        trend_output = self._apply_middle_injection(trend_output, text_context)
        
        x = seasonal_output + trend_output
        
        return x.permute(0, 2, 1)  # [B, pred_len, channels]

    def forecast(self, x_enc, text_context=None):
        if self.injection_mode == 'unimodal':
            text_context = None
        #############################################################
        # First fusion: inject text into x_enc [B, L, C] before encoder (shape preserved)
        if self.injection_mode in ('first-additive', 'first-concat'):
            if len(text_context.shape) == 2:
                text_emb = self.text_projection(text_context)  # [B, C]
                text_emb = text_emb.unsqueeze(1).expand(-1, x_enc.shape[1], -1)  # [B, L, C]
            else:
                text_emb = self.text_projection(text_context).mean(dim=1)  # [B, C]
                text_emb = text_emb.unsqueeze(1).expand(-1, x_enc.shape[1], -1)  # [B, L, C]

            if self.injection_mode == 'first-additive':
                x_enc = x_enc + text_emb
            else:
                concat = torch.cat([x_enc, text_emb], dim=-1)  # [B, L, 2*C]
                x_enc = self.first_concat_proj(concat)  # [B, L, C]
    
        x = self.encoder(x_enc, text_context=text_context)

        # Last fusion: inject text into encoder output [B, pred_len, C] (shape preserved)
        if self.injection_mode in ('last-additive', 'last-concat'):
            if len(text_context.shape) == 2:
                text_emb = self.text_projection(text_context)  # [B, C]
                text_emb = text_emb.unsqueeze(1).expand(-1, x.shape[1], -1)  # [B, pred_len, C]
            else:
                text_emb = self.text_projection(text_context).mean(dim=1)  # [B, C]
                text_emb = text_emb.unsqueeze(1).expand(-1, x.shape[1], -1)  # [B, pred_len, C]

            if self.injection_mode == 'last-additive':
                x = x + text_emb
            else:
                concat = torch.cat([x, text_emb], dim=-1)  # [B, pred_len, 2*C]
                x = self.last_fusion_concat_proj(concat)  # [B, pred_len, C]

        return x

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None, text_context=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, text_context=text_context)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, text_context=text_context)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc, text_context=text_context)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, text_context=text_context)
            return dec_out  # [B, N]
        return None
