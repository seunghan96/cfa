import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from layers.Embed import DataEmbedding
from layers.Conv_Blocks import Inception_Block_V1


def FFT_for_Period(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class TimesBlock(nn.Module):
    def __init__(self, configs):
        super(TimesBlock, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k
        # parameter-efficient design
        self.conv = nn.Sequential(
            Inception_Block_V1(configs.d_model, configs.d_ff,
                               num_kernels=configs.num_kernels),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, configs.d_model,
                               num_kernels=configs.num_kernels)
        )

    def forward(self, x):
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]
            # padding
            if (self.seq_len + self.pred_len) % period != 0:
                length = (
                                 ((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.seq_len + self.pred_len)
                out = x
            # reshape
            out = out.reshape(B, length // period, period,
                              N).permute(0, 3, 1, 2).contiguous()
            # 2D conv: from 1d Variation to 2d Variation
            out = self.conv(out)
            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len + self.pred_len), :])
        res = torch.stack(res, dim=-1)
        # adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(
            1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        # residual connection
        res = res + x
        return res


class Model(nn.Module):
    """
    Paper link: https://openreview.net/pdf?id=ju_Uqw384Oq
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        
        # Text injection configuration
        text_embedding_dim = getattr(configs, 'text_emb', None)
        injection_mode = getattr(configs, 'text_injection_mode')


        # Middle fusion modules (gated/adapter/FiLM/orthogonal) for encoder output
        feature_size = configs.d_model
        self.middle_text_proj = nn.Linear(text_embedding_dim, feature_size, bias=False)


        if injection_mode == 'gating':
            self.middle_gate_network = nn.Sequential(
                nn.Linear(feature_size, feature_size, bias=True),
                nn.ReLU(),
                nn.Dropout(configs.dropout),
                nn.Linear(feature_size, feature_size, bias=True),
                nn.Sigmoid()
            )
        else:
            self.middle_gate_network = None

        if injection_mode == 'cfa':
            adapter_bottleneck = max(1, feature_size // configs.cfa_reduction)
            self.middle_adapter_down = nn.Linear(text_embedding_dim, adapter_bottleneck, bias=False)
            self.middle_adapter_norm = nn.LayerNorm(adapter_bottleneck)
            self.middle_adapter_activation = nn.ReLU()
            self.middle_adapter_up = nn.Linear(adapter_bottleneck, feature_size, bias=False)
        else:
            self.middle_adapter_down = None
            self.middle_adapter_norm = None
            self.middle_adapter_activation = None
            self.middle_adapter_up = None

        if injection_mode == 'film':
            self.middle_text_proj_gamma_film = nn.Linear(text_embedding_dim, feature_size, bias=True)
            self.middle_text_proj_beta_film = nn.Linear(text_embedding_dim, feature_size, bias=True)
            nn.init.ones_(self.middle_text_proj_gamma_film.bias)
            nn.init.zeros_(self.middle_text_proj_beta_film.bias)
        else:
            self.middle_text_proj_gamma_film = None
            self.middle_text_proj_beta_film = None

        if injection_mode == 'orthogonal':
            self.middle_text_proj_ortho = nn.Linear(text_embedding_dim, feature_size, bias=False)
        else:
            self.middle_text_proj_ortho = None

        
        self.model = nn.ModuleList([TimesBlock(configs)
                                    for _ in range(configs.e_layers)])
        self.enc_embedding = DataEmbedding(
            configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout,
            text_embedding_dim=text_embedding_dim, injection_mode=injection_mode,
            
        )
        self.layer = configs.e_layers
        self.layer_norm = nn.LayerNorm(configs.d_model)
        self.injection_mode = injection_mode
        
        self.has_text = text_embedding_dim 
        
       
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.predict_linear = nn.Linear(
                self.seq_len, self.pred_len + self.seq_len)
            self.projection = nn.Linear(
                configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(
                configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                configs.d_model * configs.seq_len, configs.num_class)


    def _apply_middle_injection(self, x, text_context):
        """Apply text injection at middle stage (after TimesNet blocks)"""

        if len(text_context.shape) == 2:
            text_emb = self.middle_text_proj(text_context).unsqueeze(1).expand(-1, x.shape[1], -1)
        elif len(text_context.shape) == 3:
            text_emb_seq = self.middle_text_proj(text_context)
            text_emb = text_emb_seq.mean(dim=1, keepdim=True).expand(-1, x.shape[1], -1)

        if self.middle_gate_network is not None and self.injection_mode.startswith('gating'):
            gate = self.middle_gate_network(x)
            return x + gate * text_emb

        if self.middle_adapter_down is not None and self.middle_adapter_up is not None and self.injection_mode.startswith('cfa'):
            pooled_text = text_context if len(text_context.shape) == 2 else text_context.mean(dim=1)
            adapter_hidden = self.middle_adapter_down(pooled_text)
            adapter_hidden = self.middle_adapter_norm(adapter_hidden)
            adapter_hidden = self.middle_adapter_activation(adapter_hidden)
            adapter_out = self.middle_adapter_up(adapter_hidden).unsqueeze(1).expand(-1, x.shape[1], -1)
            return x + adapter_out

        if self.middle_text_proj_gamma_film is not None and self.injection_mode.startswith('film'):
            pooled_text = text_context if len(text_context.shape) == 2 else text_context.mean(dim=1)
            gamma = self.middle_text_proj_gamma_film(pooled_text).unsqueeze(1).expand(-1, x.shape[1], -1)
            beta = self.middle_text_proj_beta_film(pooled_text).unsqueeze(1).expand(-1, x.shape[1], -1)
            return gamma * x + beta

        if self.middle_text_proj_ortho is not None and self.injection_mode.startswith('orthogonal'):
            pooled_text = text_context if len(text_context.shape) == 2 else text_context.mean(dim=1)
            ortho_text = self.middle_text_proj_ortho(pooled_text).unsqueeze(1).expand(-1, x.shape[1], -1)
            proj_coeff = (ortho_text * x).sum(dim=-1, keepdim=True) / (x.pow(2).sum(dim=-1, keepdim=True) + 1e-6)
            ortho_component = ortho_text - proj_coeff * x
            return x + ortho_component

        return x

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, text_context=None):

        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc, text_context=None)  # [B,T,C]
        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(
            0, 2, 1)  # align temporal dimension
        # TimesNet
        for i in range(self.layer):
            enc_out = self.model[i](enc_out)
            enc_out = self.layer_norm(enc_out)


        enc_out = self._apply_middle_injection(enc_out, text_context)
        
        # porject back
        dec_out = self.projection(enc_out)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        return dec_out


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None, text_context=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, text_context=text_context)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(
                x_enc, x_mark_enc, x_dec, x_mark_dec, mask, text_context=text_context)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc, text_context=text_context)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc, text_context=text_context)
            return dec_out  # [B, N]
        return None
