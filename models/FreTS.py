import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2311.06184.pdf
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        if self.task_name == 'classification' or self.task_name == 'anomaly_detection' or self.task_name == 'imputation':
            self.pred_len = configs.seq_len
        else:
            self.pred_len = configs.pred_len
        self.embed_size = 128  # embed_size
        self.hidden_size = 256  # hidden_size
        self.pred_len = configs.pred_len
        self.feature_size = configs.enc_in  # channels
        self.seq_len = configs.seq_len
        self.channel_independence = configs.channel_independence
        self.sparsity_threshold = 0.01
        self.scale = 0.02
        self.embeddings = nn.Parameter(torch.randn(1, self.embed_size))
        self.r1 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
        self.i1 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
        self.rb1 = nn.Parameter(self.scale * torch.randn(self.embed_size))
        self.ib1 = nn.Parameter(self.scale * torch.randn(self.embed_size))
        self.r2 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
        self.i2 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
        self.rb2 = nn.Parameter(self.scale * torch.randn(self.embed_size))
        self.ib2 = nn.Parameter(self.scale * torch.randn(self.embed_size))

        self.fc = nn.Sequential(
            nn.Linear(self.seq_len * self.embed_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.pred_len)
        )
        
        # Text injection configuration (need to define early for LayerNorm initialization)
        text_embedding_dim = getattr(configs, 'text_emb', None)
        injection_mode = getattr(configs, 'text_injection_mode')
        self.injection_mode = injection_mode

        feature_size = self.embed_size  # for middle: [B, N, T, D], D=embed_size
        self.channels = self.feature_size  # enc_in for first/last: [B, T, N] or [B, pred_len, N]

        if injection_mode != 'unimodal' and text_embedding_dim is not None:
            self.middle_text_proj = nn.Linear(text_embedding_dim, feature_size, bias=False)
        else:
            self.middle_text_proj = None

        # First/Last fusion: project text to enc_in (channels)
        if injection_mode in ('first-additive', 'first-concat', 'last-additive', 'last-concat'):
            self.text_projection = nn.Linear(text_embedding_dim, self.channels, bias=False)
        else:
            self.text_projection = None

        if injection_mode == 'first-concat':
            self.first_concat_proj = nn.Linear(2 * self.channels, self.channels, bias=True)
        else:
            self.first_concat_proj = None

        if injection_mode == 'middle-concat':
            self.middle_concat_proj = nn.Linear(2 * self.embed_size, self.embed_size, bias=True)
        else:
            self.middle_concat_proj = None

        if injection_mode == 'last-concat':
            self.last_fusion_concat_proj = nn.Linear(2 * self.channels, self.channels, bias=True)
        else:
            self.last_fusion_concat_proj = None

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

    # dimension extension
    def tokenEmb(self, x):
        # x: [Batch, Input length, Channel]
        x = x.permute(0, 2, 1)
        x = x.unsqueeze(3)
        # N*T*1 x 1*D = N*T*D
        y = self.embeddings
        return x * y

    # frequency temporal learner
    def MLP_temporal(self, x, B, N, L):
        # [B, N, T, D]
        x = torch.fft.rfft(x, dim=2, norm='ortho')  # FFT on L dimension
        y = self.FreMLP(B, N, L, x, self.r2, self.i2, self.rb2, self.ib2)
        x = torch.fft.irfft(y, n=self.seq_len, dim=2, norm="ortho")
        return x

    # frequency channel learner
    def MLP_channel(self, x, B, N, L):
        # [B, N, T, D]
        x = x.permute(0, 2, 1, 3)
        # [B, T, N, D]
        x = torch.fft.rfft(x, dim=2, norm='ortho')  # FFT on N dimension
        y = self.FreMLP(B, L, N, x, self.r1, self.i1, self.rb1, self.ib1)
        x = torch.fft.irfft(y, n=self.feature_size, dim=2, norm="ortho")
        x = x.permute(0, 2, 1, 3)
        # [B, N, T, D]
        return x

    # frequency-domain MLPs
    # dimension: FFT along the dimension, r: the real part of weights, i: the imaginary part of weights
    # rb: the real part of bias, ib: the imaginary part of bias
    def FreMLP(self, B, nd, dimension, x, r, i, rb, ib):
        o1_real = torch.zeros([B, nd, dimension // 2 + 1, self.embed_size],
                              device=x.device)
        o1_imag = torch.zeros([B, nd, dimension // 2 + 1, self.embed_size],
                              device=x.device)

        o1_real = F.relu(
            torch.einsum('bijd,dd->bijd', x.real, r) - \
            torch.einsum('bijd,dd->bijd', x.imag, i) + \
            rb
        )

        o1_imag = F.relu(
            torch.einsum('bijd,dd->bijd', x.imag, r) + \
            torch.einsum('bijd,dd->bijd', x.real, i) + \
            ib
        )

        y = torch.stack([o1_real, o1_imag], dim=-1)
        y = F.softshrink(y, lambd=self.sparsity_threshold)
        y = torch.view_as_complex(y)
        return y


    def _apply_middle_injection(self, x, text_context):
        """Apply middle fusion on freq features x [B, N, T, D]. unimodal: text_context is None or middle_text_proj is None, return x unchanged."""
        if text_context is None or self.middle_text_proj is None:
            return x
        if len(text_context.shape) == 2:
            text_emb = self.middle_text_proj(text_context)  # [B, D]
            text_emb = text_emb.unsqueeze(1).unsqueeze(1).expand(-1, x.shape[1], x.shape[2], -1)
        elif len(text_context.shape) == 3:
            text_emb_seq = self.middle_text_proj(text_context)  # [B, text_len, D]
            text_emb = text_emb_seq.mean(dim=1, keepdim=True).unsqueeze(1).expand(-1, x.shape[1], x.shape[2], -1)


        if self.injection_mode == 'gating':
            gate = self.middle_gate_network(x)
            return x + gate * text_emb

        if self.injection_mode == 'cfa':        
            pooled_text = text_context if len(text_context.shape) == 2 else text_context.mean(dim=1)
            adapter_hidden = self.middle_adapter_down(pooled_text)
            adapter_hidden = self.middle_adapter_norm(adapter_hidden)
            adapter_hidden = self.middle_adapter_activation(adapter_hidden)
            adapter_out = self.middle_adapter_up(adapter_hidden).unsqueeze(1).unsqueeze(1).expand(-1, x.shape[1], x.shape[2], -1)
            return x + adapter_out

        if self.injection_mode == 'film':                
            pooled_text = text_context if len(text_context.shape) == 2 else text_context.mean(dim=1)
            gamma = self.middle_text_proj_gamma_film(pooled_text).unsqueeze(1).unsqueeze(1).expand(-1, x.shape[1], x.shape[2], -1)
            beta = self.middle_text_proj_beta_film(pooled_text).unsqueeze(1).unsqueeze(1).expand(-1, x.shape[1], x.shape[2], -1)
            return gamma * x + beta

        if self.injection_mode == 'orthogonal':                        
            pooled_text = text_context if len(text_context.shape) == 2 else text_context.mean(dim=1)
            ortho_text = self.middle_text_proj_ortho(pooled_text).unsqueeze(1).unsqueeze(1).expand(-1, x.shape[1], x.shape[2], -1)
            proj_coeff = (ortho_text * x).sum(dim=-1, keepdim=True) / (x.pow(2).sum(dim=-1, keepdim=True) + 1e-6)
            ortho_component = ortho_text - proj_coeff * x
            return x + ortho_component

        if self.injection_mode == 'middle-concat':
            concat = torch.cat([x, text_emb], dim=-1)  # [B, N, T, 2*D]
            return self.middle_concat_proj(concat)

        if self.injection_mode == 'middle-additive':
            return x + text_emb

        return x

    def forecast(self, x_enc, text_context=None):
        if self.injection_mode == 'unimodal':
            text_context = None

        # First fusion: inject into x_enc [B, T, N] before pipeline (shape preserved)
        if self.injection_mode in ('first-additive', 'first-concat'):
            if len(text_context.shape) == 2:
                text_emb = self.text_projection(text_context)  # [B, N]
                text_emb = text_emb.unsqueeze(1).expand(-1, x_enc.shape[1], -1)  # [B, T, N]
            else:
                text_emb = self.text_projection(text_context).mean(dim=1)
                text_emb = text_emb.unsqueeze(1).expand(-1, x_enc.shape[1], -1)

            if self.injection_mode == 'first-additive':
                x_enc = x_enc + text_emb
            if self.injection_mode == 'first-concat':
                concat = torch.cat([x_enc, text_emb], dim=-1)  # [B, T, 2*N]
                x_enc = self.first_concat_proj(concat)

        # x: [Batch, Input length, Channel]
        B, T, N = x_enc.shape
        x = self.tokenEmb(x_enc)
        bias = x
        if self.channel_independence == '0':
            x = self.MLP_channel(x, B, N, T)
        x = self.MLP_temporal(x, B, N, T)
        x = x + bias

        enc_text = None if self.injection_mode in ('first-additive', 'first-concat', 'last-additive', 'last-concat') else text_context
        if enc_text is not None:
            x = self._apply_middle_injection(x, enc_text)

        x = self.fc(x.reshape(B, N, -1)).permute(0, 2, 1)  # [B, pred_len, N]

        # Last fusion: inject into output [B, pred_len, N] (shape preserved)
        if self.injection_mode in ('last-additive', 'last-concat'):
            if len(text_context.shape) == 2:
                text_emb = self.text_projection(text_context)
                text_emb = text_emb.unsqueeze(1).expand(-1, x.shape[1], -1)
            else:
                text_emb = self.text_projection(text_context).mean(dim=1)
                text_emb = text_emb.unsqueeze(1).expand(-1, x.shape[1], -1)

            if self.injection_mode == 'last-additive':
                x = x + text_emb
            else:
                concat = torch.cat([x, text_emb], dim=-1)
                x = self.last_fusion_concat_proj(concat)

        return x

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None, text_context=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, text_context=text_context)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        else:
            raise ValueError('Only forecast tasks implemented yet')
