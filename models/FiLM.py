import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import signal
from scipy import special as ss

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def transition(N):
    Q = np.arange(N, dtype=np.float64)
    R = (2 * Q + 1)[:, None]  # / theta
    j, i = np.meshgrid(Q, Q)
    A = np.where(i < j, -1, (-1.) ** (i - j + 1)) * R
    B = (-1.) ** Q[:, None] * R
    return A, B


class HiPPO_LegT(nn.Module):
    def __init__(self, N, dt=1.0, discretization='bilinear'):
        """
        N: the order of the HiPPO projection
        dt: discretization step size - should be roughly inverse to the length of the sequence
        """
        super(HiPPO_LegT, self).__init__()
        self.N = N
        A, B = transition(N)
        C = np.ones((1, N))
        D = np.zeros((1,))
        A, B, _, _, _ = signal.cont2discrete((A, B, C, D), dt=dt, method=discretization)

        B = B.squeeze(-1)

        self.register_buffer('A', torch.Tensor(A).to(device))
        self.register_buffer('B', torch.Tensor(B).to(device))
        vals = np.arange(0.0, 1.0, dt)
        self.register_buffer('eval_matrix', torch.Tensor(
            ss.eval_legendre(np.arange(N)[:, None], 1 - 2 * vals).T).to(device))

    def forward(self, inputs):
        """
        inputs : (length, ...)
        output : (length, ..., N) where N is the order of the HiPPO projection
        """
        c = torch.zeros(inputs.shape[:-1] + tuple([self.N])).to(device)
        cs = []
        for f in inputs.permute([-1, 0, 1]):
            f = f.unsqueeze(-1)
            new = f @ self.B.unsqueeze(0)
            c = F.linear(c, self.A) + new
            cs.append(c)
        return torch.stack(cs, dim=0)

    def reconstruct(self, c):
        return (self.eval_matrix @ c.unsqueeze(-1)).squeeze(-1)


class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, seq_len, ratio=0.5):
        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        """
        super(SpectralConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ratio = ratio
        self.modes = min(32, seq_len // 2)
        self.index = list(range(0, self.modes))

        self.scale = (1 / (in_channels * out_channels))
        self.weights_real = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, len(self.index), dtype=torch.float))
        self.weights_imag = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, len(self.index), dtype=torch.float))

    def compl_mul1d(self, order, x, weights_real, weights_imag):
        return torch.complex(torch.einsum(order, x.real, weights_real) - torch.einsum(order, x.imag, weights_imag),
                                 torch.einsum(order, x.real, weights_imag) + torch.einsum(order, x.imag, weights_real))

    def forward(self, x):
        B, H, E, N = x.shape
        x_ft = torch.fft.rfft(x)
        out_ft = torch.zeros(B, H, self.out_channels, x.size(-1) // 2 + 1, device=x.device, dtype=torch.cfloat)
        a = x_ft[:, :, :, :self.modes]
        out_ft[:, :, :, :self.modes] = self.compl_mul1d("bjix,iox->bjox", a, self.weights_real, self.weights_imag)
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2205.08897
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.configs = configs
        # self.modes = configs.modes
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.seq_len if configs.pred_len == 0 else configs.pred_len

        self.seq_len_all = self.seq_len + self.label_len

        self.output_attention = configs.output_attention
        self.layers = configs.e_layers
        self.enc_in = configs.enc_in
        self.e_layers = configs.e_layers
        
        # Text injection configuration (needs to be defined early)
        text_embedding_dim = getattr(configs, 'text_emb', None)
        injection_mode = getattr(configs, 'text_injection_mode')
        self.injection_mode = injection_mode
        self.channels = configs.enc_in

        # Shared text -> channels (first/middle/last fusion)
        if text_embedding_dim is not None and injection_mode != 'unimodal':
            self.text_projection = nn.Linear(text_embedding_dim, self.channels, bias=False)
        else:
            self.text_projection = None

        if injection_mode == 'first-concat':
            self.first_concat_proj = nn.Linear(2 * self.channels, self.channels, bias=True)
        else:
            self.first_concat_proj = None

        if injection_mode == 'middle-concat':
            self.middle_concat_proj = nn.Linear(2 * self.channels, self.channels, bias=True)
        else:
            self.middle_concat_proj = None

        if injection_mode == 'last-concat':
            self.last_fusion_concat_proj = nn.Linear(2 * self.channels, self.channels, bias=True)
        else:
            self.last_fusion_concat_proj = None

        # b, s, f means b, f
        # Default affine parameters for instance-wise normalization
        self.affine_weight = nn.Parameter(torch.ones(1, 1, configs.enc_in))
        self.affine_bias = nn.Parameter(torch.zeros(1, 1, configs.enc_in))

        if injection_mode == 'gating':
            self.middle_gate_network = nn.Sequential(
                nn.Linear(self.channels, self.channels, bias=True),
                nn.ReLU(),
                nn.Dropout(configs.dropout),
                nn.Linear(self.channels, self.channels, bias=True),
                nn.Sigmoid()
            )
        else:
            self.middle_gate_network = None

        if injection_mode == 'cfa':
            adapter_bottleneck = max(1, self.channels // configs.cfa_reduction)
            self.middle_adapter_down = nn.Linear(text_embedding_dim, adapter_bottleneck, bias=False)
            self.middle_adapter_norm = nn.LayerNorm(adapter_bottleneck)
            self.middle_adapter_activation = nn.ReLU()
            self.middle_adapter_up = nn.Linear(adapter_bottleneck, self.channels, bias=False)
        else:
            self.middle_adapter_down = None
            self.middle_adapter_norm = None
            self.middle_adapter_activation = None
            self.middle_adapter_up = None

        if injection_mode == 'film':
            self.middle_text_proj_gamma_film = nn.Linear(text_embedding_dim, self.channels, bias=True)
            self.middle_text_proj_beta_film = nn.Linear(text_embedding_dim, self.channels, bias=True)
            nn.init.ones_(self.middle_text_proj_gamma_film.bias)
            nn.init.zeros_(self.middle_text_proj_beta_film.bias)
        else:
            self.middle_text_proj_gamma_film = None
            self.middle_text_proj_beta_film = None

        if injection_mode == 'orthogonal':
            self.middle_text_proj_ortho = nn.Linear(text_embedding_dim, self.channels, bias=False)
        else:
            self.middle_text_proj_ortho = None
        


        self.multiscale = [1, 2, 4]
        self.window_size = [256]
        configs.ratio = 0.5
        self.legts = nn.ModuleList(
            [HiPPO_LegT(N=n, dt=1. / self.pred_len / i) for n in self.window_size for i in self.multiscale])
        self.spec_conv_1 = nn.ModuleList([SpectralConv1d(in_channels=n, out_channels=n,
                                                         seq_len=min(self.pred_len, self.seq_len),
                                                         ratio=configs.ratio) for n in
                                          self.window_size for _ in range(len(self.multiscale))])
        self.mlp = nn.Linear(len(self.multiscale) * len(self.window_size), 1)

        if self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(
                configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                configs.enc_in * configs.seq_len, configs.num_class)
     
        

    def _apply_affine_transform(self, x, text_context):
        """Apply instance-wise normalization with optional text-based modifiers (middle-inst)"""
        return x * self.affine_weight + self.affine_bias
    
    def _reverse_affine_transform(self, x, text_context):
        """Reverse instance-wise normalization with optional text-based modifiers (middle-inst)"""
        return (x - self.affine_bias) / (self.affine_weight + 1e-10)

    
    def _apply_middle_injection(self, x, text_context):
        """Apply text injection at the middle stage. unimodal: text_context is None, return x unchanged."""
        if text_context is None or self.text_projection is None:
            return x
        if len(text_context.shape) == 2:
            text_emb = self.text_projection(text_context)
            text_emb = text_emb.unsqueeze(1).expand(-1, x.shape[1], -1)
        else:
            text_emb = self.text_projection(text_context.mean(dim=1))
            text_emb = text_emb.unsqueeze(1).expand(-1, x.shape[1], -1)

        if self.injection_mode == 'gating':
            gate = self.middle_gate_network(x)
            return x + gate * text_emb

        if self.injection_mode == 'cfa':
            pooled_text = text_context if len(text_context.shape) == 2 else text_context.mean(dim=1)
            adapter_hidden = self.middle_adapter_down(pooled_text)
            adapter_hidden = self.middle_adapter_norm(adapter_hidden)
            adapter_hidden = self.middle_adapter_activation(adapter_hidden)
            adapter_out = self.middle_adapter_up(adapter_hidden).unsqueeze(1).expand(-1, x.shape[1], -1)
            return x + adapter_out

        if self.injection_mode == 'film':
            pooled_text = text_context if len(text_context.shape) == 2 else text_context.mean(dim=1)
            gamma = self.middle_text_proj_gamma_film(pooled_text).unsqueeze(1).expand(-1, x.shape[1], -1)
            beta = self.middle_text_proj_beta_film(pooled_text).unsqueeze(1).expand(-1, x.shape[1], -1)
            return gamma * x + beta

        if self.injection_mode == 'orthogonal':
            pooled_text = text_context if len(text_context.shape) == 2 else text_context.mean(dim=1)
            ortho_text = self.middle_text_proj_ortho(pooled_text).unsqueeze(1).expand(-1, x.shape[1], -1)
            proj_coeff = (ortho_text * x).sum(dim=-1, keepdim=True) / (x.pow(2).sum(dim=-1, keepdim=True) + 1e-6)
            ortho_component = ortho_text - proj_coeff * x
            return x + ortho_component

        if self.injection_mode == 'middle-additive':
            return x + text_emb
        if self.injection_mode == 'middle-concat':
            concat = torch.cat([x, text_emb], dim=-1)
            return self.middle_concat_proj(concat)

        return x
        


    def forecast(self, x_enc, x_mark_enc, x_dec_true, x_mark_dec, text_context=None):
        if self.injection_mode == 'unimodal':
            text_context = None

        # First fusion: inject into x_enc [B, L, C] before processing (shape preserved)
        if self.injection_mode in ('first-additive', 'first-concat') and self.text_projection is not None:
            if len(text_context.shape) == 2:
                text_emb = self.text_projection(text_context)
                text_emb = text_emb.unsqueeze(1).expand(-1, x_enc.shape[1], -1)
            else:
                text_emb = self.text_projection(text_context.mean(dim=1))
                text_emb = text_emb.unsqueeze(1).expand(-1, x_enc.shape[1], -1)
            if self.injection_mode == 'first-additive':
                x_enc = x_enc + text_emb
            else:
                x_enc = self.first_concat_proj(torch.cat([x_enc, text_emb], dim=-1))

        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_enc /= stdev

        x_enc = self._apply_affine_transform(x_enc, text_context)

        enc_text = None if self.injection_mode in ('first-additive', 'first-concat', 'last-additive', 'last-concat') else text_context
        x_enc = self._apply_middle_injection(x_enc, enc_text)
        
        x_decs = []
        jump_dist = 0
        for i in range(0, len(self.multiscale) * len(self.window_size)):
            x_in_len = self.multiscale[i % len(self.multiscale)] * self.pred_len
            x_in = x_enc[:, -x_in_len:]
            legt = self.legts[i]
            x_in_c = legt(x_in.transpose(1, 2)).permute([1, 2, 3, 0])[:, :, :, jump_dist:]
            out1 = self.spec_conv_1[i](x_in_c)
            if self.seq_len >= self.pred_len:
                x_dec_c = out1.transpose(2, 3)[:, :, self.pred_len - 1 - jump_dist, :]
            else:
                x_dec_c = out1.transpose(2, 3)[:, :, -1, :]
            x_dec = x_dec_c @ legt.eval_matrix[-self.pred_len:, :].T
            x_decs.append(x_dec)
        x_dec = torch.stack(x_decs, dim=-1)
        x_dec = self.mlp(x_dec).squeeze(-1).permute(0, 2, 1)

        # De-Normalization from Non-stationary Transformer
        x_dec = self._reverse_affine_transform(x_dec, text_context)
        x_dec = x_dec * stdev
        x_dec = x_dec + means

        # Last fusion: inject into output [B, pred_len, C] (shape preserved)
        if self.injection_mode in ('last-additive', 'last-concat') and self.text_projection is not None:
            if len(text_context.shape) == 2:
                text_emb = self.text_projection(text_context)
                text_emb = text_emb.unsqueeze(1).expand(-1, x_dec.shape[1], -1)
            else:
                text_emb = self.text_projection(text_context.mean(dim=1))
                text_emb = text_emb.unsqueeze(1).expand(-1, x_dec.shape[1], -1)
            if self.injection_mode == 'last-additive':
                x_dec = x_dec + text_emb
            else:
                x_dec = self.last_fusion_concat_proj(torch.cat([x_dec, text_emb], dim=-1))

        return x_dec

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None, text_context=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, text_context=text_context)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask, text_context=text_context)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc, text_context=text_context)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc, text_context=text_context)
            return dec_out  # [B, N]
        return None
