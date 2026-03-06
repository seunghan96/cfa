import torch
import torch.nn as nn
from layers.Autoformer_EncDec import series_decomp
from layers.Embed import DataEmbedding_wo_pos
from layers.StandardNorm import Normalize


class DFT_series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, top_k=5):
        super(DFT_series_decomp, self).__init__()
        self.top_k = top_k

    def forward(self, x):
        xf = torch.fft.rfft(x)
        freq = abs(xf)
        freq[0] = 0
        top_k_freq, top_list = torch.topk(freq, 5)
        xf[freq <= top_k_freq.min()] = 0
        x_season = torch.fft.irfft(xf)
        x_trend = x - x_season
        return x_season, x_trend


class MultiScaleSeasonMixing(nn.Module):
    """
    Bottom-up mixing season pattern
    """

    def __init__(self, configs):
        super(MultiScaleSeasonMixing, self).__init__()

        self.down_sampling_layers = torch.nn.ModuleList(
            [
                nn.Sequential(
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** i),
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                    ),
                    nn.GELU(),
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                    ),

                )
                for i in range(configs.down_sampling_layers)
            ]
        )

    def forward(self, season_list):
        # If only one scale, return as is
        if len(season_list) == 1:
            return [season_list[0].permute(0, 2, 1)]

        # mixing high->low
        out_high = season_list[0]
        out_low = season_list[1]
        out_season_list = [out_high.permute(0, 2, 1)]

        for i in range(len(season_list) - 1):
            out_low_res = self.down_sampling_layers[i](out_high)
            out_low = out_low + out_low_res
            out_high = out_low
            if i + 2 <= len(season_list) - 1:
                out_low = season_list[i + 2]
            out_season_list.append(out_high.permute(0, 2, 1))

        return out_season_list


class MultiScaleTrendMixing(nn.Module):
    """
    Top-down mixing trend pattern
    """

    def __init__(self, configs):
        super(MultiScaleTrendMixing, self).__init__()

        self.up_sampling_layers = torch.nn.ModuleList(
            [
                nn.Sequential(
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                        configs.seq_len // (configs.down_sampling_window ** i),
                    ),
                    nn.GELU(),
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** i),
                        configs.seq_len // (configs.down_sampling_window ** i),
                    ),
                )
                for i in reversed(range(configs.down_sampling_layers))
            ])

    def forward(self, trend_list):
        # If only one scale, return as is
        if len(trend_list) == 1:
            return [trend_list[0].permute(0, 2, 1)]

        # mixing low->high
        trend_list_reverse = trend_list.copy()
        trend_list_reverse.reverse()
        out_low = trend_list_reverse[0]
        out_high = trend_list_reverse[1]
        out_trend_list = [out_low.permute(0, 2, 1)]

        for i in range(len(trend_list_reverse) - 1):
            out_high_res = self.up_sampling_layers[i](out_low)
            out_high = out_high + out_high_res
            out_low = out_high
            if i + 2 <= len(trend_list_reverse) - 1:
                out_high = trend_list_reverse[i + 2]
            out_trend_list.append(out_low.permute(0, 2, 1))

        out_trend_list.reverse()
        return out_trend_list


class PastDecomposableMixing(nn.Module):
    def __init__(self, configs):
        super(PastDecomposableMixing, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.down_sampling_window = configs.down_sampling_window

        self.layer_norm = nn.LayerNorm(configs.d_model)
        self.dropout = nn.Dropout(configs.dropout)
        self.channel_independence = configs.channel_independence

        if configs.decomp_method == 'moving_avg':
            self.decompsition = series_decomp(configs.moving_avg)
        elif configs.decomp_method == "dft_decomp":
            self.decompsition = DFT_series_decomp(configs.top_k)
        else:
            raise ValueError('decompsition is error')

        if configs.channel_independence == 0:
            self.cross_layer = nn.Sequential(
                nn.Linear(in_features=configs.d_model, out_features=configs.d_ff),
                nn.GELU(),
                nn.Linear(in_features=configs.d_ff, out_features=configs.d_model),
            )

        # Mixing season
        self.mixing_multi_scale_season = MultiScaleSeasonMixing(configs)

        # Mxing trend
        self.mixing_multi_scale_trend = MultiScaleTrendMixing(configs)

        self.out_cross_layer = nn.Sequential(
            nn.Linear(in_features=configs.d_model, out_features=configs.d_ff),
            nn.GELU(),
            nn.Linear(in_features=configs.d_ff, out_features=configs.d_model),
        )

    def forward(self, x_list):
        length_list = []
        for x in x_list:
            _, T, _ = x.size()
            length_list.append(T)

        # Decompose to obtain the season and trend
        season_list = []
        trend_list = []
        for x in x_list:
            season, trend = self.decompsition(x)
            if self.channel_independence == 0:
                season = self.cross_layer(season)
                trend = self.cross_layer(trend)
            season_list.append(season.permute(0, 2, 1))
            trend_list.append(trend.permute(0, 2, 1))

        # bottom-up season mixing
        out_season_list = self.mixing_multi_scale_season(season_list)
        # top-down trend mixing
        out_trend_list = self.mixing_multi_scale_trend(trend_list)

        out_list = []
        for ori, out_season, out_trend, length in zip(x_list, out_season_list, out_trend_list,
                                                      length_list):
            out = out_season + out_trend
            if self.channel_independence:
                out = ori + self.out_cross_layer(out)
            out_list.append(out[:, :length, :])
        return out_list


class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.down_sampling_window = configs.down_sampling_window
        self.channel_independence = configs.channel_independence
        self.pdm_blocks = nn.ModuleList([PastDecomposableMixing(configs)
                                         for _ in range(configs.e_layers)])

        self.preprocess = series_decomp(configs.moving_avg)
        self.enc_in = configs.enc_in

        if self.channel_independence == 1:
            self.enc_embedding = DataEmbedding_wo_pos(1, configs.d_model, configs.embed, configs.freq,
                                                      configs.dropout)
        else:
            self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                      configs.dropout)

        self.layer = configs.e_layers
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.predict_layers = torch.nn.ModuleList(
                [
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** i),
                        configs.pred_len,
                    )
                    for i in range(configs.down_sampling_layers + 1)
                ]
            )

            if self.channel_independence == 1:
                self.projection_layer = nn.Linear(
                    configs.d_model, 1, bias=True)
            else:
                self.projection_layer = nn.Linear(
                    configs.d_model, configs.c_out, bias=True)

                self.out_res_layers = torch.nn.ModuleList([
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** i),
                        configs.seq_len // (configs.down_sampling_window ** i),
                    )
                    for i in range(configs.down_sampling_layers + 1)
                ])

                self.regression_layers = torch.nn.ModuleList(
                    [
                        torch.nn.Linear(
                            configs.seq_len // (configs.down_sampling_window ** i),
                            configs.pred_len,
                        )
                        for i in range(configs.down_sampling_layers + 1)
                    ]
                )

        # Text injection configuration (need to define early for Normalize initialization)
        text_embedding_dim = getattr(configs, 'text_emb', None)
        injection_mode = getattr(configs, 'text_injection_mode')
        self.injection_mode = injection_mode
        self.channels = configs.enc_in
        self.middle_like_modes = ('gating', 'cfa', 'film', 'orthogonal', 'middle-additive', 'middle-concat')

        # Shared text -> channels (first/middle/last fusion)
        if text_embedding_dim is not None and injection_mode != 'unimodal':
            self.middle_text_proj = nn.Linear(text_embedding_dim, self.channels, bias=False)
        else:
            self.middle_text_proj = None

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

        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.normalize_layers = torch.nn.ModuleList(
                [
                    Normalize(self.configs.enc_in, affine=True, non_norm=True if configs.use_norm == 0 else False,
                             text_embedding_dim=None,
                             injection_mode=injection_mode)
                    for i in range(configs.down_sampling_layers + 1)
                ]
            )
        


    def out_projection(self, dec_out, i, out_res):
        dec_out = self.projection_layer(dec_out)
        out_res = out_res.permute(0, 2, 1)
        out_res = self.out_res_layers[i](out_res)
        out_res = self.regression_layers[i](out_res).permute(0, 2, 1)
        dec_out = dec_out + out_res
        return dec_out

    def pre_enc(self, x_list):
        if self.channel_independence == 1:
            return (x_list, None)
        else:
            out1_list = []
            out2_list = []
            for x in x_list:
                x_1, x_2 = self.preprocess(x)
                out1_list.append(x_1)
                out2_list.append(x_2)
            return (out1_list, out2_list)

    def __multi_scale_process_inputs(self, x_enc, x_mark_enc):
        if self.configs.down_sampling_method == 'max':
            down_pool = torch.nn.MaxPool1d(self.configs.down_sampling_window, return_indices=False)
        elif self.configs.down_sampling_method == 'avg':
            down_pool = torch.nn.AvgPool1d(self.configs.down_sampling_window)
        elif self.configs.down_sampling_method == 'conv':
            padding = 1 if torch.__version__ >= '1.5.0' else 2
            down_pool = nn.Conv1d(in_channels=self.configs.enc_in, out_channels=self.configs.enc_in,
                                  kernel_size=3, padding=padding,
                                  stride=self.configs.down_sampling_window,
                                  padding_mode='circular',
                                  bias=False)
        else:
            # If down_sampling_method is not set, return as a list with single element
            return [x_enc], [x_mark_enc] if x_mark_enc is not None else None
        # B,T,C -> B,C,T
        x_enc = x_enc.permute(0, 2, 1)

        x_enc_ori = x_enc
        x_mark_enc_mark_ori = x_mark_enc

        x_enc_sampling_list = []
        x_mark_sampling_list = []
        x_enc_sampling_list.append(x_enc.permute(0, 2, 1))
        x_mark_sampling_list.append(x_mark_enc)

        for i in range(self.configs.down_sampling_layers):
            x_enc_sampling = down_pool(x_enc_ori)

            x_enc_sampling_list.append(x_enc_sampling.permute(0, 2, 1))
            x_enc_ori = x_enc_sampling

            if x_mark_enc is not None:
                x_mark_sampling_list.append(x_mark_enc_mark_ori[:, ::self.configs.down_sampling_window, :])
                x_mark_enc_mark_ori = x_mark_enc_mark_ori[:, ::self.configs.down_sampling_window, :]

        x_enc = x_enc_sampling_list
        x_mark_enc = x_mark_sampling_list if x_mark_enc is not None else None

        return x_enc, x_mark_enc

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, text_context=None):
        if self.injection_mode == 'unimodal':
            text_context = None

        # First fusion: inject into x_enc [B, L, C] before multi-scale processing (shape preserved)
        if self.injection_mode in ('first-additive', 'first-concat') and self.middle_text_proj is not None:
            if len(text_context.shape) == 2:
                text_emb = self.middle_text_proj(text_context)
                text_emb = text_emb.unsqueeze(1).expand(-1, x_enc.shape[1], -1)
            else:
                text_emb = self.middle_text_proj(text_context.mean(dim=1))
                text_emb = text_emb.unsqueeze(1).expand(-1, x_enc.shape[1], -1)
            if self.injection_mode == 'first-additive':
                x_enc = x_enc + text_emb
            else:
                x_enc = self.first_concat_proj(torch.cat([x_enc, text_emb], dim=-1))

        x_enc, x_mark_enc = self.__multi_scale_process_inputs(x_enc, x_mark_enc)

        x_list = []
        x_mark_list = []
        norm_text_ctx = text_context if self.injection_mode in self.middle_like_modes else None
        if x_mark_enc is not None:
            for i, x, x_mark in zip(range(len(x_enc)), x_enc, x_mark_enc):
                B, T, N = x.size()
                x = self.normalize_layers[i](x, 'norm', text_context=norm_text_ctx)
                if self.channel_independence == 1:
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                x_list.append(x)
                x_mark = x_mark.repeat(N, 1, 1)
                x_mark_list.append(x_mark)
        else:
            for i, x in zip(range(len(x_enc)), x_enc, ):
                B, T, N = x.size()
                x = self.normalize_layers[i](x, 'norm', text_context=norm_text_ctx)
                if self.channel_independence == 1:
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                x_list.append(x)

        # embedding
        enc_out_list = []
        x_list = self.pre_enc(x_list)
        if x_mark_enc is not None:
            for i, x, x_mark in zip(range(len(x_list[0])), x_list[0], x_mark_list):
                enc_out = self.enc_embedding(x, x_mark)  # [B,T,C]
                enc_out_list.append(enc_out)
        else:
            for i, x in zip(range(len(x_list[0])), x_list[0]):
                enc_out = self.enc_embedding(x, None)  # [B,T,C]
                enc_out_list.append(enc_out)

        # Past Decomposable Mixing as encoder for past
        for i in range(self.layer):
            enc_out_list = self.pdm_blocks[i](enc_out_list)

        # Future Multipredictor Mixing as decoder for future
        dec_out_list = self.future_multi_mixing(B, enc_out_list, x_list)

        dec_out = torch.stack(dec_out_list, dim=-1).sum(-1)
        norm_text_ctx = text_context if self.injection_mode in self.middle_like_modes else None
        enc_text = None if self.injection_mode in ('first-additive', 'first-concat', 'last-additive', 'last-concat') else text_context
        dec_out = self._apply_middle_injection(dec_out, enc_text)
        dec_out = self.normalize_layers[0](dec_out, 'denorm', text_context=norm_text_ctx)

        # Last fusion: inject into output [B, pred_len, C] (shape preserved)
        if self.injection_mode in ('last-additive', 'last-concat') and self.middle_text_proj is not None:
            if len(text_context.shape) == 2:
                text_emb = self.middle_text_proj(text_context)
                text_emb = text_emb.unsqueeze(1).expand(-1, dec_out.shape[1], -1)
            else:
                text_emb = self.middle_text_proj(text_context.mean(dim=1))
                text_emb = text_emb.unsqueeze(1).expand(-1, dec_out.shape[1], -1)
            if self.injection_mode == 'last-additive':
                dec_out = dec_out + text_emb
            else:
                dec_out = self.last_fusion_concat_proj(torch.cat([dec_out, text_emb], dim=-1))

        return dec_out

    def _apply_middle_injection(self, x, text_context):
        """Apply middle fusion on dec_out [B, L, C] before denorm"""
        if text_context is None or self.middle_text_proj is None:
            return x

        if len(text_context.shape) == 2:
            text_emb = self.middle_text_proj(text_context).unsqueeze(1).expand(-1, x.shape[1], -1)
        elif len(text_context.shape) == 3:
            text_emb_seq = self.middle_text_proj(text_context)
            text_emb = text_emb_seq.mean(dim=1, keepdim=True).expand(-1, x.shape[1], -1)
        else:
            return x

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

    def future_multi_mixing(self, B, enc_out_list, x_list):
        dec_out_list = []
        if self.channel_independence == 1:
            x_list = x_list[0]
            for i, enc_out in zip(range(len(x_list)), enc_out_list):
                dec_out = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(
                    0, 2, 1)  # align temporal dimension
                dec_out = self.projection_layer(dec_out)
                dec_out = dec_out.reshape(B, self.configs.c_out, self.pred_len).permute(0, 2, 1).contiguous()
                dec_out_list.append(dec_out)

        else:
            for i, enc_out, out_res in zip(range(len(x_list[0])), enc_out_list, x_list[1]):
                dec_out = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(
                    0, 2, 1)  # align temporal dimension
                dec_out = self.out_projection(dec_out, i, out_res)
                dec_out_list.append(dec_out)

        return dec_out_list

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None, text_context=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out_list = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, text_context=text_context)
            return dec_out_list
        else:
            raise ValueError('Only forecast tasks implemented yet')
