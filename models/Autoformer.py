import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding_wo_pos
from layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp
import math
import numpy as np


class Model(nn.Module):
    """
    Autoformer is the first method to achieve the series-wise connection,
    with inherent O(LlogL) complexity
    Paper link: https://openreview.net/pdf?id=I55UqU-M11y
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention

        # Text injection configuration
        text_embedding_dim = getattr(configs, 'text_emb', None)
        injection_mode = getattr(configs, 'text_injection_mode')
        self.channels = configs.c_out

        # Last fusion: in Autoformer (after decoder output)
        if injection_mode in ('last-additive', 'last-concat') and text_embedding_dim is not None:
            self.text_projection = nn.Linear(text_embedding_dim, self.channels, bias=False)
        else:
            self.text_projection = None
        if injection_mode == 'last-concat':
            self.last_fusion_concat_proj = nn.Linear(2 * self.channels, self.channels, bias=True)
        else:
            self.last_fusion_concat_proj = None

        # Decomp
        kernel_size = configs.moving_avg
        self.decomp = series_decomp(kernel_size)

        # Embedding with text injection support
        self.enc_embedding = DataEmbedding_wo_pos(
            configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout,
            text_embedding_dim=text_embedding_dim, injection_mode=injection_mode,
            
        )
        # Encoder
        self.injection_mode = injection_mode
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=configs.output_attention),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model),
            injection_mode=injection_mode,
            text_embedding_dim=text_embedding_dim,
            cfa_reduction=configs.cfa_reduction
        )


        # Decoder
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.dec_embedding = DataEmbedding_wo_pos(
                configs.dec_in, configs.d_model, configs.embed, configs.freq, configs.dropout,
                text_embedding_dim=text_embedding_dim, injection_mode=injection_mode,
                
            )
            self.decoder = Decoder(
                [
                    DecoderLayer(
                        AutoCorrelationLayer(
                            AutoCorrelation(True, configs.factor, attention_dropout=configs.dropout,
                                            output_attention=False),
                            configs.d_model, configs.n_heads),
                        AutoCorrelationLayer(
                            AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout,
                                            output_attention=False),
                            configs.d_model, configs.n_heads),
                        configs.d_model,
                        configs.c_out,
                        configs.d_ff,
                        moving_avg=configs.moving_avg,
                        dropout=configs.dropout,
                        activation=configs.activation,
                    )
                    for l in range(configs.d_layers)
                ],
                norm_layer=my_Layernorm(configs.d_model),
                projection=nn.Linear(configs.d_model, configs.c_out, bias=True),
                injection_mode=injection_mode,
                text_embedding_dim=text_embedding_dim,
                cfa_reduction=configs.cfa_reduction
            )
        if self.task_name == 'imputation':
            self.projection = nn.Linear(
                configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(
                configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                configs.d_model * configs.seq_len, configs.num_class)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, text_context=None):
        if self.injection_mode == 'unimodal':
            text_context = None

        # decomp init
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]], device=x_enc.device)
        seasonal_init, trend_init = self.decomp(x_enc)
        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_init = torch.cat([seasonal_init[:, -self.label_len:, :], zeros], dim=1)

        # enc: first fusion in Embed when first-additive/first-concat; middle in Encoder when enc_text passed
        enc_text = None if self.injection_mode in ('first-additive', 'first-concat', 'last-additive', 'last-concat') else text_context
        enc_out = self.enc_embedding(x_enc, x_mark_enc, text_context=text_context)
        enc_out, attns = self.encoder(enc_out, attn_mask=None, text_context=enc_text)

        # dec: first fusion in Embed; middle in Decoder when enc_text passed
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec, text_context=text_context)
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None,
                                                 trend=trend_init, text_context=enc_text)
        dec_out = trend_part + seasonal_part

        # Last fusion: inject into output [B, pred_len, c_out] (in Autoformer)
        if self.injection_mode in ('last-additive', 'last-concat') and self.text_projection is not None:
            if len(text_context.shape) == 2:
                text_emb = self.text_projection(text_context).unsqueeze(1).expand(-1, dec_out.shape[1], -1)
            else:
                text_emb = self.text_projection(text_context.mean(dim=1)).unsqueeze(1).expand(-1, dec_out.shape[1], -1)
            if self.injection_mode == 'last-additive':
                dec_out = dec_out + text_emb
            else:
                dec_out = self.last_fusion_concat_proj(torch.cat([dec_out, text_emb], dim=-1))

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
