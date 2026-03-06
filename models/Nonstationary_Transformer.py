import torch
import torch.nn as nn
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer
from layers.SelfAttention_Family import DSAttention, AttentionLayer
from layers.Embed import DataEmbedding
import torch.nn.functional as F


class Projector(nn.Module):
    '''
    MLP to learn the De-stationary factors
    Paper link: https://openreview.net/pdf?id=ucNDIDRNjjv
    '''

    def __init__(self, enc_in, seq_len, hidden_dims, hidden_layers, output_dim, kernel_size=3):
        super(Projector, self).__init__()

        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.series_conv = nn.Conv1d(in_channels=seq_len, out_channels=1, kernel_size=kernel_size, padding=padding,
                                     padding_mode='circular', bias=False)

        layers = [nn.Linear(2 * enc_in, hidden_dims[0]), nn.ReLU()]
        for i in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_dims[i], hidden_dims[i + 1]), nn.ReLU()]

        layers += [nn.Linear(hidden_dims[-1], output_dim, bias=False)]
        self.backbone = nn.Sequential(*layers)

    def forward(self, x, stats):
        # x:     B x S x E
        # stats: B x 1 x E
        # y:     B x O
        batch_size = x.shape[0]
        x = self.series_conv(x)  # B x 1 x E
        x = torch.cat([x, stats], dim=1)  # B x 2 x E
        x = x.view(batch_size, -1)  # B x 2E
        y = self.backbone(x)  # B x O

        return y


class Model(nn.Module):
    """
    Paper link: https://openreview.net/pdf?id=ucNDIDRNjjv
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.output_attention = configs.output_attention

        # Text injection configuration
        text_embedding_dim = getattr(configs, 'text_emb', None)
        injection_mode = getattr(configs, 'text_injection_mode', 'last')
        self.injection_mode = injection_mode
        self.channels = configs.c_out

        if text_embedding_dim is not None and injection_mode != 'unimodal':
            self.text_projection = nn.Linear(text_embedding_dim, self.channels, bias=False)
        else:
            self.text_projection = None
        if injection_mode == 'last-concat':
            self.last_fusion_concat_proj = nn.Linear(2 * self.channels, self.channels, bias=True)
        else:
            self.last_fusion_concat_proj = None

        # Embedding
        self.enc_embedding = DataEmbedding(
            configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout,
            text_embedding_dim=text_embedding_dim,
            injection_mode=injection_mode,
        )

        # Encoder
        encoder_layers = [
            EncoderLayer(
                AttentionLayer(
                    DSAttention(False, configs.factor, attention_dropout=configs.dropout,
                                output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                configs.d_model,
                configs.d_ff,
                dropout=configs.dropout,
                activation=configs.activation,
                text_embedding_dim=text_embedding_dim,
                injection_mode=injection_mode,
                cfa_reduction=configs.cfa_reduction
            ) for l in range(configs.e_layers)
        ]
        self.encoder = Encoder(
            encoder_layers,
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            injection_mode=injection_mode
        )
        # Decoder
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.dec_embedding = DataEmbedding(
                configs.dec_in, configs.d_model, configs.embed, configs.freq, configs.dropout,
                text_embedding_dim=text_embedding_dim,
                injection_mode=injection_mode,
            )
            self.decoder = Decoder(
                [
                    DecoderLayer(
                        AttentionLayer(
                            DSAttention(True, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=False),
                            configs.d_model, configs.n_heads),
                        AttentionLayer(
                            DSAttention(False, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=False),
                            configs.d_model, configs.n_heads),
                        configs.d_model,
                        configs.d_ff,
                        dropout=configs.dropout,
                        activation=configs.activation,
                        text_embedding_dim=text_embedding_dim,
                        injection_mode=injection_mode
                    )
                    for l in range(configs.d_layers)
                ],
                norm_layer=torch.nn.LayerNorm(configs.d_model),
                projection=nn.Linear(configs.d_model, configs.c_out, bias=True),
                injection_mode=injection_mode,
                text_embedding_dim=text_embedding_dim
            )
        if self.task_name == 'imputation':
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(configs.d_model * configs.seq_len, configs.num_class)

        self.tau_learner = Projector(enc_in=configs.enc_in, seq_len=configs.seq_len, hidden_dims=configs.p_hidden_dims,
                                     hidden_layers=configs.p_hidden_layers, output_dim=1)
        self.delta_learner = Projector(enc_in=configs.enc_in, seq_len=configs.seq_len,
                                       hidden_dims=configs.p_hidden_dims, hidden_layers=configs.p_hidden_layers,
                                       output_dim=configs.seq_len)
        



    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, text_context=None):
        # Naive mode: completely ignore text_context
        if self.injection_mode == 'unimodal':
            text_context = None

        enc_text = None if text_context is None or self.injection_mode in ('first-additive', 'first-concat', 'last-additive', 'last-concat') else text_context

        x_raw = x_enc.clone().detach()

        # Normalization
        mean_enc = x_enc.mean(1, keepdim=True).detach()  # B x 1 x E
        x_enc = x_enc - mean_enc
        std_enc = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()  # B x 1 x E
        x_enc = x_enc / std_enc
        # B x S x E, B x 1 x E -> B x 1, positive scalar
        tau = self.tau_learner(x_raw, std_enc).exp()
        # B x S x E, B x 1 x E -> B x S
        delta = self.delta_learner(x_raw, mean_enc)

        x_dec_new = torch.cat([x_enc[:, -self.label_len:, :], torch.zeros_like(x_dec[:, -self.pred_len:, :])],
                              dim=1).to(x_enc.device).clone()

        enc_out = self.enc_embedding(x_enc, x_mark_enc, text_context=text_context)
        enc_out, attns = self.encoder(enc_out, attn_mask=None, tau=tau, delta=delta, text_context=enc_text)

        dec_out = self.dec_embedding(x_dec_new, x_mark_dec, text_context=text_context)
        dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None, tau=tau, delta=delta, text_context=enc_text)

        dec_out = dec_out * std_enc + mean_enc

        # Last fusion: inject into dec_out [B, L, c_out]
        if text_context is not None and self.text_projection is not None and self.injection_mode in ('last-additive', 'last-concat'):
            if len(text_context.shape) == 2:
                text_emb = self.text_projection(text_context)
            else:
                text_emb = self.text_projection(text_context.mean(dim=1))
            text_emb = text_emb.unsqueeze(1).expand(-1, dec_out.shape[1], -1)
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
            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask, text_context=text_context)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc, text_context=text_context)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc, text_context=text_context)
            return dec_out  # [B, L, D]
        return None
