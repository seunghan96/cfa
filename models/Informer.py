import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.SelfAttention_Family import ProbAttention, AttentionLayer
from layers.Embed import DataEmbedding


class Model(nn.Module):
    """
    Informer with Propspare attention in O(LlogL) complexity
    Paper link: https://ojs.aaai.org/index.php/AAAI/article/view/17325/17132
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.label_len = configs.label_len

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

        # Embedding with text injection support
        self.enc_embedding = DataEmbedding(
            configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout,
            text_embedding_dim=text_embedding_dim, injection_mode=injection_mode,
            
        )
        self.dec_embedding = DataEmbedding(
            configs.dec_in, configs.d_model, configs.embed, configs.freq, configs.dropout,
            text_embedding_dim=text_embedding_dim, injection_mode=injection_mode,
            
        )

        # Encoder with text injection support
        encoder_layers = [
            EncoderLayer(
                AttentionLayer(
                    ProbAttention(False, configs.factor, attention_dropout=configs.dropout,
                                  output_attention=configs.output_attention),
                    configs.d_model, configs.n_heads),
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
            [
                ConvLayer(configs.d_model) for l in range(configs.e_layers - 1)
            ] if configs.distil and ('forecast' in configs.task_name) else None,
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            injection_mode=injection_mode
        )
        
      
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        ProbAttention(True, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    AttentionLayer(
                        ProbAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                    text_embedding_dim=None,
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

    def long_forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, text_context=None):
        # Naive mode: completely ignore text_context
        if self.injection_mode == 'unimodal':
            text_context = None

        enc_text = None if text_context is None or self.injection_mode in ('first-additive', 'first-concat', 'last-additive', 'last-concat') else text_context

        enc_out = self.enc_embedding(x_enc, x_mark_enc, text_context=text_context)
        dec_out = self.dec_embedding(x_dec, x_mark_dec, text_context=text_context)

        enc_out, attns = self.encoder(enc_out, attn_mask=None, text_context=enc_text)

        dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None, text_context=enc_text)

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
        return dec_out  # [B, L, D]
    


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None, text_context=None):
        if self.task_name == 'long_term_forecast':
            dec_out = self.long_forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, text_context=text_context)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'short_term_forecast':
            dec_out = self.short_forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, text_context=text_context)
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
