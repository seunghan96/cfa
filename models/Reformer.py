import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import ReformerLayer
from layers.Embed import DataEmbedding


class Model(nn.Module):
    """
    Reformer with O(LlogL) complexity
    Paper link: https://openreview.net/forum?id=rkgNKkHtvB
    """

    def __init__(self, configs, bucket_size=4, n_hashes=4):
        """
        bucket_size: int, 
        n_hashes: int, 
        """
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len

        # Text injection configuration
        text_embedding_dim = getattr(configs, 'text_emb', None)
        injection_mode = getattr(configs, 'text_injection_mode')



        self.enc_embedding = DataEmbedding(
            configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout,
            text_embedding_dim=text_embedding_dim, injection_mode=injection_mode,
            
        )
        # Encoder
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

        encoder_layers = [
            EncoderLayer(
                ReformerLayer(None, configs.d_model, configs.n_heads,
                              bucket_size=bucket_size, n_hashes=n_hashes),
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
        


        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                configs.d_model * configs.seq_len, configs.num_class)
        else:
            self.projection = nn.Linear(
                configs.d_model, configs.c_out, bias=True)



    def long_forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, text_context=None):
        if self.injection_mode == 'unimodal':
            text_context = None

        enc_text = None if text_context is None or self.injection_mode in ('first-additive', 'first-concat', 'last-additive', 'last-concat') else text_context
        
        # add placeholder
        x_enc = torch.cat([x_enc, x_dec[:, -self.pred_len:, :]], dim=1)
        if x_mark_enc is not None:
            x_mark_enc = torch.cat(
                [x_mark_enc, x_mark_dec[:, -self.pred_len:, :]], dim=1)

        enc_out = self.enc_embedding(x_enc, x_mark_enc, text_context=text_context)  # [B,T,C]
        enc_out, attns = self.encoder(enc_out, attn_mask=None, text_context=enc_text)
        dec_out = self.projection(enc_out)

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
            dec_out = self.imputation(x_enc, x_mark_enc, text_context=text_context)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc, text_context=text_context)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc, text_context=text_context)
            return dec_out  # [B, N]
        return None
