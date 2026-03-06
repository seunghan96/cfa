import torch
from torch import nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2211.14730.pdf
    """

    def __init__(self, configs, patch_len=16, stride=8):
        """
        patch_len: int, patch len for patch_embedding
        stride: int, stride for patch_embedding
        """
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.c_out
        
        # Text injection configuration
        text_embedding_dim = getattr(configs, 'text_emb', None)
        injection_mode = getattr(configs, 'text_injection_mode')
        
       
        if text_embedding_dim is not None and injection_mode != 'unimodal':
            self.text_projection = nn.Linear(text_embedding_dim, self.channels, bias=False)
        else:
            self.text_projection = None
        if injection_mode == 'last-concat':
            self.last_fusion_concat_proj = nn.Linear(2 * self.channels, self.channels, bias=True)
        else:
            self.last_fusion_concat_proj = None

        padding = stride
        if patch_len>configs.pred_len:
            patch_len = configs.pred_len
        # patching and embedding
        self.patch_embedding = PatchEmbedding(
            configs.d_model, patch_len, stride, padding, configs.dropout,
            text_embedding_dim=None,
            injection_mode=None,
        )

        # Encoder
        self.injection_mode = injection_mode
        encoder_layers = [
            EncoderLayer(
                AttentionLayer(
                    FullAttention(False, configs.factor, attention_dropout=configs.dropout,
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
        
      

        # Prediction Head
        self.head_nf = configs.d_model * \
                       int((configs.seq_len - patch_len) / stride + 2)
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.head = FlattenHead(configs.enc_in, self.head_nf, configs.pred_len,
                                    head_dropout=configs.dropout)
        elif self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
            self.head = FlattenHead(configs.enc_in, self.head_nf, configs.seq_len,
                                    head_dropout=configs.dropout)
        elif self.task_name == 'classification':
            self.flatten = nn.Flatten(start_dim=-2)
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                self.head_nf * configs.enc_in, configs.num_class)

   

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, text_context=None):
        if self.injection_mode == 'unimodal':
            text_context = None
            
        enc_text = None if text_context is None or self.injection_mode in ('first-additive', 'first-concat', 'last-additive', 'last-concat') else text_context

        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev

        x_enc = x_enc.permute(0, 2, 1)
        enc_out, n_vars = self.patch_embedding(x_enc, text_context=text_context)


        enc_out, attns = self.encoder(enc_out, text_context=enc_text)
        

        
        # z: [bs x nvars x patch_num x d_model]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Decoder
        dec_out = self.head(enc_out)  # z: [bs x nvars x target_window]
        dec_out = dec_out.permute(0, 2, 1)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        
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
