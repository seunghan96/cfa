import math
import torch
import torch.nn as nn
from data_provider.data_factory import data_provider



class FourierFilter(nn.Module):
    """
    Fourier Filter: to time-variant and time-invariant term
    """
    def __init__(self, mask_spectrum):
        super(FourierFilter, self).__init__()
        self.mask_spectrum = mask_spectrum
        
    def forward(self, x):
        xf = torch.fft.rfft(x, dim=1)
        mask = torch.ones_like(xf)
        mask[:, self.mask_spectrum, :] = 0
        x_var = torch.fft.irfft(xf*mask, dim=1)
        x_inv = x - x_var
        
        return x_var, x_inv


class MLP(nn.Module):
    '''
    Multilayer perceptron to encode/decode high dimension representation of sequential data
    '''
    def __init__(self, 
                 f_in, 
                 f_out, 
                 hidden_dim=128, 
                 hidden_layers=2, 
                 dropout=0.05,
                 activation='tanh'): 
        super(MLP, self).__init__()
        self.f_in = f_in
        self.f_out = f_out
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise NotImplementedError
        
        layers = [nn.Linear(self.f_in, self.hidden_dim), 
                  self.activation, nn.Dropout(self.dropout)]
        for i in range(self.hidden_layers-2):
            layers += [nn.Linear(self.hidden_dim, self.hidden_dim),
                       self.activation, nn.Dropout(dropout)]
        
        layers += [nn.Linear(hidden_dim, f_out)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        # x:     B x S x f_in
        # y:     B x S x f_out
        y = self.layers(x)
        return y


class KPLayer(nn.Module):
    """
    A demonstration of finding one step transition of linear system by DMD iteratively
    """
    def __init__(self): 
        super(KPLayer, self).__init__()
        
        self.K = None # B E E

    def one_step_forward(self, z, return_rec=False, return_K=False):
        B, input_len, E = z.shape
        assert input_len > 1, 'snapshots number should be larger than 1'
        x, y = z[:, :-1], z[:, 1:]

        # solve linear system
        self.K = torch.linalg.lstsq(x, y).solution # B E E
        if torch.isnan(self.K).any():
            print('Encounter K with nan, replace K by identity matrix')
            self.K = torch.eye(self.K.shape[1]).to(self.K.device).unsqueeze(0).repeat(B, 1, 1)

        z_pred = torch.bmm(z[:, -1:], self.K)
        if return_rec:
            z_rec = torch.cat((z[:, :1], torch.bmm(x, self.K)), dim=1)
            return z_rec, z_pred

        return z_pred
    
    def forward(self, z, pred_len=1):
        assert pred_len >= 1, 'prediction length should not be less than 1'
        z_rec, z_pred= self.one_step_forward(z, return_rec=True)
        z_preds = [z_pred]
        for i in range(1, pred_len):
            z_pred = torch.bmm(z_pred, self.K)
            z_preds.append(z_pred)
        z_preds = torch.cat(z_preds, dim=1)
        return z_rec, z_preds


class KPLayerApprox(nn.Module):
    """
    Find koopman transition of linear system by DMD with multistep K approximation
    """
    def __init__(self): 
        super(KPLayerApprox, self).__init__()
        
        self.K = None # B E E
        self.K_step = None # B E E

    def forward(self, z, pred_len=1):
        # z:       B L E, koopman invariance space representation
        # z_rec:   B L E, reconstructed representation
        # z_pred:  B S E, forecasting representation
        B, input_len, E = z.shape
        assert input_len > 1, 'snapshots number should be larger than 1'
        x, y = z[:, :-1], z[:, 1:]

        # solve linear system
        self.K = torch.linalg.lstsq(x, y).solution # B E E

        if torch.isnan(self.K).any():
            print('Encounter K with nan, replace K by identity matrix')
            self.K = torch.eye(self.K.shape[1]).to(self.K.device).unsqueeze(0).repeat(B, 1, 1)

        z_rec = torch.cat((z[:, :1], torch.bmm(x, self.K)), dim=1) # B L E
        
        if pred_len <= input_len:
            self.K_step = torch.linalg.matrix_power(self.K, pred_len)
            if torch.isnan(self.K_step).any():
                print('Encounter multistep K with nan, replace it by identity matrix')
                self.K_step = torch.eye(self.K_step.shape[1]).to(self.K_step.device).unsqueeze(0).repeat(B, 1, 1)
            z_pred = torch.bmm(z[:, -pred_len:, :], self.K_step)
        else:
            self.K_step = torch.linalg.matrix_power(self.K, input_len)
            if torch.isnan(self.K_step).any():
                print('Encounter multistep K with nan, replace it by identity matrix')
                self.K_step = torch.eye(self.K_step.shape[1]).to(self.K_step.device).unsqueeze(0).repeat(B, 1, 1)
            temp_z_pred, all_pred = z, []
            for _ in range(math.ceil(pred_len / input_len)):
                temp_z_pred = torch.bmm(temp_z_pred, self.K_step)
                all_pred.append(temp_z_pred)
            z_pred = torch.cat(all_pred, dim=1)[:, :pred_len, :]

        return z_rec, z_pred


class TimeVarKP(nn.Module):
    """
    Koopman Predictor with DMD (analysitical solution of Koopman operator)
    Utilize local variations within individual sliding window to predict the future of time-variant term
    """
    def __init__(self,
                 enc_in=8,
                 input_len=96,
                 pred_len=96,
                 seg_len=24,
                 dynamic_dim=128,
                 encoder=None,
                 decoder=None,
                 multistep=False,
                ):
        super(TimeVarKP, self).__init__()
        self.input_len = input_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.seg_len = seg_len
        self.dynamic_dim = dynamic_dim
        self.multistep = multistep
        self.encoder, self.decoder = encoder, decoder            
        self.freq = math.ceil(self.input_len / self.seg_len)  # segment number of input
        # Ensure freq >= 2 for KPLayer to work (requires input_len > 1)
        # Instead of changing seg_len (which affects encoder/decoder), we adjust padding
        if self.freq < 2:
            self.freq = 2
        self.step = math.ceil(self.pred_len / self.seg_len)   # segment number of output
        if self.step < 1:
            self.step = 1
        self.padding_len = self.seg_len * self.freq - self.input_len
        # Approximate mulitstep K by KPLayerApprox when pred_len is large
        self.dynamics = KPLayerApprox() if self.multistep else KPLayer() 

    def forward(self, x):
        # x: B L C
        B, L, C = x.shape

        # Ensure we have enough data for freq segments of seg_len each
        needed_len = self.seg_len * self.freq
        
        # Pad or trim x to exactly needed_len
        if L < needed_len:
            # Need to pad: repeat the last timestep
            pad_size = needed_len - L
            last_timestep = x[:, -1:, :]  # [B, 1, C]
            padding = last_timestep.repeat(1, pad_size, 1)  # [B, pad_size, C]
            res = torch.cat([x, padding], dim=1)  # [B, needed_len, C]
        elif L > needed_len:
            # Need to trim: take the last needed_len timesteps
            res = x[:, -needed_len:, :]  # [B, needed_len, C]
        else:
            # Perfect length
            res = x

        # Verify res has correct length
        current_len = res.shape[1]
        if current_len != needed_len:
            # Fallback: ensure exact length
            if current_len < needed_len:
                pad_size = needed_len - current_len
                last_timestep = res[:, -1:, :]
                padding = last_timestep.repeat(1, pad_size, 1)
                res = torch.cat([res, padding], dim=1)
            else:
                res = res[:, :needed_len, :]

        # Split into freq segments, each of length seg_len
        res = res.chunk(self.freq, dim=1)     # F x [B, seg_len, C]
        res = torch.stack(res, dim=1)  # [B, freq, seg_len, C]
        res = res.reshape(B, self.freq, self.seg_len * self.enc_in)   # [B, freq, seg_len*enc_in]

        res = self.encoder(res) # [B, freq, H]
        x_rec, x_pred = self.dynamics(res, self.step) # [B, freq, H], [B, step, H]

        x_rec = self.decoder(x_rec) # [B, freq, seg_len*enc_in]
        x_rec = x_rec.reshape(B, self.freq, self.seg_len, self.enc_in)
        x_rec = x_rec.reshape(B, -1, self.enc_in)[:, :self.input_len, :]  # [B, input_len, enc_in]
        
        x_pred = self.decoder(x_pred)     # [B, step, seg_len*enc_in]
        x_pred = x_pred.reshape(B, self.step, self.seg_len, self.enc_in)
        x_pred = x_pred.reshape(B, -1, self.enc_in)[:, :self.pred_len, :] # [B, pred_len, enc_in]

        return x_rec, x_pred


class TimeInvKP(nn.Module):
    """
    Koopman Predictor with learnable Koopman operator
    Utilize lookback and forecast window snapshots to predict the future of time-invariant term
    """
    def __init__(self,
                 input_len=96,
                 pred_len=96,
                 dynamic_dim=128,
                 encoder=None,
                 decoder=None):
        super(TimeInvKP, self).__init__()
        self.dynamic_dim = dynamic_dim
        self.input_len = input_len
        self.pred_len = pred_len
        self.encoder = encoder
        self.decoder = decoder

        K_init = torch.randn(self.dynamic_dim, self.dynamic_dim)
        U, _, V = torch.svd(K_init) # stable initialization
        self.K = nn.Linear(self.dynamic_dim, self.dynamic_dim, bias=False)
        self.K.weight.data = torch.mm(U, V.t())
    
    def forward(self, x):
        # x: B L C
        res = x.transpose(1, 2) # B C L
        res = self.encoder(res) # B C H
        res = self.K(res) # B C H
        res = self.decoder(res) # B C S
        res = res.transpose(1, 2) # B S C

        return res


class Model(nn.Module):
    '''
    Paper link: https://arxiv.org/pdf/2305.18803.pdf
    '''
    def __init__(self, configs, dynamic_dim=128, hidden_dim=64, hidden_layers=2, num_blocks=3, multistep=False):
        """
        mask_spectrum: list, shared frequency spectrums
        seg_len: int, segment length of time series
        dynamic_dim: int, latent dimension of koopman embedding
        hidden_dim: int, hidden dimension of en/decoder
        hidden_layers: int, number of hidden layers of en/decoder
        num_blocks: int, number of Koopa blocks
        multistep: bool, whether to use approximation for multistep K
        alpha: float, spectrum filter ratio
        """
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.enc_in = configs.enc_in
        self.input_len = configs.seq_len
        self.pred_len = configs.pred_len

        self.seg_len = self.pred_len
        self.num_blocks = num_blocks
        self.dynamic_dim = dynamic_dim
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.multistep = multistep
        self.alpha = 0.2
        self.mask_spectrum = self._get_mask_spectrum(configs)

        self.disentanglement = FourierFilter(self.mask_spectrum)

        # shared encoder/decoder to make koopman embedding consistent
        self.time_inv_encoder = MLP(f_in=self.input_len, f_out=self.dynamic_dim, activation='relu',
                    hidden_dim=self.hidden_dim, hidden_layers=self.hidden_layers)
        self.time_inv_decoder = MLP(f_in=self.dynamic_dim, f_out=self.pred_len, activation='relu',
                           hidden_dim=self.hidden_dim, hidden_layers=self.hidden_layers)
        self.time_inv_kps = self.time_var_kps = nn.ModuleList([
                                TimeInvKP(input_len=self.input_len,
                                    pred_len=self.pred_len, 
                                    dynamic_dim=self.dynamic_dim,
                                    encoder=self.time_inv_encoder, 
                                    decoder=self.time_inv_decoder)
                                for _ in range(self.num_blocks)])

        # shared encoder/decoder to make koopman embedding consistent
        self.time_var_encoder = MLP(f_in=self.seg_len*self.enc_in, f_out=self.dynamic_dim, activation='tanh',
                           hidden_dim=self.hidden_dim, hidden_layers=self.hidden_layers)
        self.time_var_decoder = MLP(f_in=self.dynamic_dim, f_out=self.seg_len*self.enc_in, activation='tanh',
                           hidden_dim=self.hidden_dim, hidden_layers=self.hidden_layers)
        self.time_var_kps = nn.ModuleList([
                    TimeVarKP(enc_in=configs.enc_in,
                        input_len=self.input_len,
                        pred_len=self.pred_len,
                        seg_len=self.seg_len,
                        dynamic_dim=self.dynamic_dim,
                        encoder=self.time_var_encoder,
                        decoder=self.time_var_decoder,
                        multistep=self.multistep)
                    for _ in range(self.num_blocks)])
        
        # Text injection configuration
        text_embedding_dim = getattr(configs, 'text_emb', None)
        injection_mode = getattr(configs, 'text_injection_mode')
        self.injection_mode = injection_mode
        self.channels = configs.enc_in

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
            dropout_val = getattr(configs, 'dropout', 0.1)
            self.middle_gate_network = nn.Sequential(
                nn.Linear(self.channels, self.channels, bias=True),
                nn.ReLU(),
                nn.Dropout(dropout_val),
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
        
     

    def _get_mask_spectrum(self, configs):
        """
        get shared frequency spectrums
        """
        train_data, train_loader = data_provider(configs, 'train')
        amps = 0.0
        for data in train_loader:
            lookback_window = data[0]
            amps += abs(torch.fft.rfft(lookback_window, dim=1)).mean(dim=0).mean(dim=1)
        mask_spectrum = amps.topk(int(amps.shape[0]*self.alpha)).indices
        return mask_spectrum # as the spectrums of time-invariant component


    def _apply_middle_injection(self, x, text_context):
        """Apply text injection at middle stage (after Koopman mixing, before denorm)"""
        if text_context is None or self.middle_text_proj is None:
            return x
        
        # Project text context
        if len(text_context.shape) == 2:
            text_emb = self.middle_text_proj(text_context)  # [B, enc_in]
            text_emb = text_emb.unsqueeze(1).expand(-1, x.shape[1], -1)  # [B, L, enc_in]
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
            gamma = self.middle_text_proj_gamma_film(text_context if len(text_context.shape) == 2 else text_context.mean(dim=1))
            beta = self.middle_text_proj_beta_film(text_context if len(text_context.shape) == 2 else text_context.mean(dim=1))
            gamma = gamma.unsqueeze(1).expand(-1, x.shape[1], -1)
            beta = beta.unsqueeze(1).expand(-1, x.shape[1], -1)
            return gamma * x + beta

        if self.injection_mode == 'orthogonal':
            ortho_text = self.middle_text_proj_ortho(text_context if len(text_context.shape) == 2 else text_context.mean(dim=1))
            ortho_text = ortho_text.unsqueeze(1).expand(-1, x.shape[1], -1)
            proj_coeff = (ortho_text * x).sum(dim=-1, keepdim=True) / (x.pow(2).sum(dim=-1, keepdim=True) + 1e-6)
            ortho_component = ortho_text - proj_coeff * x
            return x + ortho_component

        if self.injection_mode == 'middle-additive':
            return x + text_emb
        if self.injection_mode == 'middle-concat':
            concat = torch.cat([x, text_emb], dim=-1)
            return self.middle_concat_proj(concat)

        return x

    def forecast(self, x_enc, text_context=None):
        if self.injection_mode == 'unimodal':
            text_context = None

        # First fusion: inject into x_enc [B, L, C] before processing (shape preserved)
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

        # Series Stationarization adopted from NSformer
        mean_enc = x_enc.mean(1, keepdim=True).detach()  # B x 1 x E
        x_enc = x_enc - mean_enc
        std_enc = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_enc = x_enc / std_enc

        # Koopman Forecasting
        residual, forecast = x_enc, None
        for i in range(self.num_blocks):
            time_var_input, time_inv_input = self.disentanglement(residual)
            time_inv_output = self.time_inv_kps[i](time_inv_input)
            time_var_backcast, time_var_output = self.time_var_kps[i](time_var_input)
            residual = residual - time_var_backcast
            if forecast is None:
                forecast = (time_inv_output + time_var_output)
            else:
                forecast += (time_inv_output + time_var_output)

        # Series Stationarization adopted from NSformer
        res = forecast * std_enc + mean_enc

        enc_text = None if self.injection_mode in ('first-additive', 'first-concat', 'last-additive', 'last-concat') else text_context
        res = self._apply_middle_injection(res, enc_text)

        # Last fusion: inject into output [B, pred_len, C] (shape preserved)
        if self.injection_mode in ('last-additive', 'last-concat') and self.middle_text_proj is not None:
            if len(text_context.shape) == 2:
                text_emb = self.middle_text_proj(text_context)
                text_emb = text_emb.unsqueeze(1).expand(-1, res.shape[1], -1)
            else:
                text_emb = self.middle_text_proj(text_context.mean(dim=1))
                text_emb = text_emb.unsqueeze(1).expand(-1, res.shape[1], -1)
            if self.injection_mode == 'last-additive':
                res = res + text_emb
            else:
                res = self.last_fusion_concat_proj(torch.cat([res, text_emb], dim=-1))

        return res        
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None, text_context=None):
        if self.task_name == 'long_term_forecast':
            dec_out = self.forecast(x_enc, text_context=text_context)
            return dec_out[:, -self.pred_len:, :] # [B, L, D]
