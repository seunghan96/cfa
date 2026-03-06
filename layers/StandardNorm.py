import torch
import torch.nn as nn


class Normalize(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=False, subtract_last=False, non_norm=False,
                 text_embedding_dim=None, injection_mode='last'):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(Normalize, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        self.non_norm = non_norm
        if self.affine:
            self._init_params()
        
        # Text injection for middle-inst normalization (modify affine_weight and affine_bias)
        self.has_text = text_embedding_dim is not None
        self.injection_mode = injection_mode
        if self.has_text and self.affine and (injection_mode == 'middle-inst' or injection_mode == 'middle-inst-layer' or
                                             injection_mode == 'first-middle-inst' or injection_mode == 'middle-inst-last' or
                                             injection_mode == 'first-middle-inst-last'):
            self.text_proj_alpha_inst = nn.Linear(text_embedding_dim, num_features, bias=True)
            self.text_proj_beta_inst = nn.Linear(text_embedding_dim, num_features, bias=True)
            nn.init.ones_(self.text_proj_alpha_inst.bias)
            nn.init.zeros_(self.text_proj_beta_inst.bias)
        else:
            self.text_proj_alpha_inst = None
            self.text_proj_beta_inst = None

    def forward(self, x, mode: str, text_context=None):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x, text_context=text_context)
        elif mode == 'denorm':
            x = self._denormalize(x, text_context=text_context)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x, text_context=None):
        if self.non_norm:
            return x
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            # Apply affine transformation with optional text-based modifiers (middle-inst)
            if self.has_text and text_context is not None and self.text_proj_alpha_inst is not None and \
               (self.injection_mode == 'middle-inst' or self.injection_mode == 'middle-inst-layer' or
                self.injection_mode == 'first-middle-inst' or self.injection_mode == 'middle-inst-last' or
                self.injection_mode == 'first-middle-inst-last'):
                # Get alpha and beta modifiers from text
                if len(text_context.shape) == 2:  # [B, text_embedding_dim]
                    alpha_modifier = self.text_proj_alpha_inst(text_context)  # [B, num_features]
                    beta_modifier = self.text_proj_beta_inst(text_context)
                elif len(text_context.shape) == 3:  # [B, text_seq_len, text_embedding_dim]
                    text_emb_seq = text_context.mean(dim=1)
                    alpha_modifier = self.text_proj_alpha_inst(text_emb_seq)
                    beta_modifier = self.text_proj_beta_inst(text_emb_seq)
                else:
                    alpha_modifier = None
                    beta_modifier = None
                
                if alpha_modifier is not None:
                    # Apply: (x * affine_weight + affine_bias) * alpha_modifier + beta_modifier
                    # = x * (affine_weight * alpha_modifier) + (affine_bias * alpha_modifier + beta_modifier)
                    adjusted_weight = self.affine_weight * alpha_modifier.unsqueeze(1)  # [num_features] * [B, num_features] -> [B, num_features] (broadcast)
                    adjusted_bias = self.affine_bias * alpha_modifier.unsqueeze(1) + beta_modifier.unsqueeze(1)  # [B, num_features]
                    x = x * adjusted_weight + adjusted_bias
                else:
                    x = x * self.affine_weight + self.affine_bias
            else:
                x = x * self.affine_weight + self.affine_bias
        return x

    def _denormalize(self, x, text_context=None):
        if self.non_norm:
            return x
        if self.affine:
            # Reverse affine transformation with optional text-based modifiers (middle-inst)
            if self.has_text and text_context is not None and self.text_proj_alpha_inst is not None and \
               (self.injection_mode == 'middle-inst' or self.injection_mode == 'middle-inst-layer' or
                self.injection_mode == 'first-middle-inst' or self.injection_mode == 'middle-inst-last' or
                self.injection_mode == 'first-middle-inst-last'):
                # Get the same modifiers used in encoding
                if len(text_context.shape) == 2:  # [B, text_embedding_dim]
                    alpha_modifier = self.text_proj_alpha_inst(text_context)  # [B, num_features]
                    beta_modifier = self.text_proj_beta_inst(text_context)
                elif len(text_context.shape) == 3:  # [B, text_seq_len, text_embedding_dim]
                    text_emb_seq = text_context.mean(dim=1)
                    alpha_modifier = self.text_proj_alpha_inst(text_emb_seq)
                    beta_modifier = self.text_proj_beta_inst(text_emb_seq)
                else:
                    alpha_modifier = None
                    beta_modifier = None
                
                if alpha_modifier is not None:
                    adjusted_weight = self.affine_weight * alpha_modifier.unsqueeze(1)
                    adjusted_bias = self.affine_bias * alpha_modifier.unsqueeze(1) + beta_modifier.unsqueeze(1)
                    x = x - adjusted_bias
                    x = x / (adjusted_weight + self.eps * self.eps)
                else:
                    x = x - self.affine_bias
                    x = x / (self.affine_weight + self.eps * self.eps)
            else:
                x = x - self.affine_bias
                x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x
