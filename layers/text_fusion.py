"""
Novel Text Fusion Methods for Multimodal Time Series Forecasting

This module implements 4 novel fusion methods that preserve time series information
while selectively incorporating text information:

1. Gated Additive: Text-conditioned gate to control text injection
2. Residual Adapter: Low-rank text perturbation as residual
3. FiLM (Feature-wise Linear Modulation): Scale and bias modulation
4. Orthogonal Decomposition: Inject only orthogonal component of text
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedAdditiveFusion(nn.Module):
    """
    Gated Additive Fusion (Text-conditioned Gate)
    
    Core idea: Don't add text directly, but first determine "how much text is needed
    for this time embedding" based on the time embedding itself.
    
    Form: gate ∈ [0,1]^d computed from time embedding
    Output: h_t + gate(e_text) ⊙ e_text
    """
    def __init__(self, feature_dim, text_embedding_dim, dropout=0.1):
        super(GatedAdditiveFusion, self).__init__()
        self.text_projection = nn.Linear(text_embedding_dim, feature_dim, bias=False)
        
        # Gate network: takes time embedding and outputs gate values
        self.gate_network = nn.Sequential(
            nn.Linear(feature_dim, feature_dim, bias=True),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, feature_dim, bias=True),
            nn.Sigmoid()  # Gate values in [0, 1]
        )
    
    def forward(self, ts_embedding, text_context):
        """
        Args:
            ts_embedding: [B, L, D] or [B, D] time series embedding
            text_context: [B, text_embedding_dim] or [B, text_seq_len, text_embedding_dim]
        
        Returns:
            fused_embedding: [B, L, D] or [B, D] same shape as ts_embedding
        """
        # Project text to feature dimension
        if len(text_context.shape) == 2:  # [B, text_embedding_dim]
            text_emb = self.text_projection(text_context)  # [B, D]
        elif len(text_context.shape) == 3:  # [B, text_seq_len, text_embedding_dim]
            text_emb_seq = self.text_projection(text_context)  # [B, text_seq_len, D]
            text_emb = text_emb_seq.mean(dim=1)  # [B, D]
        else:
            raise ValueError(f"Unexpected text_context shape: {text_context.shape}")
        
        # Expand text_emb to match ts_embedding shape
        if len(ts_embedding.shape) == 3:  # [B, L, D]
            text_emb = text_emb.unsqueeze(1).expand(-1, ts_embedding.shape[1], -1)  # [B, L, D]
        
        # Compute gate from time series embedding
        gate = self.gate_network(ts_embedding)  # [B, L, D] or [B, D]
        
        # Gated addition: only add text where gate is high
        return ts_embedding + gate * text_emb


class ResidualAdapterFusion(nn.Module):
    """
    Residual Adapter-style Text Injection
    (Text as a low-rank perturbation)
    
    Core idea: Text works as a low-dimensional residual on TS embedding,
    not redefining the representation but providing "directional correction hints".
    
    Structure:
    - TS backbone remains unchanged
    - text → small bottleneck → residual added to TS
    - Only adapter is trainable
    """
    def __init__(self, feature_dim, text_embedding_dim, bottleneck_ratio=8, dropout=0.1):
        super(ResidualAdapterFusion, self).__init__()
        bottleneck_dim = max(1, feature_dim // bottleneck_ratio)
        
        # Down-projection: text → bottleneck
        self.adapter_down = nn.Linear(text_embedding_dim, bottleneck_dim, bias=False)
        self.adapter_norm = nn.LayerNorm(bottleneck_dim)
        self.adapter_activation = nn.ReLU()
        self.adapter_dropout = nn.Dropout(dropout)
        
        # Up-projection: bottleneck → feature_dim
        self.adapter_up = nn.Linear(bottleneck_dim, feature_dim, bias=False)
        
        # Initialize adapter_up to near-zero for stable training
        nn.init.zeros_(self.adapter_up.weight)
    
    def forward(self, ts_embedding, text_context):
        """
        Args:
            ts_embedding: [B, L, D] or [B, D] time series embedding
            text_context: [B, text_embedding_dim] or [B, text_seq_len, text_embedding_dim]
        
        Returns:
            fused_embedding: [B, L, D] or [B, D] same shape as ts_embedding
        """
        # Pool text if needed
        if len(text_context.shape) == 3:  # [B, text_seq_len, text_embedding_dim]
            pooled_text = text_context.mean(dim=1)  # [B, text_embedding_dim]
        else:
            pooled_text = text_context  # [B, text_embedding_dim]
        
        # Adapter forward pass
        adapter_hidden = self.adapter_down(pooled_text)  # [B, bottleneck_dim]
        adapter_hidden = self.adapter_norm(adapter_hidden)
        adapter_hidden = self.adapter_activation(adapter_hidden)
        adapter_hidden = self.adapter_dropout(adapter_hidden)
        adapter_out = self.adapter_up(adapter_hidden)  # [B, D]
        
        # Expand to match ts_embedding shape
        if len(ts_embedding.shape) == 3:  # [B, L, D]
            adapter_out = adapter_out.unsqueeze(1).expand(-1, ts_embedding.shape[1], -1)  # [B, L, D]
        
        # Add as residual
        return ts_embedding + adapter_out


class FiLMFusion(nn.Module):
    """
    Text-conditioned Scaling / Modulation (FiLM-style)
    
    Core idea: Text doesn't add values, but only adjusts the scale and bias
    of TS features. This preserves pattern shapes while changing semantics.
    
    Form: h̃_t = γ(e_text) ⊙ h_t + β(e_text)
    or: h̃_t = γ(e_text) ⊙ h_t (bias-free version)
    """
    def __init__(self, feature_dim, text_embedding_dim, use_bias=True):
        super(FiLMFusion, self).__init__()
        self.use_bias = use_bias
        
        # Gamma (scale) projection
        self.gamma_proj = nn.Linear(text_embedding_dim, feature_dim, bias=True)
        # Initialize bias to 1.0 so initially no scaling
        nn.init.ones_(self.gamma_proj.bias)
        
        if use_bias:
            # Beta (bias/shift) projection
            self.beta_proj = nn.Linear(text_embedding_dim, feature_dim, bias=True)
            # Initialize bias to 0.0 so initially no shift
            nn.init.zeros_(self.beta_proj.bias)
        else:
            self.beta_proj = None
    
    def forward(self, ts_embedding, text_context):
        """
        Args:
            ts_embedding: [B, L, D] or [B, D] time series embedding
            text_context: [B, text_embedding_dim] or [B, text_seq_len, text_embedding_dim]
        
        Returns:
            fused_embedding: [B, L, D] or [B, D] same shape as ts_embedding
        """
        # Pool text if needed
        if len(text_context.shape) == 3:  # [B, text_seq_len, text_embedding_dim]
            pooled_text = text_context.mean(dim=1)  # [B, text_embedding_dim]
        else:
            pooled_text = text_context  # [B, text_embedding_dim]
        
        # Compute gamma and beta
        gamma = self.gamma_proj(pooled_text)  # [B, D]
        
        # Expand to match ts_embedding shape
        if len(ts_embedding.shape) == 3:  # [B, L, D]
            gamma = gamma.unsqueeze(1).expand(-1, ts_embedding.shape[1], -1)  # [B, L, D]
        
        # Apply modulation: γ ⊙ h_t
        modulated = gamma * ts_embedding
        
        if self.use_bias:
            beta = self.beta_proj(pooled_text)  # [B, D]
            if len(ts_embedding.shape) == 3:  # [B, L, D]
                beta = beta.unsqueeze(1).expand(-1, ts_embedding.shape[1], -1)  # [B, L, D]
            return modulated + beta
        else:
            return modulated


class OrthogonalFusion(nn.Module):
    """
    Orthogonal Decomposition Injection
    
    Core idea: Explicitly ensure TS information is preserved by injecting text
    only in the direction orthogonal to the TS embedding space.
    
    Form:
    - Project text onto TS space
    - Remove overlapping component
    - Add only orthogonal component
    
    e'_text = e_text - Proj_{h_t}(e_text)
    h̃_t = h_t + e'_text
    """
    def __init__(self, feature_dim, text_embedding_dim):
        super(OrthogonalFusion, self).__init__()
        self.text_projection = nn.Linear(text_embedding_dim, feature_dim, bias=False)
    
    def forward(self, ts_embedding, text_context):
        """
        Args:
            ts_embedding: [B, L, D] or [B, D] time series embedding
            text_context: [B, text_embedding_dim] or [B, text_seq_len, text_embedding_dim]
        
        Returns:
            fused_embedding: [B, L, D] or [B, D] same shape as ts_embedding
        """
        # Project text to feature dimension
        if len(text_context.shape) == 2:  # [B, text_embedding_dim]
            text_emb = self.text_projection(text_context)  # [B, D]
        elif len(text_context.shape) == 3:  # [B, text_seq_len, text_embedding_dim]
            text_emb_seq = self.text_projection(text_context)  # [B, text_seq_len, D]
            text_emb = text_emb_seq.mean(dim=1)  # [B, D]
        else:
            raise ValueError(f"Unexpected text_context shape: {text_context.shape}")
        
        # Handle different shapes
        if len(ts_embedding.shape) == 3:  # [B, L, D]
            # For each time step, compute orthogonal component
            # text_emb: [B, D] -> expand to [B, L, D]
            text_emb_expanded = text_emb.unsqueeze(1).expand(-1, ts_embedding.shape[1], -1)  # [B, L, D]
            
            # Compute projection coefficient: <text_emb, ts_emb> / ||ts_emb||^2
            # For each position [B, L], compute dot product along D dimension
            dot_product = (text_emb_expanded * ts_embedding).sum(dim=-1, keepdim=True)  # [B, L, 1]
            ts_norm_sq = ts_embedding.pow(2).sum(dim=-1, keepdim=True) + 1e-8  # [B, L, 1]
            proj_coeff = dot_product / ts_norm_sq  # [B, L, 1]
            
            # Remove projection component: text_ortho = text_emb - proj_coeff * ts_emb
            text_ortho = text_emb_expanded - proj_coeff * ts_embedding  # [B, L, D]
            print('HAHAHAAHAHA')
            print('HAHAHAAHAHA')
            print('HAHAHAAHAHA')
            print('HAHAHAAHAHA')
            
            return ts_embedding + text_ortho
        else:  # [B, D]
            # Compute projection coefficient
            dot_product = (text_emb * ts_embedding).sum(dim=-1, keepdim=True)  # [B, 1]
            ts_norm_sq = ts_embedding.pow(2).sum(dim=-1, keepdim=True) + 1e-8  # [B, 1]
            proj_coeff = dot_product / ts_norm_sq  # [B, 1]
            
            # Remove projection component
            text_ortho = text_emb - proj_coeff * ts_embedding  # [B, D]
            
            return ts_embedding + text_ortho


def apply_text_fusion(ts_embedding, text_context, fusion_type, fusion_module):
    """
    Unified interface for applying text fusion.

    Args:
        ts_embedding: Time series embedding [B, L, D] or [B, D]
        text_context: Text context [B, text_embedding_dim] or [B, text_seq_len, text_embedding_dim]
        fusion_type: One of 'gating', 'cfa', 'film', 'orthogonal'
        fusion_module: The fusion module instance

    Returns:
        Fused embedding with same shape as ts_embedding
    """
    if text_context is None:
        return ts_embedding

    if fusion_type == 'gating' and fusion_module is not None:
        return fusion_module(ts_embedding, text_context)
    elif fusion_type == 'cfa' and fusion_module is not None:
        return fusion_module(ts_embedding, text_context)
    elif fusion_type == 'film' and fusion_module is not None:
        return fusion_module(ts_embedding, text_context)
    elif fusion_type == 'orthogonal' and fusion_module is not None:
        return fusion_module(ts_embedding, text_context)

    return ts_embedding
