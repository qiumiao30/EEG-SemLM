import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveAppearanceGuidance(nn.Module):
    def __init__(self,
                 base_guidance_scale: float = 1.5,
                 hidden_size: int = 768,
                 min_scale: float = 0.1,
                 max_scale: float = 3.0):
        super().__init__()
        self.base_guidance_scale = base_guidance_scale
        self.min_scale = min_scale
        self.max_scale = max_scale
        
        # Similarity evaluation network
        self.similarity_net = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        
        # Feature importance weights
        self.feature_importance = nn.Parameter(torch.ones(hidden_size))
        
    def compute_adaptive_scale(self, cross_features: torch.Tensor, 
                               self_features: torch.Tensor) -> torch.Tensor:
        """Compute adaptive guidance scale."""
        # 1. Compute weighted cosine similarity
        weighted_cross = cross_features * self.feature_importance
        weighted_self = self_features * self.feature_importance
        cosine_sim = F.cosine_similarity(weighted_cross, weighted_self, dim=-1)
        
        # 2. Calculate feature difference
        # feature_diff = torch.norm(cross_features - self_features, dim=-1)
        
        # 3. Combine features for evaluation
        combined_features = torch.cat([cross_features, self_features], dim=-1)
        
        # 4. Predict adaptive factor through network
        adaptation_factor = self.similarity_net(combined_features).squeeze(-1)
        
        # 5. Compute final scale based on base guidance strength
        adaptive_scale = self.base_guidance_scale * (1.0 - cosine_sim * adaptation_factor)
        
        # 6. Clamp within the effective range
        adaptive_scale = torch.clamp(adaptive_scale, min=self.min_scale, max=self.max_scale)
        
        return adaptive_scale

    def apply_guidance(self, 
                       cross_features: torch.Tensor, 
                       self_features: torch.Tensor,
                       confidence_threshold: float = 0.8) -> torch.Tensor:
        """Apply adaptive guidance."""
        # 1. Compute adaptive guidance scale
        adaptive_scale = self.compute_adaptive_scale(cross_features, self_features)
        
        # 2. Calculate feature confidence
        feature_confidence = torch.sigmoid((cross_features * self_features).sum(dim=-1))
        
        # 3. Dynamic guidance based on confidence
        guidance_mask = (feature_confidence > confidence_threshold)
        
        # 4. Apply adaptive guidance
        guided_features = (
            self_features + 
            guidance_mask.unsqueeze(-1) * 
            adaptive_scale.unsqueeze(-1) * 
            (cross_features - self_features)
        )
        
        return guided_features   