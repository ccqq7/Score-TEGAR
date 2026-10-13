from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.classifier import load_text_anchors
from models.baselines import CLIPAdapter, LinearProbe
from models.tegar import TEGAR
from utils.kg_builder import KGDefinition


class CachedLinearProbe(LinearProbe):
    """Linear probe with the cached-feature interface used by strict runs."""

    def __init__(self, *, num_labels: int, text_anchors_path: str) -> None:
        anchors, label_names = load_text_anchors(text_anchors_path, num_labels)
        super().__init__(feat_dim=int(anchors.shape[1]), num_labels=num_labels)
        self.register_buffer("text_anchors", anchors)
        self.label_names = label_names

    def forward_from_features(self, visual_features: torch.Tensor, return_aux: bool = False):
        logits = super().forward(visual_features)
        if not return_aux:
            return logits
        visual = F.normalize(visual_features.float(), dim=-1)
        clip_logits = 100.0 * (visual @ self.text_anchors.T)
        classifier_weights = F.normalize(self.head.weight, dim=-1)
        anchor_similarity = (classifier_weights * self.text_anchors).sum(dim=-1)
        base = self.text_anchors.unsqueeze(0).expand(visual.shape[0], -1, -1)
        learned = classifier_weights.unsqueeze(0).expand(visual.shape[0], -1, -1)
        return logits, {
            "visual_features": visual_features,
            "clip_logits": clip_logits,
            "delta": logits - clip_logits,
            "classifier_weights": classifier_weights,
            "classifier_anchor_similarity": anchor_similarity,
            "classifier_anchor_sim": float(anchor_similarity.detach().mean().cpu().item()),
            "initial_nodes": base,
            "final_nodes": learned,
        }

    def forward(self, visual_features: torch.Tensor, return_aux: bool = False):
        return self.forward_from_features(visual_features, return_aux=return_aux)


class CachedCLIPAdapter(CLIPAdapter):
    """CLIP-Adapter operating exclusively on frozen cached visual features."""

    def __init__(
        self,
        *,
        num_labels: int,
        text_anchors_path: str,
        reduction: int = 4,
        alpha: float = 0.2,
    ) -> None:
        anchors, label_names = load_text_anchors(text_anchors_path, num_labels)
        super().__init__(
            text_anchors=anchors,
            feat_dim=int(anchors.shape[1]),
            reduction=reduction,
            alpha=alpha,
        )
        self.label_names = label_names

    def forward_from_features(self, visual_features: torch.Tensor, return_aux: bool = False):
        features = visual_features.float()
        adapted = self.adapter(features)
        mixed = self.alpha * adapted + (1.0 - self.alpha) * features
        normalized_features = F.normalize(features, dim=-1)
        normalized_mixed = F.normalize(mixed, dim=-1)
        scale = self.logit_scale.exp().clamp(min=0.5, max=100.0)
        clip_logits = scale * (normalized_features @ self.text_anchors.T)
        logits = scale * (normalized_mixed @ self.text_anchors.T)
        if not return_aux:
            return logits
        cosine = (normalized_features * normalized_mixed).sum(dim=-1)
        return logits, {
            "visual_features": visual_features,
            "adapted_visual_features": normalized_mixed,
            "feature_cosine": cosine,
            "feature_drift": 1.0 - cosine,
            "mean_feature_drift": float((1.0 - cosine).detach().mean().cpu().item()),
            "clip_logits": clip_logits,
            "delta": logits - clip_logits,
        }

    def forward(self, visual_features: torch.Tensor, return_aux: bool = False):
        return self.forward_from_features(visual_features, return_aux=return_aux)


class FeatureTEGAR(nn.Module):
    """Controlled feature-space counterpart to Score-TEGAR.

    Both models consume the same frozen image features, text anchors, typed
    graph and seen-label loss.  This variant applies the learned residual to
    per-image label embeddings *before* cosine similarity, making embedding
    drift directly measurable without reopening the frozen visual backbone.
    """

    def __init__(
        self,
        *,
        num_labels: int,
        kg_path: str,
        text_anchors_path: str,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        exclusion_beta_init: float = 0.01,
        exclusion_mode: str | None = None,
        disabled_relations: Sequence[str] = (),
        fixed_gate: bool = False,
        gate_initial_activation: float = 0.5,
        gate_control: str | None = None,
        gate_shuffle_seed: int = 20260826,
        logit_scale_init: float = math.log(10.0),
    ) -> None:
        super().__init__()
        anchors, label_names = load_text_anchors(text_anchors_path, num_labels)
        self.register_buffer("text_anchors", anchors)
        self.label_names = label_names
        self.feat_dim = int(anchors.shape[1])
        self.anchor_encoder = nn.Sequential(
            nn.Linear(self.feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.graph = TEGAR(
            kg_path=kg_path,
            hidden_dim=hidden_dim,
            visual_dim=self.feat_dim,
            num_layers=num_layers,
            dropout=dropout,
            exclusion_beta_init=exclusion_beta_init,
            apply_layernorm=True,
            apply_activation=True,
            use_pairnorm=False,
            fixed_gate=fixed_gate,
            exclusion_mode=exclusion_mode,
            disabled_relations=disabled_relations,
            gate_initial_activation=gate_initial_activation,
            gate_control=gate_control,
            gate_shuffle_seed=gate_shuffle_seed,
            # A feature-space control has no score-domain residual to consume
            # strict exclusion evidence. Keep exclusion as an explicit latent
            # feature update; its effect is not claimed to be score-monotonic.
            legacy_latent_exclusion=True,
        )
        self.embedding_delta_head = nn.Linear(hidden_dim, self.feat_dim)
        nn.init.xavier_uniform_(self.embedding_delta_head.weight, gain=0.01)
        nn.init.zeros_(self.embedding_delta_head.bias)
        self.residual_scale_raw = nn.Parameter(torch.tensor(-2.0, dtype=torch.float32))
        self.logit_scale = nn.Parameter(torch.tensor(float(logit_scale_init), dtype=torch.float32))

    def forward_from_features(self, visual_features: torch.Tensor, return_aux: bool = False):
        visual = F.normalize(visual_features.float(), dim=-1)
        batch_size = int(visual.shape[0])
        initial = self.anchor_encoder(self.text_anchors).unsqueeze(0).expand(batch_size, -1, -1)
        if return_aux:
            refined, graph_aux = self.graph(initial, visual, return_aux=True)
        else:
            refined = self.graph(initial, visual, return_aux=False)
            graph_aux = {}
        residual_scale = torch.sigmoid(self.residual_scale_raw)
        embedding_delta = residual_scale * self.embedding_delta_head(refined)
        base = self.text_anchors.unsqueeze(0).expand(batch_size, -1, -1)
        adapted_anchors = F.normalize(base + embedding_delta, dim=-1)
        temperature = self.logit_scale.exp().clamp(min=0.5, max=100.0)
        logits = temperature * torch.einsum("bd,bnd->bn", visual, adapted_anchors)
        if not return_aux:
            return logits
        cosine = (adapted_anchors * base).sum(dim=-1)
        clip_logits = temperature * torch.einsum("bd,bnd->bn", visual, base)
        aux = dict(graph_aux)
        aux.update(
            {
                "visual_features": visual_features,
                "base_anchors": base,
                "adapted_anchors": adapted_anchors,
                "embedding_delta": embedding_delta,
                "embedding_cosine": cosine,
                "embedding_drift": 1.0 - cosine,
                "mean_embedding_drift": float((1.0 - cosine).detach().mean().cpu().item()),
                "classifier_anchor_sim": float(cosine.detach().mean().cpu().item()),
                "clip_logits": clip_logits,
                "delta": logits - clip_logits,
                "initial_nodes": base,
                "final_nodes": adapted_anchors,
                "residual_scale": float(residual_scale.detach().cpu().item()),
            }
        )
        return logits, aux

    def forward(self, visual_features: torch.Tensor, return_aux: bool = False):
        return self.forward_from_features(visual_features, return_aux=return_aux)


class GCNZAnchor(nn.Module):
    """Adapted graph-ZSL baseline that generates classifier weights from anchors."""

    def __init__(
        self,
        *,
        num_labels: int,
        kg_path: str,
        text_anchors_path: str,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        logit_scale_init: float = math.log(10.0),
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be positive")
        anchors, label_names = load_text_anchors(text_anchors_path, num_labels)
        self.register_buffer("text_anchors", anchors)
        self.label_names = label_names
        self.feat_dim = int(anchors.shape[1])
        kg = KGDefinition(kg_path)
        adjacency = kg.get_homogeneous_adjacency(sparse=False).float()
        adjacency = adjacency + torch.eye(adjacency.shape[0], dtype=adjacency.dtype)
        degree = adjacency.sum(dim=1).clamp_min(1.0)
        inv_sqrt = degree.rsqrt()
        normalized = inv_sqrt[:, None] * adjacency * inv_sqrt[None, :]
        self.register_buffer("normalized_adjacency", normalized)

        dimensions = [self.feat_dim]
        if num_layers == 1:
            dimensions.append(self.feat_dim)
        else:
            dimensions.extend([hidden_dim] * (num_layers - 1))
            dimensions.append(self.feat_dim)
        self.layers = nn.ModuleList(
            [nn.Linear(dimensions[idx], dimensions[idx + 1], bias=False) for idx in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)
        self.logit_scale = nn.Parameter(torch.tensor(float(logit_scale_init), dtype=torch.float32))
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight, gain=0.1)

    def classifier_weights(self) -> torch.Tensor:
        hidden = self.text_anchors
        for layer_index, layer in enumerate(self.layers):
            hidden = self.normalized_adjacency @ hidden
            hidden = layer(hidden)
            if layer_index < len(self.layers) - 1:
                hidden = self.dropout(F.relu(hidden))
        # A residual anchor path matches common graph-ZSL practice and prevents
        # a randomly initialized GCN from discarding the frozen semantic prior.
        return F.normalize(self.text_anchors + hidden, dim=-1)

    def forward_from_features(self, visual_features: torch.Tensor, return_aux: bool = False):
        visual = F.normalize(visual_features.float(), dim=-1)
        classifier_weights = self.classifier_weights()
        temperature = self.logit_scale.exp().clamp(min=0.5, max=100.0)
        logits = temperature * (visual @ classifier_weights.T)
        if not return_aux:
            return logits
        anchor_similarity = (classifier_weights * self.text_anchors).sum(dim=-1)
        base = self.text_anchors.unsqueeze(0).expand(visual.shape[0], -1, -1)
        generated = classifier_weights.unsqueeze(0).expand(visual.shape[0], -1, -1)
        clip_logits = temperature * (visual @ self.text_anchors.T)
        return logits, {
            "visual_features": visual_features,
            "classifier_weights": classifier_weights,
            "classifier_anchor_similarity": anchor_similarity,
            "mean_classifier_anchor_similarity": float(anchor_similarity.detach().mean().cpu().item()),
            "classifier_anchor_sim": float(anchor_similarity.detach().mean().cpu().item()),
            "clip_logits": clip_logits,
            "delta": logits - clip_logits,
            "initial_nodes": base,
            "final_nodes": generated,
        }

    def forward(self, visual_features: torch.Tensor, return_aux: bool = False):
        return self.forward_from_features(visual_features, return_aux=return_aux)


def trainable_parameter_count(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))
