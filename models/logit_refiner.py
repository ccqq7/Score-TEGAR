from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.classifier import load_text_anchors
from models.clip_backbone import CLIPVisualBackbone
from models.tegar import HomogeneousGAT, TEGAR


class ProbabilitySpaceRefiner(nn.Module):
    def __init__(
        self,
        num_labels: int,
        kg_path: str,
        text_anchors_path: str,
        clip_model_name: str = "ViT-B-32",
        clip_pretrained_path: str | None = None,
        clip_default_pretrained: str | None = None,
        graph_type: str = "tegar",
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        exclusion_beta_init: float = 0.01,
        use_cached_features: bool = False,
        fixed_gate: bool = False,
        learnable_temperature: bool = True,
        min_uncertainty_gate: float = 0.10,
        scene_expert_count: int = 4,
        use_relation_consensus: bool = True,
        negative_exclusion: bool = True,
    ) -> None:
        super().__init__()
        if graph_type not in {"tegar", "homo_gat", "none"}:
            raise ValueError(f"Unsupported graph_type: {graph_type}")

        self.graph_type = graph_type
        self.use_cached_features = bool(use_cached_features)
        self.num_labels = int(num_labels)
        self.fixed_gate = bool(fixed_gate)
        self.learnable_temperature = bool(learnable_temperature)
        self.use_relation_consensus = bool(use_relation_consensus)
        self.negative_exclusion = bool(negative_exclusion)
        self.backbone = None
        if not self.use_cached_features:
            self.backbone = CLIPVisualBackbone(
                model_name=clip_model_name,
                pretrained_path=clip_pretrained_path,
                default_pretrained=clip_default_pretrained,
            )

        anchors, label_names = load_text_anchors(text_anchors_path, num_labels)
        self.register_buffer("text_anchors", anchors)
        self.label_names = label_names
        self.feat_dim = anchors.shape[1]

        self.node_encoder = nn.Sequential(
            nn.Linear(2 + self.feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # Focus graph updates on ambiguous labels, where AP ranking can actually change.
        self.uncertainty_scale = 6.0
        self.min_uncertainty_gate = float(min_uncertainty_gate)

        if graph_type == "tegar":
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
                negative_exclusion=negative_exclusion,
            )
        elif graph_type == "homo_gat":
            self.graph = HomogeneousGAT(
                kg_path=kg_path,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
                apply_layernorm=True,
                apply_activation=True,
                use_pairnorm=False,
            )
        else:
            self.graph = None

        self.delta_head = nn.Linear(hidden_dim, 1)
        nn.init.xavier_uniform_(self.delta_head.weight, gain=0.01)
        nn.init.zeros_(self.delta_head.bias)
        self.relation_names = ("often_cooccur", "statistical_exclusion", "hierarchical")
        if graph_type == "tegar":
            self.scene_expert_count = max(1, int(scene_expert_count))
            self.scene_router = nn.Sequential(
                nn.Linear(self.feat_dim, hidden_dim),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(hidden_dim, self.scene_expert_count),
            )
            nn.init.xavier_uniform_(self.scene_router[0].weight, gain=0.1)
            nn.init.zeros_(self.scene_router[0].bias)
            nn.init.xavier_uniform_(self.scene_router[2].weight, gain=0.1)
            nn.init.zeros_(self.scene_router[2].bias)
            self.relation_delta_heads = nn.ModuleDict(
                {
                    name: nn.ModuleList(
                        [nn.Linear(hidden_dim, 1) for _ in range(self.scene_expert_count)]
                    )
                    for name in self.relation_names
                }
            )
            for head_list in self.relation_delta_heads.values():
                for head in head_list:
                    nn.init.xavier_uniform_(head.weight, gain=0.01)
                    nn.init.zeros_(head.bias)
            self.relation_fusion_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        else:
            self.relation_delta_heads = None
            self.scene_expert_count = 0
            self.scene_router = None

        if self.learnable_temperature:
            self.logit_temperature = nn.Parameter(torch.tensor(3.0, dtype=torch.float32))
        else:
            self.register_buffer("logit_temperature", torch.tensor(3.0, dtype=torch.float32))

    def compute_clip_scores(self, visual_cls: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            visual_norm = F.normalize(visual_cls.float(), dim=-1)
            text_norm = F.normalize(self.text_anchors, dim=-1)
            scores = visual_norm @ text_norm.T
        return scores

    def compute_refined_scores(
        self,
        clip_scores: torch.Tensor,
        visual_cls: torch.Tensor,
        return_aux: bool = False,
    ):
        batch_size = clip_scores.size(0)
        score_input = clip_scores.unsqueeze(-1)
        clip_probs = torch.sigmoid(self.uncertainty_scale * clip_scores)
        uncertainty = 4.0 * clip_probs * (1.0 - clip_probs)
        uncertainty_input = uncertainty.unsqueeze(-1)
        text_expanded = self.text_anchors.unsqueeze(0).expand(batch_size, -1, -1)
        node_input = torch.cat([score_input, uncertainty_input, text_expanded], dim=-1)
        node_features = self.node_encoder(node_input)

        graph_aux: dict = {}
        if self.graph_type == "tegar":
            refined, graph_aux = self.graph(node_features, visual_cls, return_aux=True)
        elif self.graph_type == "homo_gat":
            if return_aux:
                refined, graph_aux = self.graph(node_features, return_aux=True)
            else:
                refined = self.graph(node_features, return_aux=False)
        else:
            refined = node_features

        raw_delta = self.delta_head(refined).squeeze(-1)
        relation_delta = None
        if self.graph_type == "tegar" and self.relation_delta_heads is not None:
            relation_messages = graph_aux.get("relation_messages")
            gate_values = graph_aux.get("gate_values")
            if relation_messages is not None and gate_values is not None:
                if gate_values.ndim == 3:
                    relation_weights = gate_values[:, -1, :]
                else:
                    relation_weights = gate_values
                relation_weights = relation_weights / relation_weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
                scene_weights = torch.softmax(self.scene_router(visual_cls), dim=-1)
                relation_delta_parts = []
                for relation_name in self.relation_names:
                    relation_state = relation_messages.get(relation_name)
                    if relation_state is None:
                        continue
                    expert_outputs = []
                    for expert_head in self.relation_delta_heads[relation_name]:
                        expert_delta = expert_head(relation_state).squeeze(-1)
                        expert_outputs.append(expert_delta.unsqueeze(-1))
                    expert_tensor = torch.cat(expert_outputs, dim=-1)
                    delta_part = (expert_tensor * scene_weights.unsqueeze(1)).sum(dim=-1)
                    relation_delta_parts.append(delta_part.unsqueeze(-1))
                if relation_delta_parts:
                    relation_delta_tensor = torch.cat(relation_delta_parts, dim=-1)
                    weighted_relation_delta = relation_delta_tensor * relation_weights.unsqueeze(1)
                    relation_delta = weighted_relation_delta.sum(dim=-1)
                    # Typed-edge updates help when relation branches agree on the
                    # correction direction; conflicting branches are likely noise.
                    if self.use_relation_consensus:
                        relation_consensus = relation_delta.abs() / weighted_relation_delta.abs().sum(dim=-1).clamp_min(1e-6)
                    else:
                        relation_consensus = torch.ones_like(relation_delta)
                    raw_delta = raw_delta + self.relation_fusion_scale * relation_consensus * relation_delta
        uncertainty_gate = self.min_uncertainty_gate + (1.0 - self.min_uncertainty_gate) * uncertainty
        score_delta = uncertainty_gate * raw_delta
        refined_scores = clip_scores + score_delta

        if not return_aux:
            return refined_scores, score_delta

        aux = dict(graph_aux) if isinstance(graph_aux, dict) else {}
        aux.update(
            {
                "raw_delta": raw_delta,
                "score_delta": score_delta,
                "mean_abs_delta": float(score_delta.detach().abs().mean().cpu().item()),
                "mean_uncertainty": float(uncertainty.detach().mean().cpu().item()),
                "clip_scores": clip_scores,
                "refined_scores": refined_scores,
                "uncertainty": uncertainty,
                "uncertainty_gate": uncertainty_gate,
                "node_features": node_features,
                "refined": refined,
                "initial_nodes": node_features,
                "final_nodes": refined,
            }
        )
        if relation_delta is not None:
            aux["relation_delta"] = relation_delta
            aux["relation_fusion_scale"] = float(self.relation_fusion_scale.detach().cpu().item())
            aux["scene_weights"] = scene_weights
            aux["relation_consensus"] = relation_consensus
        return refined_scores, score_delta, aux

    def forward_from_features(self, visual_cls: torch.Tensor, return_aux: bool = False):
        clip_scores = self.compute_clip_scores(visual_cls)

        if return_aux:
            refined_scores, score_delta, aux = self.compute_refined_scores(clip_scores, visual_cls, return_aux=True)
        else:
            refined_scores, score_delta = self.compute_refined_scores(clip_scores, visual_cls, return_aux=False)
            aux = {}

        temperature = self.logit_temperature.abs().clamp(min=0.5, max=20.0)
        logits = temperature * refined_scores

        if not return_aux:
            return logits

        clip_logits = temperature * clip_scores

        aux.update(
            {
                "visual_features": visual_cls,
                "visual_cls": visual_cls,
                "clip_logits": clip_logits,
                "delta": score_delta,
                "logit_temperature": float(temperature.detach().cpu().item()),
                "used_spatial_attn": False,
                "used_patches": False,
            }
        )
        return logits, aux

    def forward(self, images: torch.Tensor, return_aux: bool = False):
        if self.backbone is None:
            raise RuntimeError("Use forward_from_features() for cached-feature mode.")
        visual_cls = self.backbone(images, return_patches=False)
        return self.forward_from_features(visual_cls, return_aux=return_aux)


LogitSpaceRefiner = ProbabilitySpaceRefiner
