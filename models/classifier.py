from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbone import ResNetBackbone
from models.clip_backbone import CLIPVisualBackbone
from models.tegar import HomogeneousGAT, TEGAR
from utils.kg_builder import KGDefinition


class PrototypeInitializer(nn.Module):
    def __init__(self, num_labels: int, visual_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.label_embeddings = nn.Parameter(torch.randn(num_labels, hidden_dim) * 0.02)
        self.visual_key = nn.Linear(visual_dim, hidden_dim)
        self.visual_value = nn.Linear(visual_dim, hidden_dim)
        self.visual_bias = nn.Linear(visual_dim, hidden_dim)
        self.scale = 1.0 / math.sqrt(hidden_dim)

    def forward(self, visual_features: torch.Tensor) -> torch.Tensor:
        batch_size = visual_features.size(0)
        label_tokens = self.label_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        key = self.visual_key(visual_features).unsqueeze(1)
        value = self.visual_value(visual_features).unsqueeze(1)
        bias = self.visual_bias(visual_features).unsqueeze(1)
        attention = torch.sigmoid((label_tokens * key).sum(dim=-1, keepdim=True) * self.scale)
        return label_tokens + attention * value + bias


class BaselineClassifier(nn.Module):
    def __init__(self, num_labels: int, backbone: str = "resnet50", pretrained: bool = True) -> None:
        super().__init__()
        if backbone != "resnet50":
            raise ValueError(f"Unsupported backbone: {backbone}")
        self.backbone = ResNetBackbone(pretrained=pretrained)
        self.head = nn.Linear(self.backbone.feat_dim, num_labels)

    def forward(self, images: torch.Tensor, return_aux: bool = False):
        visual_features = self.backbone(images)
        logits = self.head(visual_features)
        if return_aux:
            return logits, {}
        return logits


class TEGARClassifier(nn.Module):
    def __init__(
        self,
        num_labels: int,
        kg_path: str,
        backbone: str = "resnet50",
        tegar_layers: int = 2,
        tegar_dim: int = 256,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        if backbone != "resnet50":
            raise ValueError(f"Unsupported backbone: {backbone}")
        self.backbone = ResNetBackbone(pretrained=pretrained)
        self.prototype_init = PrototypeInitializer(num_labels, self.backbone.feat_dim, tegar_dim)
        self.tegar = TEGAR(
            kg_path=kg_path,
            hidden_dim=tegar_dim,
            visual_dim=self.backbone.feat_dim,
            num_layers=tegar_layers,
        )
        self.head = nn.Linear(tegar_dim, 1)

    def forward(self, images: torch.Tensor, return_aux: bool = False):
        visual_features = self.backbone(images)
        prototypes = self.prototype_init(visual_features)
        if return_aux:
            refined, aux = self.tegar(prototypes, visual_features, return_aux=True)
        else:
            refined = self.tegar(prototypes, visual_features, return_aux=False)
            aux = {}
        logits = self.head(refined).squeeze(-1)
        if return_aux:
            return logits, aux
        return logits


class HomogeneousGATClassifier(nn.Module):
    def __init__(
        self,
        num_labels: int,
        kg_path: str,
        backbone: str = "resnet50",
        tegar_layers: int = 2,
        tegar_dim: int = 256,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        if backbone != "resnet50":
            raise ValueError(f"Unsupported backbone: {backbone}")
        self.backbone = ResNetBackbone(pretrained=pretrained)
        self.prototype_init = PrototypeInitializer(num_labels, self.backbone.feat_dim, tegar_dim)
        self.gat = HomogeneousGAT(kg_path=kg_path, hidden_dim=tegar_dim, num_layers=tegar_layers)
        self.head = nn.Linear(tegar_dim, 1)

    def forward(self, images: torch.Tensor, return_aux: bool = False):
        visual_features = self.backbone(images)
        prototypes = self.prototype_init(visual_features)
        if return_aux:
            refined, aux = self.gat(prototypes, return_aux=True)
        else:
            refined = self.gat(prototypes, return_aux=False)
            aux = {}
        logits = self.head(refined).squeeze(-1)
        if return_aux:
            return logits, aux
        return logits


def load_text_anchors(text_anchors_path: str | Path, num_labels: int) -> tuple[torch.Tensor, list[str]]:
    payload = torch.load(Path(text_anchors_path).resolve(), map_location="cpu", weights_only=False)
    anchors = payload["anchors"].float()
    label_names = list(payload["label_names"])
    if anchors.ndim != 2:
        raise ValueError(f"text_anchors.pt must store a 2D tensor, got shape {tuple(anchors.shape)}")
    if anchors.shape[0] != num_labels:
        raise ValueError(f"text anchor count mismatch: {anchors.shape[0]} vs num_labels={num_labels}")
    return F.normalize(anchors, dim=-1), label_names


class SpatialLabelAttention(nn.Module):
    def __init__(self, dim: int = 512, num_heads: int = 4) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.gate = nn.Parameter(torch.tensor(-2.0, dtype=torch.float32))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.zeros_(self.q_proj.bias)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.zeros_(self.k_proj.bias)
        nn.init.zeros_(self.v_proj.weight)
        nn.init.zeros_(self.v_proj.bias)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        text_anchors: torch.Tensor,
        patch_tokens: torch.Tensor,
        visual_cls: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = patch_tokens.size(0)
        num_labels = text_anchors.size(-2) if text_anchors.dim() == 3 else text_anchors.size(0)
        num_patches = patch_tokens.size(1)
        if text_anchors.dim() == 2:
            text_anchors = text_anchors.unsqueeze(0).expand(batch_size, -1, -1)

        q = self.q_proj(text_anchors)
        k = self.k_proj(patch_tokens)
        v = self.v_proj(patch_tokens)

        q = q.view(batch_size, num_labels, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, num_patches, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, num_patches, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn.float(), dim=-1).to(q.dtype)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, num_labels, -1)
        out = self.out_proj(out)
        base = visual_cls.unsqueeze(1).expand(batch_size, num_labels, -1)
        return base + torch.sigmoid(self.gate) * out


class GlobalLabelAttention(nn.Module):
    def __init__(self, dim: int = 512) -> None:
        super().__init__()
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.scale = dim ** -0.5
        self.gate = nn.Parameter(torch.tensor(-2.0, dtype=torch.float32))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.zeros_(self.k_proj.bias)
        nn.init.zeros_(self.v_proj.weight)
        nn.init.zeros_(self.v_proj.bias)

    def forward(self, text_anchors: torch.Tensor, visual_cls: torch.Tensor) -> torch.Tensor:
        batch_size = visual_cls.size(0)
        if text_anchors.dim() == 2:
            text_anchors = text_anchors.unsqueeze(0).expand(batch_size, -1, -1)
        key = self.k_proj(visual_cls).unsqueeze(1)
        value = self.v_proj(visual_cls).unsqueeze(1)
        attn = torch.sigmoid((text_anchors * key).sum(dim=-1, keepdim=True) * self.scale)
        base = visual_cls.unsqueeze(1).expand(batch_size, text_anchors.size(1), -1)
        return base + torch.sigmoid(self.gate) * attn * value


class ZeroShotCLIPBaseline(nn.Module):
    def __init__(
        self,
        num_labels: int,
        text_anchors_path: str,
        clip_model_name: str = "ViT-B-32",
        clip_pretrained_path: str | None = None,
        clip_default_pretrained: str | None = None,
    ) -> None:
        super().__init__()
        self.backbone = CLIPVisualBackbone(
            model_name=clip_model_name,
            pretrained_path=clip_pretrained_path,
            default_pretrained=clip_default_pretrained,
        )
        anchors, label_names = load_text_anchors(text_anchors_path, num_labels)
        self.register_buffer("text_anchors", anchors)
        self.label_names = label_names
        self.logit_scale = nn.Parameter(torch.tensor(math.log(100.0), dtype=torch.float32), requires_grad=False)

    def forward(self, images: torch.Tensor, return_aux: bool = False):
        visual = self.backbone(images, return_patches=False)
        logits = self.logit_scale.exp().clamp(max=100.0) * (visual @ self.text_anchors.T)
        if return_aux:
            return logits, {"visual_features": visual, "used_spatial_attn": False}
        return logits


class ZeroShotGraphClassifierBase(nn.Module):
    def __init__(
        self,
        num_labels: int,
        text_anchors_path: str,
        clip_model_name: str,
        clip_pretrained_path: str | None,
        clip_default_pretrained: str | None,
        graph_dim: int = 512,
        spatial_heads: int = 4,
    ) -> None:
        super().__init__()
        self.backbone = CLIPVisualBackbone(
            model_name=clip_model_name,
            pretrained_path=clip_pretrained_path,
            default_pretrained=clip_default_pretrained,
        )
        anchors, label_names = load_text_anchors(text_anchors_path, num_labels)
        if anchors.shape[1] != graph_dim:
            raise ValueError(
                f"Text anchor dim ({anchors.shape[1]}) must equal graph_dim ({graph_dim}) for CLIP-space reasoning."
            )
        self.register_buffer("text_anchors", anchors)
        self.label_names = label_names
        self.spatial_attn = SpatialLabelAttention(dim=graph_dim, num_heads=spatial_heads)
        self.global_attn = GlobalLabelAttention(dim=graph_dim)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(100.0), dtype=torch.float32))

    def init_label_nodes(self, visual_cls: torch.Tensor, patch_tokens: torch.Tensor | None) -> tuple[torch.Tensor, bool]:
        if patch_tokens is not None:
            return self.spatial_attn(self.text_anchors, patch_tokens, visual_cls), True
        return self.global_attn(self.text_anchors, visual_cls), False

    def refine_nodes(
        self,
        label_nodes: torch.Tensor,
        visual_features: torch.Tensor,
        return_aux: bool = False,
    ):
        raise NotImplementedError

    def match_nodes(self, node_repr: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        aligned_nodes = F.normalize(node_repr, dim=-1)
        logits = self.logit_scale.exp().clamp(max=100.0) * (aligned_nodes * self.text_anchors.unsqueeze(0)).sum(dim=-1)
        return logits, aligned_nodes

    def compute_anchor_regularization(
        self,
        aligned_nodes: torch.Tensor,
        initial_nodes: torch.Tensor,
        seen_indices: list[int] | None = None,
    ) -> torch.Tensor:
        if seen_indices:
            aligned_nodes = aligned_nodes[:, seen_indices, :]
            initial_nodes = initial_nodes[:, seen_indices, :]
        if aligned_nodes.numel() == 0:
            return aligned_nodes.new_zeros(())
        initial_nodes = F.normalize(initial_nodes, dim=-1)
        similarity = (aligned_nodes * initial_nodes).sum(dim=-1)
        return (1.0 - similarity).mean()

    def forward(self, images: torch.Tensor, return_aux: bool = False):
        backbone_output = self.backbone(images, return_patches=True)
        if isinstance(backbone_output, tuple):
            visual_cls, patch_tokens = backbone_output
        else:
            visual_cls = backbone_output
            patch_tokens = None
        initial_nodes, used_spatial_attn = self.init_label_nodes(visual_cls, patch_tokens)
        if return_aux:
            final_nodes, aux = self.refine_nodes(initial_nodes, visual_cls, return_aux=True)
        else:
            final_nodes = self.refine_nodes(initial_nodes, visual_cls, return_aux=False)
            aux = {}
        logits, aligned_nodes = self.match_nodes(final_nodes)
        if return_aux:
            aux = dict(aux)
            aux["visual_features"] = visual_cls
            aux["initial_nodes"] = initial_nodes
            aux["final_nodes"] = final_nodes
            aux["aligned_nodes"] = aligned_nodes
            aux["used_spatial_attn"] = used_spatial_attn
            return logits, aux
        return logits


class ZeroShotTEGARClassifier(ZeroShotGraphClassifierBase):
    def __init__(
        self,
        num_labels: int,
        kg_path: str,
        text_anchors_path: str,
        clip_model_name: str = "ViT-B-32",
        clip_pretrained_path: str | None = None,
        clip_default_pretrained: str | None = None,
        tegar_layers: int = 2,
        tegar_dim: int = 512,
        exclusion_beta_init: float = 0.01,
        dropout: float = 0.1,
        spatial_heads: int = 4,
    ) -> None:
        super().__init__(
            num_labels=num_labels,
            text_anchors_path=text_anchors_path,
            clip_model_name=clip_model_name,
            clip_pretrained_path=clip_pretrained_path,
            clip_default_pretrained=clip_default_pretrained,
            graph_dim=tegar_dim,
            spatial_heads=spatial_heads,
        )
        self.tegar = TEGAR(
            kg_path=kg_path,
            hidden_dim=tegar_dim,
            visual_dim=tegar_dim,
            num_layers=tegar_layers,
            dropout=dropout,
            exclusion_beta_init=exclusion_beta_init,
            apply_layernorm=False,
            apply_activation=False,
        )

    def refine_nodes(
        self,
        label_nodes: torch.Tensor,
        visual_features: torch.Tensor,
        return_aux: bool = False,
    ):
        if return_aux:
            return self.tegar(label_nodes, visual_features, return_aux=True)
        return self.tegar(label_nodes, visual_features, return_aux=False)


class ZeroShotHomoGATClassifier(ZeroShotGraphClassifierBase):
    def __init__(
        self,
        num_labels: int,
        kg_path: str,
        text_anchors_path: str,
        clip_model_name: str = "ViT-B-32",
        clip_pretrained_path: str | None = None,
        clip_default_pretrained: str | None = None,
        tegar_layers: int = 2,
        tegar_dim: int = 512,
        dropout: float = 0.1,
        spatial_heads: int = 4,
    ) -> None:
        super().__init__(
            num_labels=num_labels,
            text_anchors_path=text_anchors_path,
            clip_model_name=clip_model_name,
            clip_pretrained_path=clip_pretrained_path,
            clip_default_pretrained=clip_default_pretrained,
            graph_dim=tegar_dim,
            spatial_heads=spatial_heads,
        )
        self.gat = HomogeneousGAT(
            kg_path=kg_path,
            hidden_dim=tegar_dim,
            num_layers=tegar_layers,
            dropout=dropout,
            apply_layernorm=False,
            apply_activation=False,
        )

    def refine_nodes(
        self,
        label_nodes: torch.Tensor,
        visual_features: torch.Tensor,
        return_aux: bool = False,
    ):
        if return_aux:
            return self.gat(label_nodes, return_aux=True)
        return self.gat(label_nodes, return_aux=False)


class LabelConditionedVisual(nn.Module):
    def __init__(self, clip_dim: int = 512, output_dim: int = 256) -> None:
        super().__init__()
        self.q_proj = nn.Linear(clip_dim, output_dim)
        self.k_proj = nn.Linear(clip_dim, output_dim)
        self.v_proj = nn.Linear(clip_dim, output_dim)
        self.scale = output_dim ** -0.5
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.q_proj.weight, gain=0.1)
        nn.init.zeros_(self.q_proj.bias)
        nn.init.xavier_uniform_(self.k_proj.weight, gain=0.1)
        nn.init.zeros_(self.k_proj.bias)
        nn.init.xavier_uniform_(self.v_proj.weight, gain=0.1)
        nn.init.zeros_(self.v_proj.bias)

    def forward(self, text_anchors: torch.Tensor, visual_cls: torch.Tensor) -> torch.Tensor:
        batch_size = visual_cls.size(0)
        q = self.q_proj(text_anchors).unsqueeze(0).expand(batch_size, -1, -1)
        k = self.k_proj(visual_cls).unsqueeze(1)
        v = self.v_proj(visual_cls).unsqueeze(1)
        attn = torch.sigmoid((q * k).sum(dim=-1, keepdim=True) * self.scale)
        return attn * v


class ExplicitUnseenPropagation(nn.Module):
    def __init__(self, kg_path: str) -> None:
        super().__init__()
        kg = KGDefinition(kg_path)
        adj = kg.get_adjacency_matrices(sparse=False)
        self.register_buffer("adj_cooccur", adj["often_cooccur"])
        self.register_buffer("adj_exclusion", adj["statistical_exclusion"])
        self.register_buffer("adj_hierarchical", adj["hierarchical"])
        self.w_cooccur_raw = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.w_excl_raw = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
        self.w_hier_raw = nn.Parameter(torch.tensor(0.8, dtype=torch.float32))

    def forward(self, delta: torch.Tensor, seen_indices: list[int], unseen_indices: list[int]) -> torch.Tensor:
        if not seen_indices or not unseen_indices:
            return delta
        seen_t = torch.tensor(seen_indices, dtype=torch.long, device=delta.device)
        unseen_t = torch.tensor(unseen_indices, dtype=torch.long, device=delta.device)

        cooccur_sub = self.adj_cooccur[unseen_t][:, seen_t] + self.adj_cooccur[seen_t][:, unseen_t].T
        exclusion_sub = self.adj_exclusion[unseen_t][:, seen_t] + self.adj_exclusion[seen_t][:, unseen_t].T
        hier_sub = self.adj_hierarchical[unseen_t][:, seen_t] + self.adj_hierarchical[seen_t][:, unseen_t].T

        cooccur_sub = (cooccur_sub > 0).float()
        exclusion_sub = (exclusion_sub > 0).float()
        hier_sub = (hier_sub > 0).float()

        w_cooccur = F.softplus(self.w_cooccur_raw)
        w_excl = F.softplus(self.w_excl_raw)
        w_hier = F.softplus(self.w_hier_raw)
        weight_matrix = w_cooccur * cooccur_sub - w_excl * exclusion_sub + w_hier * hier_sub
        valid_rows = weight_matrix.abs().sum(dim=1) > 1e-6
        norm = weight_matrix.abs().sum(dim=1, keepdim=True).clamp_min(1e-6)
        weight_matrix = weight_matrix / norm

        seen_delta = delta[:, seen_t]
        unseen_delta = seen_delta @ weight_matrix.T

        result = delta.clone()
        if valid_rows.any():
            result[:, unseen_t[valid_rows]] = unseen_delta[:, valid_rows]
        return result


class ExplicitUnseenPropagationHomogeneous(nn.Module):
    def __init__(self, kg_path: str) -> None:
        super().__init__()
        kg = KGDefinition(kg_path)
        adj = kg.get_homogeneous_adjacency(sparse=False)
        self.register_buffer("adj_homo", adj)
        self.w_raw = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

    def forward(self, delta: torch.Tensor, seen_indices: list[int], unseen_indices: list[int]) -> torch.Tensor:
        if not seen_indices or not unseen_indices:
            return delta
        seen_t = torch.tensor(seen_indices, dtype=torch.long, device=delta.device)
        unseen_t = torch.tensor(unseen_indices, dtype=torch.long, device=delta.device)
        adjacency = (self.adj_homo[unseen_t][:, seen_t] > 0).float()
        weight = F.softplus(self.w_raw) * adjacency
        valid_rows = weight.sum(dim=1) > 1e-6
        weight = weight / weight.sum(dim=1, keepdim=True).clamp_min(1e-6)
        seen_delta = delta[:, seen_t]
        unseen_delta = seen_delta @ weight.T
        result = delta.clone()
        if valid_rows.any():
            result[:, unseen_t[valid_rows]] = unseen_delta[:, valid_rows]
        return result


class ResidualGraphClassifier(nn.Module):
    def __init__(
        self,
        num_labels: int,
        kg_path: str,
        text_anchors_path: str,
        clip_model_name: str = "ViT-B-32",
        clip_pretrained_path: str | None = None,
        clip_default_pretrained: str | None = None,
        graph_type: str = "tegar",
        graph_dim: int = 256,
        graph_layers: int = 2,
        exclusion_beta_init: float = 0.01,
        dropout: float = 0.1,
        delta_scale_init: float = 1.0,
        use_explicit_propagation: bool = True,
    ) -> None:
        super().__init__()
        if graph_type not in {"none", "tegar", "homo_gat"}:
            raise ValueError(f"Unsupported graph_type: {graph_type}")
        self.graph_type = graph_type
        self.backbone = CLIPVisualBackbone(
            model_name=clip_model_name,
            pretrained_path=clip_pretrained_path,
            default_pretrained=clip_default_pretrained,
        )
        anchors, label_names = load_text_anchors(text_anchors_path, num_labels)
        self.register_buffer("text_anchors", anchors)
        self.label_names = label_names
        self.num_labels = num_labels
        self.graph_dim = int(graph_dim)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(100.0), dtype=torch.float32))
        self.seen_indices: list[int] = []
        self.unseen_indices: list[int] = []
        self.use_explicit_propagation = bool(use_explicit_propagation)

        if self.graph_type == "none":
            return

        self.label_proj = nn.Linear(self.text_anchors.shape[1], self.graph_dim)
        self.logit_proj = nn.Linear(1, self.graph_dim)
        self.label_visual = LabelConditionedVisual(
            clip_dim=self.text_anchors.shape[1],
            output_dim=self.graph_dim,
        )
        self.fusion = nn.Linear(self.graph_dim * 3, self.graph_dim)

        if self.graph_type == "tegar":
            self.graph = TEGAR(
                kg_path=kg_path,
                hidden_dim=self.graph_dim,
                visual_dim=self.backbone.feat_dim,
                num_layers=graph_layers,
                dropout=dropout,
                exclusion_beta_init=exclusion_beta_init,
                apply_layernorm=False,
                apply_activation=False,
            )
            self.unseen_propagation = (
                ExplicitUnseenPropagation(kg_path=kg_path) if self.use_explicit_propagation else None
            )
        else:
            self.graph = HomogeneousGAT(
                kg_path=kg_path,
                hidden_dim=self.graph_dim,
                num_layers=graph_layers,
                dropout=dropout,
                apply_layernorm=False,
                apply_activation=False,
            )
            self.unseen_propagation = (
                ExplicitUnseenPropagationHomogeneous(kg_path=kg_path) if self.use_explicit_propagation else None
            )

        self.delta_head = nn.Sequential(
            nn.Linear(self.graph_dim, self.graph_dim),
            nn.Linear(self.graph_dim, 1),
        )
        self.delta_scale = nn.Parameter(torch.tensor(float(delta_scale_init), dtype=torch.float32))
        self._init_weights()

    def _init_weights(self) -> None:
        if self.graph_type == "none":
            return
        nn.init.xavier_uniform_(self.label_proj.weight, gain=0.1)
        nn.init.zeros_(self.label_proj.bias)
        nn.init.xavier_uniform_(self.logit_proj.weight, gain=0.1)
        nn.init.zeros_(self.logit_proj.bias)
        nn.init.xavier_uniform_(self.fusion.weight, gain=0.1)
        nn.init.zeros_(self.fusion.bias)
        nn.init.xavier_uniform_(self.delta_head[0].weight, gain=0.1)
        nn.init.zeros_(self.delta_head[0].bias)
        nn.init.zeros_(self.delta_head[1].weight)
        nn.init.zeros_(self.delta_head[1].bias)

    def set_seen_unseen_indices(self, seen_indices: list[int], unseen_indices: list[int]) -> None:
        self.seen_indices = list(seen_indices)
        self.unseen_indices = list(unseen_indices)

    def compute_clip_logits(self, visual_cls: torch.Tensor) -> torch.Tensor:
        anchors = F.normalize(self.text_anchors, dim=-1)
        visual_norm = F.normalize(visual_cls, dim=-1)
        logit_scale = self.logit_scale.exp().clamp(max=100.0)
        return logit_scale * (visual_norm @ anchors.T)

    def compute_graph_delta(
        self,
        visual_cls: torch.Tensor,
        clip_logits: torch.Tensor,
        return_aux: bool = False,
    ):
        batch_size = visual_cls.size(0)
        label_feat = self.label_proj(self.text_anchors).unsqueeze(0).expand(batch_size, -1, -1)
        visual_feat = self.label_visual(self.text_anchors, visual_cls)
        clip_probs = torch.sigmoid(clip_logits)
        logit_feat = self.logit_proj(clip_probs.unsqueeze(-1))
        node_features = self.fusion(torch.cat([label_feat, logit_feat, visual_feat], dim=-1))

        if self.graph_type == "tegar":
            if return_aux:
                node_repr, aux = self.graph(node_features, visual_cls, return_aux=True)
            else:
                node_repr = self.graph(node_features, visual_cls, return_aux=False)
                aux = {}
        else:
            if return_aux:
                node_repr, aux = self.graph(node_features, return_aux=True)
            else:
                node_repr = self.graph(node_features, return_aux=False)
                aux = {}

        raw_delta = self.delta_head(node_repr).squeeze(-1)
        return raw_delta, node_features, node_repr, aux

    def forward(self, images: torch.Tensor, return_aux: bool = False):
        visual_cls = self.backbone(images, return_patches=False)
        clip_logits = self.compute_clip_logits(visual_cls)

        if self.graph_type == "none":
            zero_delta = torch.zeros_like(clip_logits)
            if return_aux:
                return clip_logits, {
                    "visual_features": visual_cls,
                    "clip_logits": clip_logits,
                    "delta": zero_delta,
                    "raw_delta": zero_delta,
                    "delta_scale": 0.0,
                    "mean_abs_delta": 0.0,
                    "used_spatial_attn": False,
                }
            return clip_logits

        raw_delta, node_features, node_repr, aux = self.compute_graph_delta(
            visual_cls,
            clip_logits,
            return_aux=return_aux,
        )
        scaled_delta = self.delta_scale * raw_delta
        if self.unseen_propagation is not None and self.use_explicit_propagation and self.unseen_indices:
            scaled_delta = self.unseen_propagation(scaled_delta, self.seen_indices, self.unseen_indices)
        final_logits = clip_logits + scaled_delta

        if not return_aux:
            return final_logits

        aux = dict(aux)
        aux.update(
            {
                "visual_features": visual_cls,
                "clip_logits": clip_logits,
                "delta": scaled_delta,
                "raw_delta": raw_delta,
                "delta_scale": float(self.delta_scale.detach().cpu().item()),
                "mean_abs_delta": float(scaled_delta.detach().abs().mean().cpu().item()),
                "node_features": node_features,
                "node_repr": node_repr,
                "used_spatial_attn": False,
                "seen_indices": list(self.seen_indices),
                "unseen_indices": list(self.unseen_indices),
            }
        )
        return final_logits, aux


class PatchCrossAttention(nn.Module):
    def __init__(self, dim: int = 512, num_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Parameter(torch.tensor(-2.0, dtype=torch.float32))
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.q_proj.weight, gain=0.1)
        nn.init.zeros_(self.q_proj.bias)
        nn.init.xavier_uniform_(self.k_proj.weight, gain=0.1)
        nn.init.zeros_(self.k_proj.bias)
        nn.init.zeros_(self.v_proj.weight)
        nn.init.zeros_(self.v_proj.bias)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        fallback_cls: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, num_labels, dim = query.shape
        num_patches = key.size(1)

        q = self.q_proj(query).view(batch_size, num_labels, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, num_patches, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, num_patches, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn.float(), dim=-1).to(q.dtype)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(batch_size, num_labels, dim)
        out = self.out_proj(out)

        if fallback_cls is not None:
            base = fallback_cls.unsqueeze(1).expand(batch_size, num_labels, -1)
        else:
            base = query
        return base + torch.sigmoid(self.gate) * out


class GraphCrossAttentionClassifier(nn.Module):
    def __init__(
        self,
        num_labels: int,
        kg_path: str,
        text_anchors_path: str,
        clip_model_name: str = "ViT-B-32",
        clip_pretrained_path: str | None = None,
        clip_default_pretrained: str | None = None,
        graph_type: str = "tegar",
        tegar_layers: int = 2,
        dropout: float = 0.1,
        exclusion_beta_init: float = 0.01,
        cross_attn_heads: int = 4,
        use_patches: bool = True,
    ) -> None:
        super().__init__()
        if graph_type not in {"tegar", "homo_gat"}:
            raise ValueError(f"Unsupported graph_type: {graph_type}")
        self.graph_type = graph_type
        self.use_patches = use_patches
        self.backbone = CLIPVisualBackbone(
            model_name=clip_model_name,
            pretrained_path=clip_pretrained_path,
            default_pretrained=clip_default_pretrained,
        )
        anchors, label_names = load_text_anchors(text_anchors_path, num_labels)
        self.register_buffer("text_anchors", anchors)
        self.label_names = label_names
        self.num_labels = num_labels
        self.feat_dim = anchors.shape[1]

        if self.graph_type == "tegar":
            self.graph = TEGAR(
                kg_path=kg_path,
                hidden_dim=self.feat_dim,
                visual_dim=self.feat_dim,
                num_layers=tegar_layers,
                dropout=dropout,
                exclusion_beta_init=exclusion_beta_init,
                apply_layernorm=False,
                apply_activation=False,
            )
        else:
            self.graph = HomogeneousGAT(
                kg_path=kg_path,
                hidden_dim=self.feat_dim,
                num_layers=tegar_layers,
                dropout=dropout,
                apply_layernorm=False,
                apply_activation=False,
            )

        self.cross_attn = PatchCrossAttention(
            dim=self.feat_dim,
            num_heads=cross_attn_heads,
            dropout=dropout,
        )
        self.logit_scale = nn.Parameter(torch.tensor(math.log(100.0), dtype=torch.float32))

    def compute_clip_logits(self, visual_cls: torch.Tensor) -> torch.Tensor:
        anchors = F.normalize(self.text_anchors, dim=-1)
        visual_norm = F.normalize(visual_cls, dim=-1)
        logit_scale = self.logit_scale.exp().clamp(max=100.0)
        return logit_scale * (visual_norm @ anchors.T)

    def forward(self, images: torch.Tensor, return_aux: bool = False):
        backbone_output = self.backbone(images, return_patches=self.use_patches)
        if isinstance(backbone_output, tuple):
            visual_cls, patches = backbone_output
        else:
            visual_cls = backbone_output
            patches = None

        batch_size = visual_cls.size(0)
        label_input = self.text_anchors.unsqueeze(0).expand(batch_size, -1, -1)

        if self.graph_type == "tegar":
            if return_aux:
                enhanced_labels, graph_aux = self.graph(label_input, visual_cls, return_aux=True)
            else:
                enhanced_labels = self.graph(label_input, visual_cls, return_aux=False)
                graph_aux = {}
        else:
            if return_aux:
                enhanced_labels, graph_aux = self.graph(label_input, return_aux=True)
            else:
                enhanced_labels = self.graph(label_input, return_aux=False)
                graph_aux = {}

        if patches is not None and self.use_patches:
            visual_evidence = self.cross_attn(
                query=enhanced_labels,
                key=patches,
                value=patches,
                fallback_cls=visual_cls,
            )
            used_patches = True
        else:
            visual_evidence = visual_cls.unsqueeze(1).expand(batch_size, self.num_labels, -1)
            used_patches = False

        evidence_norm = F.normalize(visual_evidence, dim=-1)
        anchors_norm = F.normalize(self.text_anchors, dim=-1)
        logit_scale = self.logit_scale.exp().clamp(max=100.0)
        logits = logit_scale * (evidence_norm * anchors_norm.unsqueeze(0)).sum(dim=-1)
        clip_logits = self.compute_clip_logits(visual_cls)

        if not return_aux:
            return logits

        enhanced_anchor_sim = float(
            (F.normalize(enhanced_labels, dim=-1) * anchors_norm.unsqueeze(0)).sum(dim=-1).mean().detach().cpu().item()
        )
        evidence_cls_sim = float(
            (F.normalize(visual_evidence, dim=-1) * visual_cls.unsqueeze(1)).sum(dim=-1).mean().detach().cpu().item()
        )
        aux = dict(graph_aux)
        aux.update(
            {
                "visual_features": visual_cls,
                "visual_cls": visual_cls,
                "patches": patches,
                "clip_logits": clip_logits,
                "initial_nodes": label_input,
                "final_nodes": enhanced_labels,
                "enhanced_labels": enhanced_labels,
                "visual_evidence": visual_evidence,
                "used_patches": used_patches,
                "enhanced_anchor_sim": enhanced_anchor_sim,
                "evidence_cls_sim": evidence_cls_sim,
                "used_spatial_attn": used_patches,
            }
        )
        return logits, aux


class TEGARClassifierGenerator(nn.Module):
    def __init__(
        self,
        num_labels: int,
        kg_path: str,
        text_anchors_path: str,
        clip_model_name: str = "ViT-B-32",
        clip_pretrained_path: str | None = None,
        clip_default_pretrained: str | None = None,
        graph_type: str = "tegar",
        tegar_layers: int = 2,
        dropout: float = 0.1,
        exclusion_beta_init: float = 0.01,
        trainable_projections: bool = True,
        trainable_logit_scale: bool = True,
        freeze_visual_proj: bool = False,
        use_prototype_regression: bool = False,
        use_cached_features: bool = False,
    ) -> None:
        super().__init__()
        if graph_type not in {"none", "tegar", "homo_gat"}:
            raise ValueError(f"Unknown graph_type: {graph_type}")
        self.graph_type = graph_type
        self.use_prototype_regression = bool(use_prototype_regression)
        self.use_cached_features = bool(use_cached_features)
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
        self.num_labels = num_labels
        self.feat_dim = anchors.shape[1]

        if graph_type == "tegar":
            self.graph = TEGAR(
                kg_path=kg_path,
                hidden_dim=self.feat_dim,
                visual_dim=self.feat_dim,
                num_layers=tegar_layers,
                dropout=dropout,
                exclusion_beta_init=exclusion_beta_init,
                apply_layernorm=False,
                apply_activation=False,
            )
        elif graph_type == "homo_gat":
            self.graph = HomogeneousGAT(
                kg_path=kg_path,
                hidden_dim=self.feat_dim,
                num_layers=tegar_layers,
                dropout=dropout,
                apply_layernorm=False,
                apply_activation=False,
            )
        else:
            self.graph = None

        if self.use_prototype_regression:
            self.output_norm = nn.LayerNorm(self.feat_dim)
        else:
            self.visual_proj = nn.Linear(self.feat_dim, self.feat_dim, bias=False)
            self.classifier_proj = nn.Linear(self.feat_dim, self.feat_dim, bias=False)
            self.logit_scale = nn.Parameter(
                torch.tensor(math.log(100.0), dtype=torch.float32),
                requires_grad=trainable_logit_scale,
            )
        self._init_weights()

        if not self.use_prototype_regression:
            if not trainable_projections:
                for parameter in self.visual_proj.parameters():
                    parameter.requires_grad = False
                for parameter in self.classifier_proj.parameters():
                    parameter.requires_grad = False
            elif freeze_visual_proj:
                for parameter in self.visual_proj.parameters():
                    parameter.requires_grad = False

    def _init_weights(self) -> None:
        if hasattr(self, "visual_proj"):
            nn.init.eye_(self.visual_proj.weight)
        if hasattr(self, "classifier_proj"):
            nn.init.eye_(self.classifier_proj.weight)

    def compute_clip_logits(self, visual_cls: torch.Tensor) -> torch.Tensor:
        anchors = F.normalize(self.text_anchors, dim=-1)
        visual_norm = F.normalize(visual_cls, dim=-1)
        if hasattr(self, "logit_scale"):
            logit_scale = self.logit_scale.exp().clamp(max=100.0)
        else:
            logit_scale = visual_cls.new_tensor(100.0)
        return logit_scale * (visual_norm @ anchors.T)

    def generate_classifiers(self, visual_cls: torch.Tensor, return_aux: bool = False):
        batch_size = visual_cls.size(0)
        label_input = self.text_anchors.unsqueeze(0).expand(batch_size, -1, -1)
        if self.graph is None:
            if return_aux:
                return label_input, {}
            return label_input
        if self.graph_type == "tegar":
            if return_aux:
                return self.graph(label_input, visual_cls, return_aux=True)
            return self.graph(label_input, visual_cls, return_aux=False)
        if return_aux:
            return self.graph(label_input, return_aux=True)
        return self.graph(label_input, return_aux=False)

    def forward_from_features(self, visual_cls: torch.Tensor, return_aux: bool = False):
        if self.use_prototype_regression:
            return self._forward_prototype_mode(visual_cls, return_aux=return_aux)
        if return_aux:
            classifiers, graph_aux = self.generate_classifiers(visual_cls, return_aux=True)
        else:
            classifiers = self.generate_classifiers(visual_cls, return_aux=False)
            graph_aux = {}

        projected_visual = F.normalize(self.visual_proj(visual_cls), dim=-1)
        projected_classifiers = F.normalize(self.classifier_proj(classifiers), dim=-1)
        logit_scale = self.logit_scale.exp().clamp(max=100.0)
        logits = logit_scale * (projected_visual.unsqueeze(1) * projected_classifiers).sum(dim=-1)

        if not return_aux:
            return logits

        anchors_norm = F.normalize(self.text_anchors, dim=-1)
        classifier_anchor_sim = float(
            (F.normalize(classifiers, dim=-1) * anchors_norm.unsqueeze(0)).sum(dim=-1).mean().detach().cpu().item()
        )
        aux = dict(graph_aux) if isinstance(graph_aux, dict) else {}
        aux.update(
            {
                "visual_features": visual_cls,
                "visual_cls": visual_cls,
                "clip_logits": self.compute_clip_logits(visual_cls),
                "classifiers": classifiers,
                "initial_nodes": self.text_anchors.unsqueeze(0).expand(visual_cls.size(0), -1, -1),
                "final_nodes": classifiers,
                "classifier_anchor_sim": classifier_anchor_sim,
                "logit_scale_value": float(logit_scale.detach().cpu().item()),
                "used_spatial_attn": False,
                "used_patches": False,
            }
        )
        return logits, aux

    def _forward_prototype_mode(self, visual_cls: torch.Tensor, return_aux: bool = False):
        batch_size = visual_cls.size(0)
        label_input = self.text_anchors.unsqueeze(0).expand(batch_size, -1, -1)
        if self.graph_type == "tegar":
            if return_aux:
                classifiers, graph_aux = self.graph(label_input, visual_cls, return_aux=True)
            else:
                classifiers = self.graph(label_input, visual_cls, return_aux=False)
                graph_aux = {}
        elif self.graph_type == "homo_gat":
            if return_aux:
                classifiers, graph_aux = self.graph(label_input, return_aux=True)
            else:
                classifiers = self.graph(label_input, return_aux=False)
                graph_aux = {}
        else:
            classifiers = label_input
            graph_aux = {}

        visual_norm = F.normalize(visual_cls, dim=-1)
        transformed_classifiers = self.output_norm(classifiers)
        classifier_norm = F.normalize(transformed_classifiers, dim=-1)
        logits = 100.0 * (visual_norm.unsqueeze(1) * classifier_norm).sum(dim=-1)

        if not return_aux:
            return logits

        anchors_norm = F.normalize(self.text_anchors, dim=-1)
        classifier_anchor_sim = float(
            (classifier_norm * anchors_norm.unsqueeze(0)).sum(dim=-1).mean().detach().cpu().item()
        )
        aux = dict(graph_aux) if isinstance(graph_aux, dict) else {}
        aux.update(
            {
                "visual_features": visual_cls,
                "visual_cls": visual_cls,
                "clip_logits": 100.0
                * (F.normalize(visual_cls, dim=-1) @ F.normalize(self.text_anchors, dim=-1).T),
                "classifiers": classifiers,
                "initial_nodes": label_input,
                "final_nodes": classifiers,
                "classifier_anchor_sim": classifier_anchor_sim,
                "used_spatial_attn": False,
                "used_patches": False,
            }
        )
        return logits, aux

    def forward(self, images: torch.Tensor, return_aux: bool = False):
        if self.backbone is None:
            raise RuntimeError("This model was created for cached features only. Use forward_from_features().")
        visual_cls = self.backbone(images, return_patches=False)
        return self.forward_from_features(visual_cls, return_aux=return_aux)


def create_model(
    mode: str,
    num_labels: int,
    kg_path: str,
    backbone: str = "resnet50",
    tegar_layers: int = 2,
    tegar_dim: int = 256,
    pretrained: bool = True,
    clip_model_name: str = "ViT-B-32",
    clip_pretrained_path: str | None = None,
    clip_default_pretrained: str | None = None,
    text_anchors_path: str | None = None,
    exclusion_beta_init: float = 0.01,
    dropout: float = 0.1,
    spatial_heads: int = 4,
    cross_attn_heads: int = 4,
    delta_scale_init: float = 1.0,
    use_explicit_propagation: bool = True,
    use_cached_features: bool = False,
) -> nn.Module:
    if mode == "baseline":
        return BaselineClassifier(num_labels=num_labels, backbone=backbone, pretrained=pretrained)
    if mode == "tegar":
        return TEGARClassifier(
            num_labels=num_labels,
            kg_path=kg_path,
            backbone=backbone,
            tegar_layers=tegar_layers,
            tegar_dim=tegar_dim,
            pretrained=pretrained,
        )
    if mode == "homogeneous_gat":
        return HomogeneousGATClassifier(
            num_labels=num_labels,
            kg_path=kg_path,
            backbone=backbone,
            tegar_layers=tegar_layers,
            tegar_dim=tegar_dim,
            pretrained=pretrained,
        )

    if text_anchors_path is None:
        raise ValueError(f"Mode {mode} requires text_anchors_path.")

    if mode == "zs_clip":
        return TEGARClassifierGenerator(
            num_labels=num_labels,
            kg_path=kg_path,
            text_anchors_path=text_anchors_path,
            clip_model_name=clip_model_name,
            clip_pretrained_path=clip_pretrained_path,
            clip_default_pretrained=clip_default_pretrained,
            graph_type="none",
            tegar_layers=tegar_layers,
            dropout=dropout,
            exclusion_beta_init=exclusion_beta_init,
            trainable_projections=False,
            trainable_logit_scale=False,
            use_cached_features=use_cached_features,
        )
    if mode == "zs_proj_only":
        return TEGARClassifierGenerator(
            num_labels=num_labels,
            kg_path=kg_path,
            text_anchors_path=text_anchors_path,
            clip_model_name=clip_model_name,
            clip_pretrained_path=clip_pretrained_path,
            clip_default_pretrained=clip_default_pretrained,
            graph_type="none",
            tegar_layers=tegar_layers,
            dropout=dropout,
            exclusion_beta_init=exclusion_beta_init,
            trainable_projections=True,
            trainable_logit_scale=True,
            use_cached_features=use_cached_features,
        )
    if mode == "zs_homo_gen":
        return TEGARClassifierGenerator(
            num_labels=num_labels,
            kg_path=kg_path,
            text_anchors_path=text_anchors_path,
            clip_model_name=clip_model_name,
            clip_pretrained_path=clip_pretrained_path,
            clip_default_pretrained=clip_default_pretrained,
            graph_type="homo_gat",
            tegar_layers=tegar_layers,
            dropout=dropout,
            exclusion_beta_init=exclusion_beta_init,
            trainable_projections=True,
            trainable_logit_scale=True,
            freeze_visual_proj=True,
            use_cached_features=use_cached_features,
        )
    if mode == "zs_tegar_gen":
        return TEGARClassifierGenerator(
            num_labels=num_labels,
            kg_path=kg_path,
            text_anchors_path=text_anchors_path,
            clip_model_name=clip_model_name,
            clip_pretrained_path=clip_pretrained_path,
            clip_default_pretrained=clip_default_pretrained,
            graph_type="tegar",
            tegar_layers=tegar_layers,
            dropout=dropout,
            exclusion_beta_init=exclusion_beta_init,
            trainable_projections=True,
            trainable_logit_scale=True,
            freeze_visual_proj=True,
            use_cached_features=use_cached_features,
        )
    if mode == "zs_proto_only":
        return TEGARClassifierGenerator(
            num_labels=num_labels,
            kg_path=kg_path,
            text_anchors_path=text_anchors_path,
            clip_model_name=clip_model_name,
            clip_pretrained_path=clip_pretrained_path,
            clip_default_pretrained=clip_default_pretrained,
            graph_type="none",
            tegar_layers=tegar_layers,
            dropout=dropout,
            exclusion_beta_init=exclusion_beta_init,
            trainable_projections=False,
            trainable_logit_scale=False,
            use_prototype_regression=True,
            use_cached_features=use_cached_features,
        )
    if mode == "zs_homo_proto":
        return TEGARClassifierGenerator(
            num_labels=num_labels,
            kg_path=kg_path,
            text_anchors_path=text_anchors_path,
            clip_model_name=clip_model_name,
            clip_pretrained_path=clip_pretrained_path,
            clip_default_pretrained=clip_default_pretrained,
            graph_type="homo_gat",
            tegar_layers=tegar_layers,
            dropout=dropout,
            exclusion_beta_init=exclusion_beta_init,
            trainable_projections=False,
            trainable_logit_scale=False,
            use_prototype_regression=True,
            use_cached_features=use_cached_features,
        )
    if mode == "zs_tegar_proto":
        return TEGARClassifierGenerator(
            num_labels=num_labels,
            kg_path=kg_path,
            text_anchors_path=text_anchors_path,
            clip_model_name=clip_model_name,
            clip_pretrained_path=clip_pretrained_path,
            clip_default_pretrained=clip_default_pretrained,
            graph_type="tegar",
            tegar_layers=tegar_layers,
            dropout=dropout,
            exclusion_beta_init=exclusion_beta_init,
            trainable_projections=False,
            trainable_logit_scale=False,
            use_prototype_regression=True,
            use_cached_features=use_cached_features,
        )
    if mode in {
        "zs_logit_tegar",
        "zs_logit_homo",
        "zs_logit_none",
        "zs_logit_tegar_no_gate",
        "zs_logit_tegar_no_uncertainty_gate",
        "zs_logit_tegar_no_experts",
        "zs_logit_tegar_no_consensus",
        "zs_logit_tegar_no_exclusion_neg",
        "zs_logit_tegar_reg",
        "zs_logit_homo_reg",
        "zs_logit_tegar_fixed_temp",
        "zs_logit_tegar_k2",
        "zs_logit_tegar_k8",
    }:
        from models.logit_refiner import ProbabilitySpaceRefiner

        logit_refiner_hidden_dim = 128
        graph_type = {
            "zs_logit_tegar": "tegar",
            "zs_logit_homo": "homo_gat",
            "zs_logit_none": "none",
            "zs_logit_tegar_no_gate": "tegar",
            "zs_logit_tegar_no_uncertainty_gate": "tegar",
            "zs_logit_tegar_no_experts": "tegar",
            "zs_logit_tegar_no_consensus": "tegar",
            "zs_logit_tegar_no_exclusion_neg": "tegar",
            "zs_logit_tegar_reg": "tegar",
            "zs_logit_homo_reg": "homo_gat",
            "zs_logit_tegar_fixed_temp": "tegar",
            "zs_logit_tegar_k2": "tegar",
            "zs_logit_tegar_k8": "tegar",
        }[mode]
        return ProbabilitySpaceRefiner(
            num_labels=num_labels,
            kg_path=kg_path,
            text_anchors_path=text_anchors_path,
            clip_model_name=clip_model_name,
            clip_pretrained_path=clip_pretrained_path,
            clip_default_pretrained=clip_default_pretrained,
            graph_type=graph_type,
            hidden_dim=logit_refiner_hidden_dim,
            num_layers=tegar_layers,
            dropout=dropout,
            exclusion_beta_init=exclusion_beta_init,
            use_cached_features=use_cached_features,
            fixed_gate=(mode == "zs_logit_tegar_no_gate"),
            learnable_temperature=(mode != "zs_logit_tegar_fixed_temp"),
            min_uncertainty_gate=(1.0 if mode == "zs_logit_tegar_no_uncertainty_gate" else 0.10),
            scene_expert_count={
                "zs_logit_tegar_no_experts": 1,
                "zs_logit_tegar_k2": 2,
                "zs_logit_tegar_k8": 8,
            }.get(mode, 4),
            use_relation_consensus=(mode != "zs_logit_tegar_no_consensus"),
            negative_exclusion=(mode != "zs_logit_tegar_no_exclusion_neg"),
        )
    if mode == "zs_clip_legacy":
        return ZeroShotCLIPBaseline(
            num_labels=num_labels,
            text_anchors_path=text_anchors_path,
            clip_model_name=clip_model_name,
            clip_pretrained_path=clip_pretrained_path,
            clip_default_pretrained=clip_default_pretrained,
        )
    if mode == "zs_tegar_v3":
        return GraphCrossAttentionClassifier(
            num_labels=num_labels,
            kg_path=kg_path,
            text_anchors_path=text_anchors_path,
            clip_model_name=clip_model_name,
            clip_pretrained_path=clip_pretrained_path,
            clip_default_pretrained=clip_default_pretrained,
            graph_type="tegar",
            tegar_layers=tegar_layers,
            dropout=dropout,
            exclusion_beta_init=exclusion_beta_init,
            cross_attn_heads=cross_attn_heads,
            use_patches=True,
        )
    if mode == "zs_homo_v3":
        return GraphCrossAttentionClassifier(
            num_labels=num_labels,
            kg_path=kg_path,
            text_anchors_path=text_anchors_path,
            clip_model_name=clip_model_name,
            clip_pretrained_path=clip_pretrained_path,
            clip_default_pretrained=clip_default_pretrained,
            graph_type="homo_gat",
            tegar_layers=tegar_layers,
            dropout=dropout,
            exclusion_beta_init=exclusion_beta_init,
            cross_attn_heads=cross_attn_heads,
            use_patches=True,
        )
    if mode == "zs_tegar_no_patch":
        return GraphCrossAttentionClassifier(
            num_labels=num_labels,
            kg_path=kg_path,
            text_anchors_path=text_anchors_path,
            clip_model_name=clip_model_name,
            clip_pretrained_path=clip_pretrained_path,
            clip_default_pretrained=clip_default_pretrained,
            graph_type="tegar",
            tegar_layers=tegar_layers,
            dropout=dropout,
            exclusion_beta_init=exclusion_beta_init,
            cross_attn_heads=cross_attn_heads,
            use_patches=False,
        )
    if mode in {"zs_tegar_delta", "zs_tegar"}:
        return ResidualGraphClassifier(
            num_labels=num_labels,
            kg_path=kg_path,
            text_anchors_path=text_anchors_path,
            clip_model_name=clip_model_name,
            clip_pretrained_path=clip_pretrained_path,
            clip_default_pretrained=clip_default_pretrained,
            graph_type="tegar",
            graph_dim=tegar_dim,
            graph_layers=tegar_layers,
            exclusion_beta_init=exclusion_beta_init,
            dropout=dropout,
            delta_scale_init=delta_scale_init,
            use_explicit_propagation=use_explicit_propagation,
        )
    if mode == "zs_tegar_no_prop":
        return ResidualGraphClassifier(
            num_labels=num_labels,
            kg_path=kg_path,
            text_anchors_path=text_anchors_path,
            clip_model_name=clip_model_name,
            clip_pretrained_path=clip_pretrained_path,
            clip_default_pretrained=clip_default_pretrained,
            graph_type="tegar",
            graph_dim=tegar_dim,
            graph_layers=tegar_layers,
            exclusion_beta_init=exclusion_beta_init,
            dropout=dropout,
            delta_scale_init=delta_scale_init,
            use_explicit_propagation=False,
        )
    if mode in {"zs_homo_delta", "zs_homo_gat"}:
        return ResidualGraphClassifier(
            num_labels=num_labels,
            kg_path=kg_path,
            text_anchors_path=text_anchors_path,
            clip_model_name=clip_model_name,
            clip_pretrained_path=clip_pretrained_path,
            clip_default_pretrained=clip_default_pretrained,
            graph_type="homo_gat",
            graph_dim=tegar_dim,
            graph_layers=tegar_layers,
            exclusion_beta_init=exclusion_beta_init,
            dropout=dropout,
            delta_scale_init=delta_scale_init,
            use_explicit_propagation=use_explicit_propagation,
        )
    raise ValueError(f"Unsupported mode: {mode}")
