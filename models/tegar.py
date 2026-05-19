from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.kg_builder import KGDefinition


# ---------------------------------------------------------------------------
# PairNorm: prevents over-smoothing in deep GCNs
# Reference: "PairNorm: Tackling Oversmoothing in GNNs" (ICLR 2020)
# ---------------------------------------------------------------------------
class PairNorm(nn.Module):
    """PairNorm layer to prevent over-smoothing in GCNs.

    Centers node features and scales by pairwise distance,
    keeping representations distinguishable across layers.
    """

    def __init__(self, scale: float = 1.0) -> None:
        super().__init__()
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, num_nodes, dim)  or  (num_nodes, dim)
        col_mean = x.mean(dim=-2, keepdim=True)
        x = x - col_mean
        rownorm = (x.pow(2).sum(dim=-1, keepdim=True) + 1e-6).sqrt()
        x = self.scale * x / rownorm
        return x


def _softplus_inverse(value: float) -> float:
    value_tensor = torch.tensor(float(value), dtype=torch.float32)
    return float(torch.log(torch.exp(value_tensor) - 1.0))


def _masked_attention_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_value = torch.finfo(logits.dtype).min
    masked_logits = logits.masked_fill(~mask, mask_value)
    attention = torch.softmax(masked_logits.float(), dim=-1).to(logits.dtype)
    attention = attention * mask.to(dtype=attention.dtype)
    attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return attention


class TEGARLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        visual_dim: int,
        dropout: float = 0.1,
        exclusion_beta_init: float = 0.01,
        fixed_gate: bool = False,
        negative_exclusion: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fixed_gate = bool(fixed_gate)
        self.negative_exclusion = bool(negative_exclusion)
        self.relation_names = ("often_cooccur", "statistical_exclusion", "hierarchical")
        self.relation_projections = nn.ModuleDict(
            {name: nn.Linear(hidden_dim, hidden_dim, bias=False) for name in self.relation_names}
        )
        self.attn_src = nn.ParameterDict(
            {name: nn.Parameter(torch.empty(hidden_dim)) for name in self.relation_names}
        )
        self.attn_dst = nn.ParameterDict(
            {name: nn.Parameter(torch.empty(hidden_dim)) for name in self.relation_names}
        )
        self.self_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.gate = nn.Linear(visual_dim, 3)
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.beta_raw = nn.Parameter(torch.tensor(_softplus_inverse(exclusion_beta_init), dtype=torch.float32))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for layer in self.relation_projections.values():
            nn.init.xavier_uniform_(layer.weight, gain=0.1)
        for param in list(self.attn_src.values()) + list(self.attn_dst.values()):
            nn.init.normal_(param, mean=0.0, std=0.01)
        nn.init.eye_(self.self_proj.weight)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def _relation_message(
        self,
        hidden: torch.Tensor,
        adjacency: torch.Tensor,
        relation_name: str,
    ) -> torch.Tensor:
        if adjacency.sum() == 0:
            return hidden.new_zeros(hidden.shape)

        projected = self.relation_projections[relation_name](hidden)
        src_scores = torch.einsum("bnd,d->bn", projected, self.attn_src[relation_name])
        dst_scores = torch.einsum("bnd,d->bn", projected, self.attn_dst[relation_name])
        logits = self.leaky_relu(src_scores.unsqueeze(2) + dst_scores.unsqueeze(1))

        mask = adjacency.to(dtype=torch.bool, device=hidden.device).unsqueeze(0)
        attention = _masked_attention_softmax(logits, mask)
        attention = self.dropout(attention)
        return attention @ projected

    def forward(
        self,
        hidden: torch.Tensor,
        visual_features: torch.Tensor,
        adjacency: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        often_message = self._relation_message(hidden, adjacency["often_cooccur"], "often_cooccur")
        hier_message = self._relation_message(hidden, adjacency["hierarchical"], "hierarchical")
        exclusion_message = self._relation_message(
            hidden,
            adjacency["statistical_exclusion"],
            "statistical_exclusion",
        )

        beta = F.softplus(self.beta_raw)
        exclusion_scale = -beta if self.negative_exclusion else beta
        exclusion_message = exclusion_scale * exclusion_message
        self_message = self.self_proj(hidden)

        if self.fixed_gate:
            gate_values = torch.full(
                (visual_features.size(0), 3),
                1.0 / 3.0,
                device=visual_features.device,
                dtype=hidden.dtype,
            )
        else:
            gate_values = torch.sigmoid(self.gate(visual_features))
        fused = (
            gate_values[:, 0].view(-1, 1, 1) * often_message
            + gate_values[:, 1].view(-1, 1, 1) * exclusion_message
            + gate_values[:, 2].view(-1, 1, 1) * hier_message
            + self_message
        )
        return fused, {
            "gate_values": gate_values,
            "beta": beta.detach(),
            "relation_messages": {
                "often_cooccur": often_message,
                "statistical_exclusion": exclusion_message,
                "hierarchical": hier_message,
                "self": self_message,
            },
        }


class TEGAR(nn.Module):
    def __init__(
        self,
        kg_path: str,
        hidden_dim: int = 512,
        visual_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.1,
        exclusion_beta_init: float = 0.01,
        apply_layernorm: bool = True,
        apply_activation: bool = True,
        use_pairnorm: bool = True,
        adaptive_adj: bool = False,
        residual_gating: bool = False,
        fixed_gate: bool = False,
        negative_exclusion: bool = True,
    ) -> None:
        super().__init__()
        kg_definition = KGDefinition(kg_path)
        adjacency = kg_definition.get_adjacency_matrices(sparse=False)

        self.num_labels = kg_definition.node_count
        self.num_layers = num_layers
        self.apply_layernorm = apply_layernorm
        self.apply_activation = apply_activation
        self.use_pairnorm = use_pairnorm
        self.adaptive_adj = adaptive_adj
        self.residual_gating = bool(residual_gating)
        self.fixed_gate = bool(fixed_gate)
        self.negative_exclusion = bool(negative_exclusion)
        self.layers = nn.ModuleList(
            [
                TEGARLayer(
                    hidden_dim=hidden_dim,
                    visual_dim=visual_dim,
                    dropout=dropout,
                    exclusion_beta_init=exclusion_beta_init,
                    fixed_gate=fixed_gate,
                    negative_exclusion=negative_exclusion,
                )
                for _ in range(num_layers)
            ]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.activation = nn.ReLU(inplace=True)
        if self.residual_gating:
            self.residual_gate = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))

        # PairNorm: prevents over-smoothing, enabling deeper GCNs
        if use_pairnorm:
            self.pairnorms = nn.ModuleList([PairNorm(scale=1.0) for _ in range(num_layers)])

        # Adaptive adjacency: learnable residual on top of static graph
        if adaptive_adj:
            n = kg_definition.node_count
            self.adj_residual_often = nn.Parameter(torch.zeros(n, n))
            self.adj_residual_exclusion = nn.Parameter(torch.zeros(n, n))
            self.adj_residual_hier = nn.Parameter(torch.zeros(n, n))

        self.register_buffer("adj_often", adjacency["often_cooccur"])
        self.register_buffer("adj_exclusion", adjacency["statistical_exclusion"])
        self.register_buffer("adj_hierarchical", adjacency["hierarchical"])

    def _adjacency(self) -> dict[str, torch.Tensor]:
        adj_often = self.adj_often
        adj_exclusion = self.adj_exclusion
        adj_hier = self.adj_hierarchical

        # Adaptive adjacency perturbs existing edges only. Zero init should recover
        # the original KG exactly instead of turning the graph dense.
        if self.adaptive_adj:
            often_mask = (self.adj_often > 0).to(adj_often.dtype)
            exclusion_mask = (self.adj_exclusion > 0).to(adj_exclusion.dtype)
            hier_mask = (self.adj_hierarchical > 0).to(adj_hier.dtype)
            adj_often = (adj_often + 0.1 * torch.tanh(self.adj_residual_often) * often_mask).clamp_min(0.0)
            adj_exclusion = (adj_exclusion + 0.1 * torch.tanh(self.adj_residual_exclusion) * exclusion_mask).clamp_min(0.0)
            adj_hier = (adj_hier + 0.1 * torch.tanh(self.adj_residual_hier) * hier_mask).clamp_min(0.0)

        return {
            "often_cooccur": adj_often,
            "statistical_exclusion": adj_exclusion,
            "hierarchical": adj_hier,
        }

    def forward(
        self,
        hidden: torch.Tensor,
        visual_features: torch.Tensor,
        return_aux: bool = False,
    ):
        gate_history = []
        beta_history = []
        last_relation_messages = None
        adjacency = self._adjacency()
        for layer_idx, layer in enumerate(self.layers):
            residual = hidden  # residual connection
            hidden, aux = layer(hidden, visual_features, adjacency)
            gate_history.append(aux["gate_values"])
            beta_history.append(aux["beta"])
            if "relation_messages" in aux:
                last_relation_messages = aux["relation_messages"]
            if self.apply_layernorm:
                hidden = self.norms[layer_idx](hidden)
            # PairNorm after LayerNorm to prevent over-smoothing
            if self.use_pairnorm:
                hidden = self.pairnorms[layer_idx](hidden)
            if self.apply_activation and layer_idx < self.num_layers - 1:
                hidden = self.activation(hidden)
            if self.residual_gating:
                alpha = torch.sigmoid(self.residual_gate)
                hidden = (1.0 - alpha) * hidden + alpha * residual
            else:
                hidden = hidden + residual

        if not return_aux:
            return hidden

        gate_tensor = torch.stack(gate_history, dim=1)
        beta_tensor = torch.stack(beta_history)
        aux = {"gate_values": gate_tensor, "beta": beta_tensor}
        if last_relation_messages is not None:
            aux["relation_messages"] = last_relation_messages
        return hidden, aux


class HomogeneousGATLayer(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.attn_src = nn.Parameter(torch.empty(hidden_dim))
        self.attn_dst = nn.Parameter(torch.empty(hidden_dim))
        self.self_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.proj.weight, gain=0.1)
        nn.init.normal_(self.attn_src, mean=0.0, std=0.01)
        nn.init.normal_(self.attn_dst, mean=0.0, std=0.01)
        nn.init.eye_(self.self_proj.weight)

    def forward(self, hidden: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        if adjacency.sum() == 0:
            return self.self_proj(hidden)

        projected = self.proj(hidden)
        src_scores = torch.einsum("bnd,d->bn", projected, self.attn_src)
        dst_scores = torch.einsum("bnd,d->bn", projected, self.attn_dst)
        logits = self.leaky_relu(src_scores.unsqueeze(2) + dst_scores.unsqueeze(1))
        mask = adjacency.to(dtype=torch.bool, device=hidden.device).unsqueeze(0)
        attention = _masked_attention_softmax(logits, mask)
        attention = self.dropout(attention)
        return attention @ projected + self.self_proj(hidden)


class HomogeneousGAT(nn.Module):
    def __init__(
        self,
        kg_path: str,
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.1,
        apply_layernorm: bool = True,
        apply_activation: bool = True,
        use_pairnorm: bool = True,
        residual_gating: bool = False,
    ) -> None:
        super().__init__()
        kg_definition = KGDefinition(kg_path)
        adjacency = kg_definition.get_homogeneous_adjacency(sparse=False)
        self.apply_layernorm = apply_layernorm
        self.apply_activation = apply_activation
        self.use_pairnorm = use_pairnorm
        self.residual_gating = bool(residual_gating)
        self.layers = nn.ModuleList([HomogeneousGATLayer(hidden_dim=hidden_dim, dropout=dropout) for _ in range(num_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.activation = nn.ReLU(inplace=True)
        if self.residual_gating:
            self.residual_gate = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
        if use_pairnorm:
            self.pairnorms = nn.ModuleList([PairNorm(scale=1.0) for _ in range(num_layers)])
        self.register_buffer("adjacency", adjacency)

    def forward(self, hidden: torch.Tensor, return_aux: bool = False):
        for layer_idx, layer in enumerate(self.layers):
            residual = hidden
            hidden = layer(hidden, self.adjacency)
            if self.apply_layernorm:
                hidden = self.norms[layer_idx](hidden)
            if self.use_pairnorm:
                hidden = self.pairnorms[layer_idx](hidden)
            if self.apply_activation and layer_idx < len(self.layers) - 1:
                hidden = self.activation(hidden)
            if self.residual_gating:
                alpha = torch.sigmoid(self.residual_gate)
                hidden = (1.0 - alpha) * hidden + alpha * residual
            else:
                hidden = hidden + residual
        if return_aux:
            return hidden, {}
        return hidden
