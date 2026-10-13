from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.kg_builder import KGDefinition


EMPIRICAL_RELATION_NAMES = (
    "often_cooccur",
    "statistical_exclusion",
    "hierarchical",
)
SEMANTIC_RELATION_NAME = "semantic_affinity"
EXCLUSION_RELATION_NAME = "statistical_exclusion"
VALID_EXCLUSION_MODES = {"negative", "positive", "disabled"}
VALID_GATE_CONTROLS = {"learned", "fixed", "shuffled"}


def _resolve_exclusion_mode(
    exclusion_mode: str | None,
    negative_exclusion: bool,
) -> str:
    """Resolve the explicit exclusion ablation while preserving the legacy flag.

    ``negative_exclusion=False`` historically selected the positive-control
    branch. New code should pass ``exclusion_mode`` explicitly.
    """

    mode = exclusion_mode or ("negative" if negative_exclusion else "positive")
    if mode not in VALID_EXCLUSION_MODES:
        choices = ", ".join(sorted(VALID_EXCLUSION_MODES))
        raise ValueError(f"Unsupported exclusion_mode={mode!r}; expected one of: {choices}")
    return mode


def _resolve_gate_control(gate_control: str | None, fixed_gate: bool) -> str:
    """Resolve strict gate controls without breaking the legacy fixed flag."""

    control = gate_control or ("fixed" if fixed_gate else "learned")
    if control not in VALID_GATE_CONTROLS:
        choices = ", ".join(sorted(VALID_GATE_CONTROLS))
        raise ValueError(f"Unsupported gate_control={control!r}; expected one of: {choices}")
    if fixed_gate and control != "fixed":
        raise ValueError("fixed_gate=True conflicts with a non-fixed gate_control")
    return control


def _stable_gate_permutation(
    relation_names: Sequence[str],
    disabled_relations: frozenset[str],
    seed: int,
) -> tuple[int, ...]:
    """Return a non-identity active-relation permutation using no data statistics.

    The mapping is destination-index -> source-index. Disabled destinations map
    to themselves and remain exactly zero. Hash ordering is stable across
    Python/NumPy/PyTorch RNG implementations and depends only on the declared
    relation types and an explicit seed.
    """

    active = [
        index
        for index, relation_name in enumerate(relation_names)
        if relation_name not in disabled_relations
    ]
    if len(active) < 2:
        raise ValueError("gate_control='shuffled' requires at least two active relations")
    shuffled = sorted(
        active,
        key=lambda index: hashlib.sha256(
            f"{int(seed)}\0{relation_names[index]}".encode("utf-8")
        ).digest(),
    )
    if shuffled == active:
        # A shuffled control must not silently collapse to the reference.
        offset = abs(int(seed)) % (len(active) - 1) + 1
        shuffled = shuffled[offset:] + shuffled[:offset]
    permutation = list(range(len(relation_names)))
    for destination, source in zip(active, shuffled):
        permutation[destination] = source
    return tuple(permutation)


def count_trainable_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def parameter_count_report(reference: nn.Module, control: nn.Module) -> dict[str, float | int]:
    """Return an auditable trainable-parameter comparison for graph controls."""

    reference_count = count_trainable_parameters(reference)
    control_count = count_trainable_parameters(control)
    difference = control_count - reference_count
    return {
        "reference": reference_count,
        "control": control_count,
        "difference": difference,
        "relative_difference": difference / max(reference_count, 1),
    }


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
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"exclusion_beta_init must be finite and > 0; got {value!r}")
    value_tensor = torch.tensor(float(value), dtype=torch.float32)
    return float(torch.log(torch.exp(value_tensor) - 1.0))


def _positive_beta(raw: torch.Tensor) -> torch.Tensor:
    # The epsilon protects the strict beta > 0 contract from floating-point
    # underflow for very negative but otherwise finite raw parameters.
    return F.softplus(raw) + torch.finfo(raw.dtype).eps


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
        exclusion_mode: str | None = None,
        relation_names: Sequence[str] = EMPIRICAL_RELATION_NAMES,
        legacy_latent_exclusion: bool = False,
        gate_initial_activation: float = 0.5,
        disabled_relations: Sequence[str] = (),
        gate_control: str | None = None,
        gate_shuffle_seed: int = 20260826,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gate_control = _resolve_gate_control(gate_control, fixed_gate)
        self.fixed_gate = self.gate_control == "fixed"
        self._legacy_uniform_fixed_gate = bool(fixed_gate) and gate_control is None
        self.gate_shuffle_seed = int(gate_shuffle_seed)
        self.exclusion_mode = _resolve_exclusion_mode(exclusion_mode, negative_exclusion)
        # Compatibility attribute. It describes only the legacy hidden-state
        # sign; strict score monotonicity is carried by the score-domain aux.
        self.negative_exclusion = self.exclusion_mode == "negative"
        self.legacy_latent_exclusion = bool(legacy_latent_exclusion)
        self.gate_initial_activation = float(gate_initial_activation)
        if not math.isfinite(self.gate_initial_activation) or not (
            0.0 < self.gate_initial_activation < 1.0
        ):
            raise ValueError(
                "gate_initial_activation must be finite and in (0, 1); "
                f"got {gate_initial_activation!r}"
            )
        self.relation_names = tuple(dict.fromkeys(relation_names))
        if not self.relation_names:
            raise ValueError("relation_names must contain at least one relation")
        if EXCLUSION_RELATION_NAME not in self.relation_names:
            raise ValueError(f"relation_names must contain {EXCLUSION_RELATION_NAME!r}")
        self.disabled_relations = frozenset(disabled_relations)
        unknown_disabled = self.disabled_relations.difference(self.relation_names)
        if unknown_disabled:
            raise ValueError(f"Cannot disable unknown relations: {sorted(unknown_disabled)}")
        if len(self.disabled_relations) == len(self.relation_names):
            raise ValueError("At least one relation must remain active")
        # This is deliberately a plain immutable tuple rather than a buffer:
        # it changes no parameter/state-dict keys, while multiplying the
        # sigmoid output by an exact zero makes the corresponding shared-gate
        # weight and bias rows receive exact-zero gradients.
        self._relation_gate_enabled = tuple(
            relation_name not in self.disabled_relations
            for relation_name in self.relation_names
        )
        self.gate_permutation = (
            _stable_gate_permutation(
                self.relation_names,
                self.disabled_relations,
                self.gate_shuffle_seed,
            )
            if self.gate_control == "shuffled"
            else tuple(range(len(self.relation_names)))
        )
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
        self.gate = nn.Linear(visual_dim, len(self.relation_names))
        # The hidden-state relation projection above is intentionally
        # unconstrained. This separate head produces score-domain evidence
        # that is nonnegative by construction and must be used for a monotonic
        # exclusion residual.
        self.exclusion_evidence_proj = nn.Linear(hidden_dim, 1)
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
        gate_bias = math.log(
            self.gate_initial_activation / (1.0 - self.gate_initial_activation)
        )
        nn.init.constant_(self.gate.bias, gate_bias)
        nn.init.xavier_uniform_(self.exclusion_evidence_proj.weight, gain=0.1)
        nn.init.zeros_(self.exclusion_evidence_proj.bias)

    def _relation_message(
        self,
        hidden: torch.Tensor,
        adjacency: torch.Tensor,
        relation_name: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if torch.count_nonzero(adjacency) == 0:
            empty_attention = hidden.new_zeros(
                (hidden.size(0), hidden.size(1), hidden.size(1))
            )
            return hidden.new_zeros(hidden.shape), empty_attention

        projected = self.relation_projections[relation_name](hidden)
        src_scores = torch.einsum("bnd,d->bn", projected, self.attn_src[relation_name])
        dst_scores = torch.einsum("bnd,d->bn", projected, self.attn_dst[relation_name])
        logits = self.leaky_relu(src_scores.unsqueeze(2) + dst_scores.unsqueeze(1))

        mask = adjacency.to(dtype=torch.bool, device=hidden.device).unsqueeze(0)
        attention = _masked_attention_softmax(logits, mask)
        attention = self.dropout(attention)
        return attention @ projected, attention

    def forward(
        self,
        hidden: torch.Tensor,
        visual_features: torch.Tensor,
        adjacency: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        raw_messages: dict[str, torch.Tensor] = {}
        attentions: dict[str, torch.Tensor] = {}
        for relation_name in self.relation_names:
            message, attention = self._relation_message(
                hidden,
                adjacency[relation_name],
                relation_name,
            )
            raw_messages[relation_name] = message
            attentions[relation_name] = attention

        relation_messages = dict(raw_messages)
        beta = _positive_beta(self.beta_raw)
        if self.exclusion_mode == "negative":
            hidden_exclusion_scale = -beta
            score_sign = -1.0
        elif self.exclusion_mode == "positive":
            hidden_exclusion_scale = beta
            score_sign = 1.0
        else:
            hidden_exclusion_scale = beta.new_zeros(())
            score_sign = 0.0
        if self.legacy_latent_exclusion:
            # Explicit reproduction-only path. Because the projected vector is
            # arbitrary-signed, this is never claimed to be monotonic score
            # inhibition.
            relation_messages[EXCLUSION_RELATION_NAME] = (
                hidden_exclusion_scale * raw_messages[EXCLUSION_RELATION_NAME]
            )
        else:
            # Strict modes reserve exclusion for the auditable score-domain
            # contribution below. No arbitrary-signed exclusion state can leak
            # into fused hidden features and later pass through a delta head.
            relation_messages[EXCLUSION_RELATION_NAME] = torch.zeros_like(
                raw_messages[EXCLUSION_RELATION_NAME]
            )
        self_message = self.self_proj(hidden)

        if self.fixed_gate:
            gate_mask = torch.tensor(
                self._relation_gate_enabled,
                device=visual_features.device,
                dtype=hidden.dtype,
            )
            fixed_value = (
                1.0 / sum(self._relation_gate_enabled)
                if self._legacy_uniform_fixed_gate
                else self.gate_initial_activation
            )
            gate_values = (
                fixed_value
                * gate_mask.unsqueeze(0).expand(visual_features.size(0), -1)
            )
        else:
            gate_mask = torch.tensor(
                self._relation_gate_enabled,
                device=visual_features.device,
                dtype=hidden.dtype,
            )
            learned_gate_values = torch.sigmoid(self.gate(visual_features)) * gate_mask
            if self.gate_control == "shuffled":
                permutation = torch.tensor(
                    self.gate_permutation,
                    device=visual_features.device,
                    dtype=torch.long,
                )
                gate_values = learned_gate_values.index_select(1, permutation) * gate_mask
            else:
                gate_values = learned_gate_values
        fused = self_message
        for relation_idx, relation_name in enumerate(self.relation_names):
            fused = fused + gate_values[:, relation_idx].view(-1, 1, 1) * relation_messages[relation_name]

        exclusion_idx = self.relation_names.index(EXCLUSION_RELATION_NAME)
        exclusion_gate = gate_values[:, exclusion_idx : exclusion_idx + 1]
        neighbor_evidence = F.softplus(self.exclusion_evidence_proj(hidden)).squeeze(-1)
        exclusion_evidence = torch.bmm(
            attentions[EXCLUSION_RELATION_NAME],
            neighbor_evidence.unsqueeze(-1),
        ).squeeze(-1)
        exclusion_score_contribution = (
            score_sign * beta * exclusion_gate * exclusion_evidence
        )
        return fused, {
            "gate_values": gate_values,
            "beta": beta.detach(),
            "relation_messages": {**relation_messages, "self": self_message},
            "raw_relation_messages": raw_messages,
            "exclusion_evidence": exclusion_evidence,
            "exclusion_beta": beta,
            "exclusion_gate": exclusion_gate,
            "exclusion_score_contribution": exclusion_score_contribution,
            "exclusion_mode": self.exclusion_mode,
            "legacy_latent_exclusion": self.legacy_latent_exclusion,
            "gate_control": self.gate_control,
            "gate_permutation": self.gate_permutation,
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
        exclusion_mode: str | None = None,
        disabled_relations: Sequence[str] = (),
        legacy_latent_exclusion: bool = False,
        gate_initial_activation: float = 0.5,
        gate_control: str | None = None,
        gate_shuffle_seed: int = 20260826,
    ) -> None:
        super().__init__()
        kg_definition = KGDefinition(kg_path)
        adjacency = kg_definition.get_adjacency_matrices(sparse=False)

        zero_adjacency = torch.zeros(
            (kg_definition.node_count, kg_definition.node_count),
            dtype=torch.float32,
        )
        for relation_name in EMPIRICAL_RELATION_NAMES:
            adjacency.setdefault(relation_name, zero_adjacency.clone())
        ordered_relations = list(EMPIRICAL_RELATION_NAMES)
        if SEMANTIC_RELATION_NAME in adjacency:
            ordered_relations.append(SEMANTIC_RELATION_NAME)
        ordered_relations.extend(
            sorted(name for name in adjacency if name not in ordered_relations)
        )

        self.num_labels = kg_definition.node_count
        self.num_layers = num_layers
        self.relation_names = tuple(ordered_relations)
        self.apply_layernorm = apply_layernorm
        self.apply_activation = apply_activation
        self.use_pairnorm = use_pairnorm
        self.adaptive_adj = adaptive_adj
        self.residual_gating = bool(residual_gating)
        self.gate_control = _resolve_gate_control(gate_control, fixed_gate)
        self.fixed_gate = self.gate_control == "fixed"
        self._legacy_uniform_fixed_gate = bool(fixed_gate) and gate_control is None
        self.gate_shuffle_seed = int(gate_shuffle_seed)
        self.exclusion_mode = _resolve_exclusion_mode(exclusion_mode, negative_exclusion)
        self.negative_exclusion = self.exclusion_mode == "negative"
        self.legacy_latent_exclusion = bool(legacy_latent_exclusion)
        self.gate_initial_activation = float(gate_initial_activation)
        self.disabled_relations = frozenset(disabled_relations)
        unknown_disabled = self.disabled_relations.difference(self.relation_names)
        if unknown_disabled:
            raise ValueError(f"Cannot disable unknown relations: {sorted(unknown_disabled)}")
        if SEMANTIC_RELATION_NAME in self.disabled_relations:
            raise ValueError("semantic_affinity must remain active during relation ablations")
        non_empirical_disabled = self.disabled_relations.difference(EMPIRICAL_RELATION_NAMES)
        if non_empirical_disabled:
            raise ValueError(
                "Only empirical relations may be disabled; got "
                f"{sorted(non_empirical_disabled)}"
            )
        self.layers = nn.ModuleList(
            [
                TEGARLayer(
                    hidden_dim=hidden_dim,
                    visual_dim=visual_dim,
                    dropout=dropout,
                    exclusion_beta_init=exclusion_beta_init,
                    fixed_gate=fixed_gate,
                    negative_exclusion=negative_exclusion,
                    exclusion_mode=self.exclusion_mode,
                    relation_names=self.relation_names,
                    legacy_latent_exclusion=self.legacy_latent_exclusion,
                    gate_initial_activation=gate_initial_activation,
                    disabled_relations=self.disabled_relations,
                    gate_control=(
                        None if self._legacy_uniform_fixed_gate else self.gate_control
                    ),
                    gate_shuffle_seed=self.gate_shuffle_seed,
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

        self._adjacency_buffer_names: dict[str, str] = {}
        buffer_names = {
            "often_cooccur": "adj_often",
            "statistical_exclusion": "adj_exclusion",
            "hierarchical": "adj_hierarchical",
            SEMANTIC_RELATION_NAME: "adj_semantic_affinity",
        }
        for relation_name in self.relation_names:
            buffer_name = buffer_names.get(relation_name, f"adj_{relation_name}")
            self.register_buffer(buffer_name, adjacency[relation_name])
            self._adjacency_buffer_names[relation_name] = buffer_name

        # Adaptive adjacency: learnable residual on top of static graph.
        # Historical parameter names are retained for checkpoint compatibility.
        self._adaptive_parameter_names: dict[str, str] = {}
        if adaptive_adj:
            n = kg_definition.node_count
            residual_names = {
                "often_cooccur": "adj_residual_often",
                "statistical_exclusion": "adj_residual_exclusion",
                "hierarchical": "adj_residual_hier",
                SEMANTIC_RELATION_NAME: "adj_residual_semantic_affinity",
            }
            for relation_name in self.relation_names:
                parameter_name = residual_names.get(
                    relation_name,
                    f"adj_residual_{relation_name}",
                )
                self.register_parameter(parameter_name, nn.Parameter(torch.zeros(n, n)))
                self._adaptive_parameter_names[relation_name] = parameter_name

    def _adjacency(self) -> dict[str, torch.Tensor]:
        output: dict[str, torch.Tensor] = {}
        for relation_name in self.relation_names:
            base = getattr(self, self._adjacency_buffer_names[relation_name])
            value = base
            # Adaptive adjacency perturbs existing edges only. Zero init
            # recovers the original graph instead of turning it dense.
            if self.adaptive_adj:
                residual = getattr(self, self._adaptive_parameter_names[relation_name])
                edge_mask = (base > 0).to(base.dtype)
                value = (base + 0.1 * torch.tanh(residual) * edge_mask).clamp_min(0.0)
            if relation_name in self.disabled_relations:
                value = torch.zeros_like(value)
            output[relation_name] = value
        return output

    def forward(
        self,
        hidden: torch.Tensor,
        visual_features: torch.Tensor,
        return_aux: bool = False,
    ):
        gate_history = []
        beta_history = []
        exclusion_evidence_history = []
        exclusion_gate_history = []
        exclusion_contribution_history = []
        last_relation_messages = None
        last_raw_relation_messages = None
        adjacency = self._adjacency()
        for layer_idx, layer in enumerate(self.layers):
            residual = hidden  # residual connection
            hidden, aux = layer(hidden, visual_features, adjacency)
            gate_history.append(aux["gate_values"])
            beta_history.append(aux["beta"])
            exclusion_evidence_history.append(aux["exclusion_evidence"])
            exclusion_gate_history.append(aux["exclusion_gate"])
            exclusion_contribution_history.append(aux["exclusion_score_contribution"])
            if "relation_messages" in aux:
                last_relation_messages = aux["relation_messages"]
            if "raw_relation_messages" in aux:
                last_raw_relation_messages = aux["raw_relation_messages"]
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
        exclusion_evidence_tensor = torch.stack(exclusion_evidence_history, dim=1)
        exclusion_gate_tensor = torch.stack(exclusion_gate_history, dim=1)
        exclusion_contribution_tensor = torch.stack(exclusion_contribution_history, dim=1)
        aux = {
            "gate_values": gate_tensor,
            "beta": beta_tensor,
            "relation_names": self.relation_names,
            "exclusion_mode": self.exclusion_mode,
            "legacy_latent_exclusion": self.legacy_latent_exclusion,
            "gate_control": self.gate_control,
            "gate_permutation": self.layers[-1].gate_permutation,
            # Final-layer score-domain contract consumed by the logit refiner.
            "exclusion_evidence": exclusion_evidence_tensor[:, -1, :],
            "exclusion_beta": _positive_beta(self.layers[-1].beta_raw),
            "exclusion_gate": exclusion_gate_tensor[:, -1, :],
            "exclusion_score_contribution": exclusion_contribution_tensor[:, -1, :],
            # Histories remain available for diagnostics without changing the
            # final-layer interface above.
            "exclusion_evidence_history": exclusion_evidence_tensor,
            "exclusion_gate_history": exclusion_gate_tensor,
            "exclusion_score_contribution_history": exclusion_contribution_tensor,
        }
        if last_relation_messages is not None:
            aux["relation_messages"] = last_relation_messages
        if last_raw_relation_messages is not None:
            aux["raw_relation_messages"] = last_raw_relation_messages
        return hidden, aux


class ParameterMatchedHomogeneousGAT(TEGAR):
    """Homogeneous graph control with the same trainable structure as TEGAR.

    Every relation expert and scene gate is retained, but every expert receives
    the same union adjacency. Consequently, differences from TEGAR isolate
    typed topology/semantics rather than parameter count. Semantic-affinity
    edges remain in the union even when empirical relations are ablated.
    """

    def __init__(self, *args, exclusion_mode: str | None = None, **kwargs) -> None:
        super().__init__(*args, exclusion_mode=exclusion_mode, **kwargs)

    def _adjacency(self) -> dict[str, torch.Tensor]:
        typed_adjacency = super()._adjacency()
        merged = None
        for relation_name in self.relation_names:
            relation_adjacency = typed_adjacency[relation_name]
            merged = (
                relation_adjacency
                if merged is None
                else torch.maximum(merged, relation_adjacency)
            )
        if merged is None:
            raise RuntimeError("Parameter-matched control requires at least one relation")
        merged = torch.maximum(merged, merged.transpose(-1, -2))
        return {relation_name: merged for relation_name in self.relation_names}


# Concise public name used by experiment configuration and reports.
ScoreGraphPM = ParameterMatchedHomogeneousGAT


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
