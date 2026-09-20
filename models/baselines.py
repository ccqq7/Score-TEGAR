from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.clip_backbone import _is_local_checkpoint_spec, _import_open_clip, resolve_clip_pretrained_spec


PROMPT_ENSEMBLE_TEMPLATES: list[str] = [
    "a satellite photo of {}",
    "a remote sensing image of {}",
    "an aerial photo of {}",
    "a satellite image showing {}",
    "a top-down view of {}",
    "an overhead image of {}",
    "a bird's eye view of {}",
    "a high resolution satellite photo of {}",
]


def normalize_label_text(label: str) -> str:
    return str(label).replace("_", " ").strip()


def load_text_anchor_payload(text_anchors_path: str | Path) -> tuple[torch.Tensor, list[str]]:
    payload = torch.load(Path(text_anchors_path).resolve(), map_location="cpu", weights_only=False)
    anchors = payload["anchors"].float()
    label_names = list(payload["label_names"])
    return anchors, label_names


def load_frozen_open_clip_model(
    model_name: str,
    pretrained_path: str | None,
    default_pretrained: str | None,
    device: torch.device,
):
    open_clip = _import_open_clip()
    pretrained_spec = resolve_clip_pretrained_spec(pretrained_path, default_pretrained)
    if _is_local_checkpoint_spec(pretrained_spec):
        checkpoint_path = Path(pretrained_spec).resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"CLIP checkpoint not found: {checkpoint_path}")
        previous_disable = logging.root.manager.disable
        logging.disable(logging.WARNING)
        try:
            model = open_clip.create_model(model_name, pretrained=None)
        finally:
            logging.disable(previous_disable)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        msg = model.load_state_dict(checkpoint, strict=False)
        if getattr(msg, "missing_keys", None) or getattr(msg, "unexpected_keys", None):
            print(f"Loaded CLIP checkpoint with non-strict match: {msg}")
        print(f"Loaded local CLIP checkpoint: {checkpoint_path}")
    else:
        model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained_spec)
    model = model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    tokenizer = open_clip.get_tokenizer(model_name)
    return model, tokenizer


def build_prompt_ensemble_anchors(
    *,
    label_names: list[str],
    model_name: str,
    pretrained_path: str | None,
    default_pretrained: str | None,
    device: torch.device,
    templates: list[str] | None = None,
) -> torch.Tensor:
    templates = templates or PROMPT_ENSEMBLE_TEMPLATES
    clip_model, tokenizer = load_frozen_open_clip_model(
        model_name=model_name,
        pretrained_path=pretrained_path,
        default_pretrained=default_pretrained,
        device=device,
    )
    all_features: list[torch.Tensor] = []
    with torch.no_grad():
        for template in templates:
            texts = [template.format(normalize_label_text(label)) for label in label_names]
            tokenized = tokenizer(texts).to(device)
            text_features = clip_model.encode_text(tokenized)
            text_features = F.normalize(text_features, dim=-1)
            all_features.append(text_features)
    stacked = torch.stack(all_features, dim=0).mean(dim=0)
    return F.normalize(stacked, dim=-1)


class LinearProbe(nn.Module):
    def __init__(self, feat_dim: int = 512, num_labels: int = 58) -> None:
        super().__init__()
        self.head = nn.Linear(feat_dim, num_labels)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(features.float())


class CLIPAdapter(nn.Module):
    def __init__(
        self,
        *,
        text_anchors: torch.Tensor,
        feat_dim: int = 512,
        reduction: int = 4,
        alpha: float = 0.2,
        logit_scale_init: float = math.log(100.0),
    ) -> None:
        super().__init__()
        hidden = max(1, feat_dim // reduction)
        self.adapter = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, feat_dim),
        )
        self.alpha = float(alpha)
        self.logit_scale = nn.Parameter(torch.tensor(float(logit_scale_init), dtype=torch.float32))
        self.register_buffer("text_anchors", F.normalize(text_anchors.float(), dim=-1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = features.float()
        adapted = self.adapter(features)
        mixed = self.alpha * adapted + (1.0 - self.alpha) * features
        mixed = F.normalize(mixed, dim=-1)
        scale = self.logit_scale.exp().clamp(min=0.5, max=100.0)
        return scale * (mixed @ self.text_anchors.T)


class _PromptTuningBase(nn.Module):
    def __init__(
        self,
        *,
        model_name: str,
        pretrained_path: str | None,
        default_pretrained: str | None,
        label_names: list[str],
        n_ctx: int = 16,
        ctx_init: str = "a satellite photo of",
        device: torch.device | None = None,
        logit_scale_init: float | None = None,
    ) -> None:
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.clip_model, self.tokenizer = load_frozen_open_clip_model(
            model_name=model_name,
            pretrained_path=pretrained_path,
            default_pretrained=default_pretrained,
            device=self.device,
        )
        self.label_names = list(label_names)
        self.n_ctx = int(n_ctx)
        self.ctx_dim = int(self.clip_model.token_embedding.weight.shape[1])
        self.context_length = int(self.clip_model.context_length)
        self.text_pool_type = str(getattr(self.clip_model, "text_pool_type", "argmax"))
        self.text_eos_id = getattr(self.clip_model, "text_eos_id", None)
        if logit_scale_init is None:
            initial = getattr(self.clip_model, "logit_scale", None)
            logit_scale_init = float(initial.detach().cpu().item()) if initial is not None else math.log(100.0)
        self.logit_scale = nn.Parameter(torch.tensor(float(logit_scale_init), dtype=torch.float32))
        self._build_prompt_buffers()

        init_ctx = self._initialize_context(ctx_init)
        self.ctx = nn.Parameter(init_ctx.clone())

    def _initialize_context(self, ctx_init: str) -> torch.Tensor:
        tokenized = self.tokenizer([ctx_init]).to(self.device)
        with torch.no_grad():
            init_embedding = self.clip_model.token_embedding(tokenized).detach()[0]
        eos_pos = int(tokenized[0].argmax().item())
        core = init_embedding[1:eos_pos]
        if core.numel() == 0:
            core = self.clip_model.token_embedding.weight[1:2].detach()
        if core.size(0) >= self.n_ctx:
            ctx = core[: self.n_ctx]
        else:
            repeat = math.ceil(self.n_ctx / max(1, core.size(0)))
            ctx = core.repeat(repeat, 1)[: self.n_ctx]
        return ctx.float()

    def _build_prompt_buffers(self) -> None:
        tokenized_name = self.tokenizer([normalize_label_text(label) for label in self.label_names])
        n_cls = tokenized_name.size(0)
        tokenized_prompts = torch.zeros((n_cls, self.context_length), dtype=tokenized_name.dtype)
        tokenized_prompts[:, 0] = tokenized_name[:, 0]
        for idx in range(n_cls):
            row = tokenized_name[idx]
            eos_pos = int(row.argmax().item())
            class_tokens = row[1:eos_pos]
            prompt = tokenized_prompts[idx]
            end = 1 + self.n_ctx
            prompt[end : end + class_tokens.numel()] = class_tokens
            eos_index = min(self.context_length - 1, end + class_tokens.numel())
            prompt[eos_index] = row[eos_pos]
        with torch.no_grad():
            embedding = self.clip_model.token_embedding(tokenized_prompts.to(self.device)).detach()
        self.register_buffer("tokenized_prompts", tokenized_prompts)
        self.register_buffer("token_prefix", embedding[:, :1, :].float())
        self.register_buffer("token_suffix", embedding[:, 1 + self.n_ctx :, :].float())

    def encode_with_context(self, ctx: torch.Tensor) -> torch.Tensor:
        import open_clip.model as open_clip_model  # local import to avoid hard dependency at module load

        cast_dtype = self.clip_model.transformer.get_cast_dtype()
        n_cls = len(self.label_names)
        prompts = torch.cat(
            [
                self.token_prefix.to(self.device).to(cast_dtype),
                ctx.unsqueeze(0).expand(n_cls, -1, -1).to(self.device).to(cast_dtype),
                self.token_suffix.to(self.device).to(cast_dtype),
            ],
            dim=1,
        )
        x = prompts + self.clip_model.positional_embedding.to(cast_dtype)
        x = self.clip_model.transformer(x, attn_mask=self.clip_model.attn_mask)
        x = self.clip_model.ln_final(x)
        x = open_clip_model.text_global_pool(
            x,
            self.tokenized_prompts.to(self.device),
            self.text_pool_type,
            eos_token_id=self.text_eos_id,
        )
        if self.clip_model.text_projection is not None:
            if isinstance(self.clip_model.text_projection, nn.Linear):
                x = self.clip_model.text_projection(x)
            else:
                x = x @ self.clip_model.text_projection
        return F.normalize(x.float(), dim=-1)

    def _compute_scale(self) -> torch.Tensor:
        return self.logit_scale.exp().clamp(min=0.5, max=100.0)


class CoOpPromptLearner(_PromptTuningBase):
    def __init__(
        self,
        *,
        model_name: str,
        pretrained_path: str | None,
        default_pretrained: str | None,
        label_names: list[str],
        n_ctx: int = 16,
        device: torch.device | None = None,
    ) -> None:
        super().__init__(
            model_name=model_name,
            pretrained_path=pretrained_path,
            default_pretrained=default_pretrained,
            label_names=label_names,
            n_ctx=n_ctx,
            ctx_init="a satellite photo of",
            device=device,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        visual = F.normalize(features.float(), dim=-1)
        text = self.encode_with_context(self.ctx)
        scale = self._compute_scale()
        return scale * (visual @ text.T)


class DualCoOp(nn.Module):
    def __init__(
        self,
        *,
        model_name: str,
        pretrained_path: str | None,
        default_pretrained: str | None,
        label_names: list[str],
        n_ctx: int = 16,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.base = _PromptTuningBase(
            model_name=model_name,
            pretrained_path=pretrained_path,
            default_pretrained=default_pretrained,
            label_names=label_names,
            n_ctx=n_ctx,
            ctx_init="a satellite photo of",
            device=device,
        )
        self.ctx_pos = self.base.ctx
        del self.base._parameters["ctx"]
        self.ctx_neg = nn.Parameter(self.base._initialize_context("a satellite photo without").clone())

    @property
    def label_names(self) -> list[str]:
        return self.base.label_names

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        visual = F.normalize(features.float(), dim=-1)
        text_pos = self.base.encode_with_context(self.ctx_pos)
        text_neg = self.base.encode_with_context(self.ctx_neg)
        sim_pos = visual @ text_pos.T
        sim_neg = visual @ text_neg.T
        scale = self.base._compute_scale()
        return scale * (sim_pos - sim_neg)


class MLDecoderZS(nn.Module):
    def __init__(
        self,
        *,
        text_anchors: torch.Tensor,
        feat_dim: int = 512,
        num_heads: int = 8,
        decoder_layers: int = 2,
        hidden_dim: int = 512,
    ) -> None:
        super().__init__()
        self.register_buffer("label_queries", text_anchors.float())
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_layers)
        self.visual_proj = nn.Linear(feat_dim, hidden_dim)
        self.query_proj = nn.Linear(text_anchors.shape[-1], hidden_dim)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, visual_features: torch.Tensor) -> torch.Tensor:
        memory = self.visual_proj(visual_features.float()).unsqueeze(1)
        batch_size = visual_features.size(0)
        queries = self.query_proj(self.label_queries).unsqueeze(0).expand(batch_size, -1, -1)
        decoded = self.decoder(queries, memory)
        return self.fc(decoded).squeeze(-1)


def save_prompt_ensemble_anchors(
    *,
    output_path: str | Path,
    label_names: list[str],
    model_name: str,
    pretrained_path: str | None,
    default_pretrained: str | None,
    device: torch.device,
    templates: list[str] | None = None,
) -> Path:
    output_path = Path(output_path).resolve()
    anchors = build_prompt_ensemble_anchors(
        label_names=label_names,
        model_name=model_name,
        pretrained_path=pretrained_path,
        default_pretrained=default_pretrained,
        device=device,
        templates=templates,
    )
    payload = {
        "anchors": anchors.cpu(),
        "label_names": list(label_names),
        "templates": templates or PROMPT_ENSEMBLE_TEMPLATES,
        "clip_model": model_name,
        "clip_pretrained": pretrained_path,
        "clip_default_pretrained": default_pretrained,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    meta_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    meta_path.write_text(json.dumps({k: v for k, v in payload.items() if k != "anchors"}, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path
