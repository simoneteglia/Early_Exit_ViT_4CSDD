"""Early-exit Vision Transformer built on a pretrained timm backbone.

Mirrors ETA-DyNN's ``Joint_EE_MobileNetV3``: side exits attached after
intermediate encoder blocks, trained jointly with a weighted multi-exit loss,
plus an inference routine that applies per-exit calibrators and
lower/upper confidence thresholds (score < lower -> real, score > upper ->
fake, otherwise continue to the next block).

Supported backbones: any block-structured timm ViT — the classic
``VisionTransformer`` family (DeiT, ViT) and the ``Eva`` family used for
DINOv3 (CLS + register tokens, rotary position embeddings).

Score convention: a single logit per sample, ``sigmoid(logit)`` = P(fake).
"""
from __future__ import annotations

import numpy as np
import timm
import torch
from torch import nn

DEFAULT_BACKBONE = "vit_small_patch16_dinov3.lvd1689m"


def default_exit_points(depth: int, n_exits: int = 2) -> tuple[int, ...]:
    """Block indices (0-based; the exit is placed *after* that block) evenly
    spaced in depth: depth=12 -> (3, 7), depth=6 -> (1, 3)."""
    step = depth / (n_exits + 1)
    return tuple(int(round(step * (i + 1))) - 1 for i in range(n_exits))


class ExitHead(nn.Module):
    """LayerNorm + Linear on a pooled token representation.

    pool='cls'     : CLS token (DeiT-style).
    pool='cls_avg' : concat(CLS, mean of patch tokens) — the DINO linear-probe
                     protocol; prefix tokens (CLS + registers) are excluded
                     from the average.
    """

    def __init__(self, dim: int, num_outputs: int = 1, pool: str = "cls", num_prefix_tokens: int = 1):
        super().__init__()
        if pool not in ("cls", "cls_avg"):
            raise ValueError(f"unknown exit pool '{pool}'")
        self.pool = pool
        self.num_prefix_tokens = num_prefix_tokens
        in_dim = dim * (2 if pool == "cls_avg" else 1)
        self.norm = nn.LayerNorm(in_dim)
        self.fc = nn.Linear(in_dim, num_outputs)

    def forward(self, tokens):
        feats = tokens[:, 0]
        if self.pool == "cls_avg":
            feats = torch.cat([feats, tokens[:, self.num_prefix_tokens:].mean(dim=1)], dim=-1)
        return self.fc(self.norm(feats))


class EarlyExitViT(nn.Module):
    def __init__(self,
                 backbone: str = DEFAULT_BACKBONE,
                 pretrained: bool = True,
                 exit_points: tuple[int, ...] | None = None,
                 num_outputs: int = 1,
                 disable_ee: bool = False,
                 drop_path_rate: float = 0.1,
                 exit_pool: str = "auto"):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained,
                                          num_classes=0, drop_path_rate=drop_path_rate)
        if not hasattr(self.backbone, "blocks"):
            raise ValueError(f"{backbone} is not a block-structured ViT (no .blocks)")
        self.depth = len(self.backbone.blocks)

        if exit_points is None:
            exit_points = default_exit_points(self.depth)
        exit_points = tuple(int(p) for p in exit_points)
        if list(exit_points) != sorted(set(exit_points)):
            raise ValueError(f"exit_points must be strictly increasing, got {exit_points}")
        if exit_points and not (0 <= exit_points[0] and exit_points[-1] < self.depth - 1):
            raise ValueError(f"exit_points must lie in [0, {self.depth - 2}], got {exit_points}")
        self.exit_points = exit_points

        # Exits pool like the backbone's own head: DINO-style backbones
        # (global_pool='avg') get CLS+avg, token-pooled ones (DeiT) get CLS.
        if exit_pool == "auto":
            exit_pool = "cls_avg" if getattr(self.backbone, "global_pool", "token") == "avg" else "cls"
        dim = self.backbone.num_features
        n_prefix = getattr(self.backbone, "num_prefix_tokens", 1)
        self.exits = nn.ModuleList(ExitHead(dim, num_outputs, exit_pool, n_prefix) for _ in exit_points)
        self.head = nn.Linear(dim, num_outputs)
        self.disable_ee = disable_ee

        cfg = getattr(self.backbone, "pretrained_cfg", {}) or {}
        self.input_size = tuple(cfg.get("input_size", (3, 224, 224)))
        self.config = {
            "backbone": backbone,
            "exit_points": list(exit_points),
            "num_outputs": num_outputs,
            "drop_path_rate": drop_path_rate,
            "exit_pool": exit_pool,
        }

    # Number of places an answer can come from: side exits + final head.
    @property
    def n_exits(self) -> int:
        return len(self.exit_points) + 1

    @property
    def final_exit_idx(self) -> int:
        return len(self.exit_points)

    # ------------------------------------------------- backbone plumbing --

    def embed(self, x):
        """Patch embedding + prefix tokens + position handling. Returns the
        token tensor and the rotary embedding (None for non-RoPE backbones)."""
        b = self.backbone
        x = b.patch_embed(x)
        pos = b._pos_embed(x)
        x, rope = pos if isinstance(pos, tuple) else (pos, None)
        for name in ("patch_drop", "norm_pre"):  # may be missing or None depending on the family
            layer = getattr(b, name, None)
            if layer is not None:
                x = layer(x)
        return x, rope

    def run_block(self, i, x, rope):
        block = self.backbone.blocks[i]
        if rope is None:
            return block(x)
        if getattr(self.backbone, "rope_mixed", False):
            rope = rope[i]
        return block(x, rope=rope)

    def final_logits(self, x):
        b = self.backbone
        feats = b.forward_head(b.norm(x), pre_logits=True)  # backbone's own pooling + fc_norm
        return self.head(feats)

    # ------------------------------------------------------------ forward --

    def forward(self, x, force_exit: int | None = None):
        """Training / score-extraction forward.

        Returns the list ``[logits_exit_0, ..., logits_final]`` (or just the
        final logits when ``disable_ee``). ``force_exit=e`` stops right after
        side exit ``e`` and returns the logits computed so far; used to
        measure the cumulative cost of reaching each exit.
        """
        x, rope = self.embed(x)
        outputs = []
        for idx in range(self.depth):
            x = self.run_block(idx, x, rope)
            if not self.disable_ee and idx in self.exit_points:
                e = self.exit_points.index(idx)
                outputs.append(self.exits[e](x))
                if force_exit is not None and force_exit == e:
                    return outputs
        logits = self.final_logits(x)
        if self.disable_ee:
            return logits
        outputs.append(logits)
        return outputs

    @staticmethod
    def _to_scores(logits, calibrator=None, device=None):
        p = torch.sigmoid(logits.reshape(-1))
        if calibrator is not None:
            p = torch.as_tensor(np.asarray(calibrator.predict(p.detach().cpu().numpy()), dtype=np.float32),
                                device=device)
        return p

    @torch.no_grad()
    def infer(self, x, lower, upper, calibrators=None):
        """Per-sample early-exit inference with actual compute saving: samples
        that receive an answer are dropped from the batch before the next block.

        lower/upper: thresholds for the side exits, shape (n_side,) or
        (n_side, B) so they can differ per sample (content sensitivity).
        calibrators: optional list of per-exit objects with ``.predict(1-D)``,
        length n_exits (side exits + final).
        The final head always answers (score > 0.5 -> fake).

        Returns a dict of tensors on CPU: ``answers`` (B,) in {0,1},
        ``exit_idx`` (B,), ``scores`` (B,) score at the answering exit,
        ``exit_scores`` (n_exits, B) with NaN where an exit was not computed.
        """
        device = x.device
        B = x.shape[0]
        n_side = len(self.exit_points)
        lower = torch.as_tensor(np.asarray(lower, dtype=np.float32), device=device)
        upper = torch.as_tensor(np.asarray(upper, dtype=np.float32), device=device)
        if lower.ndim == 1:
            lower = lower[:, None].expand(n_side, B)
        if upper.ndim == 1:
            upper = upper[:, None].expand(n_side, B)
        calibrators = calibrators or [None] * self.n_exits

        answers = torch.full((B,), -1, dtype=torch.long, device=device)
        exit_idx = torch.full((B,), -1, dtype=torch.long, device=device)
        scores = torch.full((B,), float("nan"), device=device)
        exit_scores = torch.full((self.n_exits, B), float("nan"), device=device)
        active = torch.arange(B, device=device)

        x, rope = self.embed(x)
        if torch.is_tensor(rope) and rope.ndim >= 3 and rope.shape[0] == B:
            raise NotImplementedError("per-sample rotary embeddings are not supported by infer()")
        for idx in range(self.depth):
            x = self.run_block(idx, x, rope)
            if idx in self.exit_points:
                e = self.exit_points.index(idx)
                s = self._to_scores(self.exits[e](x), calibrators[e], device)
                exit_scores[e, active] = s
                is_real = s < lower[e, active]
                is_fake = s > upper[e, active]
                done = is_real | is_fake
                if done.any():
                    done_idx = active[done]
                    answers[done_idx] = is_fake[done].long()
                    exit_idx[done_idx] = e
                    scores[done_idx] = s[done]
                    x, active = x[~done], active[~done]
                    if active.numel() == 0:
                        break
        if active.numel() > 0:
            s = self._to_scores(self.final_logits(x), calibrators[-1], device)
            exit_scores[-1, active] = s
            answers[active] = (s > 0.5).long()
            exit_idx[active] = n_side
            scores[active] = s

        return {k: v.cpu() for k, v in dict(answers=answers, exit_idx=exit_idx,
                                             scores=scores, exit_scores=exit_scores).items()}


_criterion = nn.BCEWithLogitsLoss()


def multi_exit_loss(outputs, targets, weights=None):
    """Weighted sum of the BCE loss at every exit (equal weights by default)."""
    if weights is None:
        weights = [1.0 / len(outputs)] * len(outputs)
    total = 0.0
    for out, w in zip(outputs, weights):
        total = total + w * _criterion(out.reshape(-1), targets.float().reshape(-1))
    return total


def save_checkpoint(model: EarlyExitViT, path, **extra):
    torch.save({"config": model.config, "state_dict": model.state_dict(), **extra}, path)


def load_model(path, device="cpu", disable_ee: bool = False) -> EarlyExitViT:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = EarlyExitViT(pretrained=False, disable_ee=disable_ee, **ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval()
