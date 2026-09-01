# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Opt-in vLLM hook for forwarding real decode logits to MaKV."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass(frozen=True)
class RuntimeRiskRequestContext:
    """Immutable request data retained by the V2 model runner."""

    prompt_token_ids: tuple[int, ...]
    request_params: Mapping[str, Any]


def submit_makv_runtime_risk(
    kv_connector: Any,
    logits: torch.Tensor | None,
    req_ids: Sequence[str],
    requests: Mapping[str, Any],
    scheduled_tokens: Mapping[str, int] | None = None,
) -> int:
    """Submit real decode logits with an explicit prompt-token mapping.

    The mapping is supplied through request ``kv_transfer_params`` as
    ``lmcache.makv.risk_token_indices``.  A missing, malformed, or exhausted
    mapping is fail-closed; in particular, the decode step is never used as a
    prompt/KV position.  The hook only handles one ordinary (non-speculative)
    logits row per request.
    """
    submit = getattr(kv_connector, "submit_precision_risk", None)
    if not callable(submit) or logits is None:
        return 0
    if logits.ndim != 2 or int(logits.shape[0]) != len(req_ids):
        return 0

    submitted = 0
    for req_index, req_id in enumerate(req_ids):
        request = requests.get(req_id)
        if request is None:
            continue
        prompt_token_ids = getattr(request, "prompt_token_ids", None)
        if not isinstance(prompt_token_ids, list) or not prompt_token_ids:
            continue
        if scheduled_tokens is not None and int(
            scheduled_tokens.get(req_id, 0)
        ) != 1:
            # A multi-token prefill row is not a decode observation.  This
            # avoids racing the first risk report with the producer PUT.
            continue

        computed_tokens = getattr(request, "num_computed_tokens", None)
        if not isinstance(computed_tokens, int):
            continue
        prompt_len = len(prompt_token_ids)
        if computed_tokens < prompt_len:
            continue
        step = computed_tokens - prompt_len
        positions = _parse_risk_positions(
            getattr(request, "kv_transfer_params", None)
        )
        if step < 0 or step >= len(positions):
            continue
        token_index = positions[step]
        if token_index < 0 or token_index >= prompt_len:
            continue
        try:
            if submit(
                req_id,
                logits[req_index],
                step,
                token_index,
                prompt_token_ids,
                getattr(request, "kv_transfer_params", None),
            ):
                submitted += 1
        except Exception:
            # Risk observation is optional.  A connector failure must not
            # turn an otherwise valid model step into a failed request.
            logger.exception(
                "MaKV runtime risk submission failed for request %s", req_id
            )
    return submitted


def _request_risk_positions(request: Any) -> list[int]:
    return _parse_risk_positions(getattr(request, "kv_transfer_params", None))


def _parse_risk_positions(params: Any) -> list[int]:
    if not isinstance(params, Mapping):
        return []
    if not bool(params.get("makv_risk_observer_enabled", True)):
        return []
    raw_positions = params.get("lmcache.makv.risk_token_indices")
    if raw_positions is None:
        raw_positions = params.get("lmcache.makv.risk_token_index")
    if isinstance(raw_positions, bool):
        return []
    if isinstance(raw_positions, int):
        raw_positions = [raw_positions]
    if not isinstance(raw_positions, Sequence) or isinstance(
        raw_positions, (str, bytes, bytearray)
    ):
        return []
    positions: list[int] = []
    for value in raw_positions:
        if isinstance(value, bool) or not isinstance(value, int):
            return []
        positions.append(value)
    return positions


def submit_makv_runtime_risk_v2(
    kv_connector: Any,
    logits: torch.Tensor | None,
    req_ids: Sequence[str],
    scheduled_tokens: Sequence[int],
    computed_tokens: Sequence[int],
    prompt_lens: Sequence[int],
    contexts: Mapping[str, RuntimeRiskRequestContext],
) -> int:
    """Forward one real decode logit row per V2 request.

    V2 does not retain ``Request`` objects in the GPU model runner, so the
    caller supplies immutable prompt/config snapshots and CPU scheduler
    mirrors.  The prompt position is always taken from the explicit mapping;
    ``step`` is only used to select an entry from that mapping.
    """
    submit = getattr(kv_connector, "submit_precision_risk", None)
    if not callable(submit) or logits is None:
        return 0
    if logits.ndim != 2 or int(logits.shape[0]) != len(req_ids):
        return 0
    if len(scheduled_tokens) != len(req_ids):
        return 0
    if len(computed_tokens) != len(req_ids) or len(prompt_lens) != len(req_ids):
        return 0

    submitted = 0
    for req_index, req_id in enumerate(req_ids):
        # Only ordinary one-token decode rows are valid risk observations.
        # In particular, do not report a prefill row before its PUT completes.
        if int(scheduled_tokens[req_index]) != 1:
            continue
        context = contexts.get(req_id)
        if context is None or not context.prompt_token_ids:
            continue
        prompt_len = int(prompt_lens[req_index])
        computed = int(computed_tokens[req_index])
        step = computed - prompt_len
        positions = _parse_risk_positions(context.request_params)
        if prompt_len != len(context.prompt_token_ids):
            continue
        if computed < prompt_len:
            continue
        if step >= len(positions):
            continue
        token_index = positions[step]
        if token_index < 0 or token_index >= prompt_len:
            continue
        try:
            if submit(
                req_id,
                logits[req_index],
                step,
                token_index,
                list(context.prompt_token_ids),
                context.request_params,
            ):
                submitted += 1
        except Exception:
            # Risk observation is optional and must not fail model execution.
            logger.exception(
                "MaKV V2 runtime risk submission failed for request %s", req_id
            )
    return submitted


__all__ = [
    "RuntimeRiskRequestContext",
    "submit_makv_runtime_risk",
    "submit_makv_runtime_risk_v2",
]
