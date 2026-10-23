# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.


"""Pluggable embedding backends for recsys models on recir.

Three backends share the ``EmbeddingBackend`` ABC from ``backend.py``:

  CPUEmbeddingBackend (cpu.py)        - torchrec DMP on CPU via gloo PG.
                                        Small tables -> TABLE_WISE, large
                                        -> ROW_WISE.

  SparseCoreEmbeddingBackend          - jax_tpu_embedding SparseCore: tables
    (sparsecore.py)                     as EmbeddingVariables sharded on
                                        SparseCore HW, lookup via
                                        tpu_sparse_dense_matmul, Adagrad
                                        fused on-chip. No gloo PG required.

  InGraphEmbeddingBackend             - ring-shift embedding lookup traced
    (in_graph.py)                       inside the compiled GSPMD program.

All three are model-agnostic. Each binds the torchrec ``EmbeddingCollection``
the model names by FQN; the model pairs the backend with its own tracing
dispatchers.

``factory.build_embedding_backend`` picks a backend by name and so links all
three. A binary that needs only one should construct it from its own submodule
instead, so it does not link torchrec.distributed.* (heavy) or
jax_tpu_embedding. This package re-exports only the interface.
"""

from __future__ import annotations

from recir.embedding.backend import (
    EmbeddingBackend,
    EmbeddingSetupResult,
    SparseOptimizerRuntime,
    SparseUpdatePlacement,
)

__all__ = [
    "EmbeddingBackend",
    "EmbeddingSetupResult",
    "SparseOptimizerRuntime",
    "SparseUpdatePlacement",
]
