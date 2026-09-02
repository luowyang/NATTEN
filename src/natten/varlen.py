#################################################################################################
# Copyright (c) 2022 - 2026 Ali Hassani.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
#################################################################################################
"""The packed-document layout for variable-length FNA (distinct from
``utils/varlen.py``, the FMHA varlen parameter helpers).
"""

import math
from typing import (
    Any,
    Callable,
    cast,
    Dict,
    Hashable,
    List,
    Optional,
    Sequence,
    Tuple,
    TYPE_CHECKING,
    Union,
)

import torch
from torch import Tensor

from natten.types import DimensionType

if TYPE_CHECKING:
    from natten.backends.varlen_fna import _VarlenFnaResolvedState

_VARLEN_INT32_MAX = 2**31 - 1


def _normalize_token_layouts(
    token_layouts: Sequence[DimensionType],
) -> Tuple[DimensionType, ...]:
    if not isinstance(token_layouts, Sequence) or isinstance(
        token_layouts, (str, bytes)
    ):
        raise TypeError("token_layouts must be a sequence of integer tuples.")
    if len(token_layouts) == 0:
        raise ValueError("token_layouts must contain at least one layout.")

    normalized = []
    for index, layout in enumerate(token_layouts):
        if not isinstance(layout, Sequence) or isinstance(layout, (str, bytes)):
            raise TypeError(f"token_layouts[{index}] must be an integer tuple.")
        normalized_layout = tuple(layout)
        if any(
            isinstance(extent, bool) or not isinstance(extent, int)
            for extent in normalized_layout
        ):
            raise TypeError(f"token_layouts[{index}] must contain only integers.")
        if any(extent < 0 for extent in normalized_layout):
            raise ValueError(
                f"token_layouts[{index}] must contain only non-negative extents."
            )
        if any(extent > _VARLEN_INT32_MAX for extent in normalized_layout):
            raise ValueError(
                f"token_layouts[{index}] extents must not exceed "
                f"{_VARLEN_INT32_MAX} (int32 range), got {normalized_layout}."
            )
        normalized.append(normalized_layout)

    na_dim = len(normalized[0])
    if na_dim not in [1, 2, 3]:
        raise ValueError(f"Only 1-D, 2-D, and 3-D FNA are supported, got {na_dim=}.")
    if any(len(layout) != na_dim for layout in normalized):
        raise ValueError("All token layouts must have the same dimensionality.")

    return cast(Tuple[DimensionType, ...], tuple(normalized))


def _prefix_offsets(counts: Sequence[int]) -> Tuple[int, ...]:
    offsets = [0]
    for count in counts:
        offsets.append(offsets[-1] + count)
    return tuple(offsets)


def _derive_repeated_shapes(
    shapes: Tuple[DimensionType, ...],
    group_axes: Tuple[int, ...],
    keep_axes: Tuple[int, ...],
) -> Tuple[DimensionType, ...]:
    """Shared shape arithmetic behind both VarlenLayout._folded and
    ._permuted: each document's shape becomes ``prod(shape[group_axes])``
    documents of shape ``shape[keep_axes]``. A zero extent on a grouped axis
    contributes zero sub-documents (the parent had zero tokens on that
    account too); a zero extent on a kept axis instead contributes that
    many empty (zero-token) sub-documents -- existing empty-document
    semantics, unaffected either way.
    """
    derived: List[DimensionType] = []
    for shape in shapes:
        # List comprehensions, not generator expressions, into math.prod/
        # tuple: torch.compile(fullgraph=True) can trace this function when
        # `layout` (and so this call) is a plain argument rather than a
        # closure-captured value, and a bare generator handed to math.prod
        # is a known dynamo tracing gap a list is not.
        count = math.prod([shape[axis] for axis in group_axes])
        kept = cast(DimensionType, tuple([shape[axis] for axis in keep_axes]))
        derived.extend([kept] * count)
    return tuple(derived)


@torch.library.custom_op("natten::varlen_build_permutation_tensors", mutates_args=())
def _varlen_build_permutation_tensors(
    shapes_flat: List[int],
    rank: int,
    group_axes: List[int],
    device: torch.device,
) -> Tuple[Tensor, Tensor]:
    """VarlenLayout._permuted's memo-miss device work, behind an opaque
    custom-op boundary for the same reason as varlen_fna.py's
    _varlen_build_schedule_tensors (see that docstring): a geometry miss
    taken at torch.compile trace time records one opaque op call instead of
    inlining the ATen ops below into the graph, so a later memo hit (a
    dict lookup) never re-enters or replays this build.

    Host-only shape arithmetic (rank-many ints per document, flattened, so
    every document's shape is recovered by slicing ``rank`` at a time) plus
    the per-document index construction the class docstring describes:
    ``arange(numel).view(shape).permute(group_axes + keep_axes).reshape(-1)
    + offset``, concatenated across documents into ``perm``, then inverted
    into ``inv`` by one scatter (``inv[perm] = arange(total)``) -- the
    inverse of a permutation is its own scatter-built lookup, not a second
    sort or argsort.
    """
    num_docs = len(shapes_flat) // rank
    shapes = [
        tuple(shapes_flat[index * rank : (index + 1) * rank])
        for index in range(num_docs)
    ]
    keep_axes = tuple(axis for axis in range(rank) if axis not in group_axes)
    order = tuple(group_axes) + keep_axes
    lengths = [math.prod(shape) for shape in shapes]
    offsets = _prefix_offsets(lengths)
    with torch.inference_mode(False):
        parts = []
        for doc_index, shape in enumerate(shapes):
            length = lengths[doc_index]
            if length == 0:
                # A zero-token document (some axis is 0, group or kept)
                # contributes no rows to permute -- see
                # _derive_repeated_shapes's docstring.
                continue
            local = torch.arange(length, dtype=torch.int64, device=device).view(shape)
            parts.append(local.permute(order).reshape(-1) + offsets[doc_index])
        total = offsets[-1]
        perm = (
            torch.cat(parts)
            if parts
            else torch.empty((0,), dtype=torch.int64, device=device)
        )
        inv = torch.empty_like(perm)
        inv[perm] = torch.arange(total, dtype=torch.int64, device=device)
    return perm, inv


@_varlen_build_permutation_tensors.register_fake
def _(
    shapes_flat: List[int],
    rank: int,
    group_axes: List[int],
    device: torch.device,
) -> Tuple[Tensor, Tensor]:
    num_docs = len(shapes_flat) // rank
    total = sum(
        math.prod(shapes_flat[index * rank : (index + 1) * rank])
        for index in range(num_docs)
    )
    return (
        torch.empty((total,), dtype=torch.int64, device=device),
        torch.empty((total,), dtype=torch.int64, device=device),
    )


class VarlenLayout:
    """Packed-document layout for variable-length neighborhood attention.

    Unrelated to ``torch.layout`` (memory format); this describes how many
    documents are packed into one sequence-packed QKV tensor and how many
    tokens each one contributes. It also owns mutable, per-object derived
    schedule state (worklists, KV-split selection, resolved tile configs),
    memoized per neighborhood-attention geometry for the object's lifetime;
    see [backends](backends.md#design-notes) for what is cached, why it
    lives on the layout rather than a separate plan object, and how it is
    keyed.

    Derived metadata and schedule tensors are created on the CUDA stream
    that is current at first use of a geometry (materialization, or the
    first call that resolves a new one); using the same layout from a
    different stream afterwards is the caller's responsibility to order
    (an event wait, or synchronizing the producer stream first), exactly as
    for any tensor produced on another stream.

    [from_tensor_list][natten.VarlenLayout.from_tensor_list] is the
    supported construction path: it builds the layout and packs a list of
    per-document tensors in one call, guaranteeing consistency between the
    returned layout and packed tensor at construction time -- they come
    from the same call::

        layout, query = natten.VarlenLayout.from_tensor_list([doc0_q, doc1_q])

    K and V can pack independently from the same ordered list of documents
    -- their own ``from_tensor_list`` call, since GQA/MQA gives them a
    different head count than Q -- as long as every call shares the same
    document order and per-document token counts, so the resulting layouts
    are interchangeable and only one needs to be kept::

        layout, query = natten.VarlenLayout.from_tensor_list([doc0_q, doc1_q])
        _, key = natten.VarlenLayout.from_tensor_list([doc0_k, doc1_k])
        _, value = natten.VarlenLayout.from_tensor_list([doc0_v, doc1_v])
        # doc{0,1}_k/_v may have fewer heads than doc{0,1}_q (GQA/MQA);
        # `layout` alone is passed to na{1,2,3}d_varlen -- key/value's own
        # (identical) layouts are discarded, not separately tracked.

    **Advanced construction / interoperability.** A layout can also be built
    directly from already-known per-document shapes::

        layout = VarlenLayout([(8, 32, 32), (4, 16, 16)], device="cuda")

    or derived from a rank-1 ``cu_seqlens`` prefix-sum tensor
    (FlashAttention-style convention; 1-D documents only, since higher-rank
    spatial shapes cannot be recovered from a single prefix sum)::

        shapes = [
            (int(cu_seqlens[i + 1] - cu_seqlens[i]),)
            for i in range(cu_seqlens.numel() - 1)
        ]
        layout = VarlenLayout(shapes, device=cu_seqlens.device)

    Both of these validate only metadata-local invariants (shapes are
    well-formed, extents fit int32, and so on). By constructing directly,
    the caller asserts that every packed tensor used with this layout
    preserves the same document order and spatial token layout the shapes
    describe -- something NATTEN cannot infer from the packed tensor alone,
    since a ``VarlenLayout`` carries only per-document extents, never row
    identity or content. Any operation that reorders, repartitions,
    inserts, removes, or independently repacks documents must rebuild the
    layout or re-establish this invariant.

    ``VarlenLayout`` defines the document order and spatial shape of the
    packed token dimension. Every tensor used with the layout must use the
    same document order and layout.

    !!! warning
        Layout metadata tensors (``cu_seqlens``, ``token_layouts``) are
        owned by the ``VarlenLayout`` and must not be modified in place.
        When document shapes, document order, or device placement changes,
        construct a new layout.

    Pickling preserves only the per-document ``shapes``; a materialized
    layout's device tensors and geometry memo do not survive the round
    trip. The unpickled layout is unmaterialized and rebuilds derived state
    on first use, same as a freshly constructed one.

    Parameters:
        shapes (Sequence[DimensionType]): Per-document spatial token layout
            (rank 1, 2, or 3), consistent across all documents. A document's
            extents may be zero (e.g. ``(0,)`` or ``(4, 0, 16)`` -- empty
            along any axis), making it an empty document (see
            [na1d_varlen][natten.na1d_varlen]'s docstring); ``shapes`` itself
            must still list at least one document.

        device (Optional[Union[torch.device, str]]): Device to materialize
            metadata tensors on immediately. ``None`` (default) defers
            materialization to the first call into a ``na{1,2,3}d_varlen``
            entry point, pinning the layout to that call's device
            thereafter; a later call from a different device raises.
            Exception: a call whose layout is all-empty performs no device
            work and does not pin.
    """

    # Deliberately no __eq__/__hash__ override: the per-geometry memo
    # described above is safe only under identity semantics. A value-equal
    # override (dataclass-style or hand-written) would let two distinct
    # objects read and overwrite each other's derived state, and once a
    # Tensor field takes part in the comparison the class is not hashable
    # at all, so it could no longer key anything.

    def __init__(
        self,
        shapes: Sequence[DimensionType],
        device: Optional[Union[torch.device, str]] = None,
    ) -> None:
        self._set_host_properties(_normalize_token_layouts(shapes))
        self._device: Optional[torch.device] = None
        self._memo: Dict[Hashable, "_VarlenFnaResolvedState"] = {}
        self._cu_seqlens: Optional[Tensor] = None
        self._token_layouts: Optional[Tensor] = None
        # Degenerate-axis lowering's derived state (see
        # natten.backends.varlen_lowering): both keyed and owned per-object,
        # same identity-semantics rationale as _memo above. _fold_memo maps
        # a leading-fold length to the derived (lower-rank) VarlenLayout;
        # _permute_memo maps a tuple of degenerate axis indices (in this
        # layout's own rank) to the derived VarlenLayout plus the (perm,
        # inv) index tensors gather/scatter the tokens with. Neither
        # survives pickling (see __getstate__/__setstate__).
        self._fold_memo: Dict[int, "VarlenLayout"] = {}
        self._permute_memo: Dict[
            Tuple[int, ...], Tuple["VarlenLayout", Tensor, Tensor]
        ] = {}
        if device is not None:
            self._materialize(torch.device(device))

    def __repr__(self) -> str:
        return (
            f"VarlenLayout(num_docs={self._num_docs}, rank={self._rank}, "
            f"total_tokens={self._total_tokens}, max_seqlen={self._max_seqlen}, "
            f"device={self._device})"
        )

    def _set_host_properties(self, shapes: Tuple[DimensionType, ...]) -> None:
        num_docs = len(shapes)
        # Empty documents (a zero extent along some axis) let num_docs
        # exceed total_tokens -- e.g. every document all-empty passes the
        # token-count fence below trivially regardless of how many there
        # are -- so document count needs its own int32 fence: worklist
        # document ids are int32 end to end, same as the token-count
        # metadata the fence below protects.
        if num_docs > _VARLEN_INT32_MAX:
            raise ValueError(
                f"Number of documents must not exceed {_VARLEN_INT32_MAX} "
                f"(worklist document ids are int32), got {num_docs=}."
            )
        # List comprehensions, not generator expressions: this constructor
        # runs under torch.compile(fullgraph=True) whenever a degenerate-
        # axis lowering call (natten.backends.varlen_lowering) constructs a
        # derived layout from a directly-passed (not closure-captured)
        # VarlenLayout argument, and a bare generator handed to sum/max/
        # math.prod is a dynamo tracing gap a list is not.
        total_tokens = sum([math.prod(shape) for shape in shapes])
        # The token COUNT must fit int32 because the schedule metadata
        # (cu_seqlens, worklists, document ids) is int32 end to end; element
        # counts (tokens * heads * head_dim) are deliberately unfenced --
        # kernel addressing is int64-safe, matching fixed FNA post-#337.
        if total_tokens > _VARLEN_INT32_MAX:
            raise ValueError(
                "Packed token count must not exceed "
                f"{_VARLEN_INT32_MAX}, got {total_tokens=}."
            )
        self._shapes = shapes
        self._rank = len(shapes[0])
        self._num_docs = num_docs
        self._total_tokens = total_tokens
        self._max_seqlen = max([math.prod(shape) for shape in shapes])
        # Cached once here (rather than recomputed per call): every document
        # sharing the same shape is what lets a call dispatch straight to the
        # fixed-shape kernels on a batched view (see
        # _neighborhood_attention_varlen_generic's uniform-dispatch branch).
        self._is_uniform = all(shape == shapes[0] for shape in shapes)

    # -- device materialization --------------------------------------------

    def _materialize(self, device: torch.device) -> None:
        if device.type != "cuda" or not torch.cuda.is_available():
            raise ValueError("Variable-length CUTLASS FNA requires a CUDA device.")
        lengths = tuple([math.prod(shape) for shape in self._shapes])
        with torch.inference_mode(False):
            cu_seqlens = torch.tensor(
                _prefix_offsets(lengths), dtype=torch.int32, device=device
            )
            token_layouts = torch.tensor(self._shapes, dtype=torch.int32, device=device)
        # An index-less device ("cuda") does not compare equal to a tensor's
        # always-qualified one ("cuda:0"), even for the same physical GPU, so
        # the pin is taken from an actually allocated tensor's `.device`
        # rather than from the caller-supplied device object.
        self._device = cu_seqlens.device
        self._cu_seqlens = cu_seqlens
        self._token_layouts = token_layouts

    def _check_device_pin(self, device: torch.device) -> None:
        if self._device is not None and self._device != device:
            raise ValueError(
                f"VarlenLayout was pinned to device {self._device} on first use "
                f"(construction or first call); got tensors on {device}. Build a "
                "separate VarlenLayout per device."
            )

    def _ensure_materialized(self, device: torch.device) -> None:
        self._check_device_pin(device)
        if self._device is None:
            self._materialize(device)

    # -- memo -----------------------------------------------------------

    def _resolve(
        self, key: Hashable, build: Callable[[], "_VarlenFnaResolvedState"]
    ) -> "_VarlenFnaResolvedState":
        entry = self._memo.get(key)
        if entry is not None:
            return entry

        # No lock: under the GIL, a racing first-touch on the same geometry
        # key at worst builds the schedule twice -- both results are
        # equivalent immutable tensor sets, dict assignment is atomic, and
        # last-write-wins is harmless. Nothing stronger is offered: a
        # VarlenLayout is not otherwise thread-safe.
        #
        # No is_torch_compiling() guard either: build() routes its device
        # tensor construction through a custom op (_varlen_build_schedule_tensors),
        # so a miss traced inside torch.compile records one opaque op call
        # instead of inlining ATen ops into the graph. See that op's
        # docstring for the recompile mechanics.
        entry = build()
        self._memo[key] = entry
        return entry

    # -- degenerate-axis lowering (natten.backends.varlen_lowering) -------

    def _folded(self, f: int) -> "VarlenLayout":
        """Leading-fold derived layout: each document's shape ``s`` becomes
        ``prod(s[:f])`` consecutive documents of shape ``s[f:]``. Pure
        metadata repartitioning -- a document's own axes are already
        row-major, so every leading index's sub-block is already
        contiguous, and no packed tensor needs a view or copy for this.
        Memoized per ``f``: a repeated call with the same ``f`` returns the
        same derived object (so ITS OWN memo/derived state -- schedule,
        further folds/permutes -- is also reused, not rebuilt per call).
        """
        derived = self._fold_memo.get(f)
        if derived is not None:
            return derived
        derived = VarlenLayout(
            _derive_repeated_shapes(
                self._shapes, tuple(range(f)), tuple(range(f, self._rank))
            )
        )
        self._fold_memo[f] = derived
        return derived

    def _permuted(
        self, group_axes: Tuple[int, ...], device: torch.device
    ) -> Tuple["VarlenLayout", Tensor, Tensor]:
        """Non-leading-fold derived layout: each document's shape ``s``
        becomes ``prod(s[group_axes])`` documents of shape ``s[keep_axes]``
        (``keep_axes`` = the complement of ``group_axes``, ascending),
        reordering ``group_axes`` to the front of each document's own token
        block. Unlike [_folded][natten.varlen.VarlenLayout._folded], the
        tokens genuinely move (a gather), so this also returns an int64
        ``perm`` index (forward: ``tokens.index_select(0, perm)``) and its
        inverse ``inv`` (backward, and un-permuting output/lse back to this
        layout's order: ``tokens.index_select(0, inv)``) -- see
        ``natten.backends.varlen_lowering._PermuteTokens``. Memoized per
        ``group_axes``. Pins/materializes THIS layout on ``device`` as a
        side effect (perm/inv are device tensors, built once for this
        layout's lifetime and reused by every later call, same device-pin
        contract as the rest of the class); the derived layout returned
        alongside them is left unmaterialized, pinning lazily like any
        other layout on its first real use.
        """
        # Checked unconditionally, before the memo lookup: a cache hit must
        # not silently skip the device-pin check (_ensure_materialized only
        # runs on a miss), same as every other entry point's unconditional
        # _check_device_pin call.
        self._check_device_pin(device)
        cached = self._permute_memo.get(group_axes)
        if cached is not None:
            return cached
        self._ensure_materialized(device)
        keep_axes = tuple(i for i in range(self._rank) if i not in group_axes)
        derived = VarlenLayout(
            _derive_repeated_shapes(self._shapes, group_axes, keep_axes)
        )
        shapes_flat = [extent for shape in self._shapes for extent in shape]
        perm, inv = _varlen_build_permutation_tensors(
            shapes_flat, self._rank, list(group_axes), device
        )
        result = (derived, perm, inv)
        self._permute_memo[group_axes] = result
        return result

    # -- public read-only properties -----------------------------------
    # cu_seqlens and token_layouts return the underlying tensors as-is,
    # not defensive copies: PyTorch has no read-only tensor view, so a copy
    # is the only way to make one, and the class docstring's warning against
    # in-place modification already forbids the misuse a copy would guard
    # against -- the same contract FlashAttention's caller-owned cu_seqlens
    # uses. Copying here would tax every legitimate interop read (the
    # documented purpose of exposing these at all) to defend against a
    # precondition violation, not prevent one.

    @property
    def cu_seqlens(self) -> Tensor:
        """Derived int32 ``[num_docs + 1]`` device tensor of prefix-summed
        per-document token counts. Not a construction input -- ``shapes`` is
        the only source of truth. Interop convention matches
        [attention][natten.attention]'s ``cumulative_seqlen_Q``/``_KV``."""
        if self._cu_seqlens is None:
            raise RuntimeError(
                "VarlenLayout has not been materialized on a device yet; "
                "construct it with an explicit device=..., or call a "
                "na{1,2,3}d_varlen entry point with it once."
            )
        return self._cu_seqlens

    @property
    def token_layouts(self) -> Tensor:
        """Derived int32 ``[num_docs, rank]`` device tensor of per-document
        spatial extents."""
        if self._token_layouts is None:
            raise RuntimeError(
                "VarlenLayout has not been materialized on a device yet; "
                "construct it with an explicit device=..., or call a "
                "na{1,2,3}d_varlen entry point with it once."
            )
        return self._token_layouts

    @property
    def num_docs(self) -> int:
        """Number of packed documents."""
        return self._num_docs

    @property
    def total_tokens(self) -> int:
        """Sum of per-document token counts; the required packed QKV
        leading-dimension size."""
        return self._total_tokens

    @property
    def max_seqlen(self) -> int:
        """Largest per-document token count."""
        return self._max_seqlen

    @property
    def rank(self) -> int:
        """Spatial rank (1, 2, or 3) shared by every packed document."""
        return self._rank

    @property
    def is_uniform(self) -> bool:
        """Whether every document shares the same spatial shape. Cached at
        construction; a layout with a single document is trivially
        uniform."""
        return self._is_uniform

    @property
    def shapes(self) -> Tuple[DimensionType, ...]:
        """Per-document spatial token layout, as normalized at construction.
        Read-only: the returned tuple must not be mutated."""
        return self._shapes

    @property
    def device(self) -> Optional[torch.device]:
        """Device this layout is pinned to, or ``None`` if not yet
        materialized. The layout pins to a device on construction with an
        explicit ``device=...``, or otherwise on first use with a
        ``na{1,2,3}d_varlen`` entry point, whichever happens first.
        Exception: a layout whose documents are all empty is never pinned
        by a ``na{1,2,3}d_varlen`` call -- the all-empty fast path returns
        before any device work, materialization included."""
        return self._device

    # -- construction / interop helpers ----------------------------------

    @classmethod
    def from_tensor_list(cls, xs: Sequence[Tensor]) -> Tuple["VarlenLayout", Tensor]:
        """Builds a layout and packs a list of per-document tensors in one step.

        Each ``xs[i]`` is shaped ``[*spatial, heads, head_dim]``, with spatial
        rank inferred as ``xs[i].ndim - 2`` and required equal, along with
        ``heads``/``head_dim``/``dtype``/``device``, across every document.
        ``from_tensor_list`` guarantees consistency between the returned
        layout and packed tensor at construction time; tensors derived from
        that packed representation must preserve its document order and
        token layout. Input tensor order is preserved in the packed
        representation.

        Packing is a pure tensor operation (``cat``/``reshape``), so this
        works the same on CPU input (e.g. packing inside a ``DataLoader``
        worker, which the layout's pickle support exists for) as on CUDA
        input, and never requires a CUDA device itself. The returned layout
        is unmaterialized regardless of ``xs``'s device -- same as directly
        constructing with no ``device=`` -- and pins lazily on the first
        ``na{1,2,3}d_varlen`` call.

        Cost: contiguous documents are packed with a single ``torch.cat``
        (each document view-reshaped, no copy). A non-contiguous document
        forces its own O(document tokens) copy inside ``reshape`` before the
        concatenation -- i.e. this path costs O(num_docs) reshape copies (at
        most) plus one ``cat``, not a single fused kernel.

        Returns:
            layout (VarlenLayout): Layout describing ``xs``'s per-document
                spatial shapes.

            packed (Tensor): ``[total_tokens, heads, head_dim]`` concatenation
                of ``xs`` in order.
        """
        if not isinstance(xs, Sequence) or isinstance(xs, (str, bytes)) or not xs:
            raise ValueError(
                "from_tensor_list requires a non-empty sequence of Tensors."
            )
        first = xs[0]
        if not isinstance(first, Tensor):
            raise TypeError(f"xs[0] must be a Tensor, got {type(first)=}.")
        rank = first.dim() - 2
        if rank not in (1, 2, 3):
            raise ValueError(
                "Each tensor must be rank (spatial rank + 2) with spatial rank in "
                f"(1, 2, 3); got xs[0].dim()={first.dim()}."
            )
        heads, head_dim = first.shape[-2], first.shape[-1]
        dtype, device = first.dtype, first.device

        shapes = []
        flat = []
        for index, x in enumerate(xs):
            if not isinstance(x, Tensor):
                raise TypeError(f"xs[{index}] must be a Tensor, got {type(x)=}.")
            if x.dim() != rank + 2:
                raise ValueError(
                    f"xs[{index}] has rank {x.dim()}; every document must share "
                    f"xs[0]'s spatial rank ({rank}), i.e. tensor rank {rank + 2}."
                )
            if x.shape[-2] != heads or x.shape[-1] != head_dim:
                raise ValueError(
                    f"xs[{index}]'s (heads, head_dim)={tuple(x.shape[-2:])} does not "
                    f"match xs[0]'s {(heads, head_dim)}."
                )
            if x.dtype != dtype:
                raise ValueError(
                    f"xs[{index}].dtype={x.dtype} does not match xs[0].dtype={dtype}."
                )
            if x.device != device:
                raise ValueError(
                    f"xs[{index}].device={x.device} does not match "
                    f"xs[0].device={device}."
                )
            shapes.append(cast(DimensionType, tuple(x.shape[:-2])))
            flat.append(x.reshape(-1, heads, head_dim))

        packed = torch.cat(flat, dim=0)
        layout = cls(shapes)
        return layout, packed

    def split(self, packed: Tensor) -> List[Tensor]:
        """Splits a packed tensor back into one view per document.

        Inverse of [from_tensor_list][natten.VarlenLayout.from_tensor_list]
        (and of a manual ``torch.cat`` following this layout's ``shapes``).
        Zero-copy when ``packed`` is contiguous: each document is a
        ``narrow`` + ``reshape`` view into ``packed``'s storage. The
        round-trip guarantee is **value** equality, not stride/contiguity
        equality -- if ``packed`` (or a prior ``from_tensor_list`` input) was
        non-contiguous, ``reshape`` may copy, and the returned view's strides
        need not match an original input's.

        Parameters:
            packed (Tensor): ``[total_tokens, ...]`` tensor; ``total_tokens``
                must equal [total_tokens][natten.VarlenLayout.total_tokens].

        Returns:
            docs (List[Tensor]): One ``[*spatial, ...]`` tensor per document,
                in layout order.
        """
        if packed.shape[0] != self._total_tokens:
            raise ValueError(
                f"packed.shape[0]={packed.shape[0]} does not match this layout's "
                f"total_tokens={self._total_tokens}."
            )
        trailing = tuple(packed.shape[1:])
        docs = []
        start = 0
        for shape in self._shapes:
            length = math.prod(shape)
            docs.append(packed.narrow(0, start, length).reshape(*shape, *trailing))
            start += length
        return docs

    # -- pickling ---------------------------------------------------------

    def __getstate__(self) -> Dict[str, Any]:
        # Host shapes only: materialized tensors/memo are process-and-device-
        # local and must not cross a pickle boundary (a materialized CUDA
        # tensor pickled in one process and unpickled in a DataLoader worker
        # would carry device state across a process boundary, which is not
        # meaningful).
        return {"shapes": self._shapes}

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self._set_host_properties(state["shapes"])
        self._device = None
        self._memo = {}
        self._cu_seqlens = None
        self._token_layouts = None
        self._fold_memo = {}
        self._permute_memo = {}
