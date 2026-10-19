import functools
import itertools
from collections.abc import Callable
from pathlib import Path
from typing import TypedDict, Unpack

import pytest
import torch
from flash_attn.cute import flash_attn_varlen_func
from mlps_shared.imports import ImportPath
from torch._subclasses.fake_tensor import FakeTensorMode

with ImportPath(Path(__file__).parents[1]):
    from examples import attn_varlen

hardware_supports_fa4 = attn_varlen.hardware_supports_fa4


class AttnKwargs(TypedDict):
    """Keyword args shared by `fa4_rope_varlen_attn` and the reference."""

    cu_seqlens: torch.Tensor
    max_seqlen: int


AttnFn = Callable[..., torch.Tensor]


def test_fake_matches_kernel_layout() -> None:
    """torch.compile traces the ops through their fake impls, so the shapes and
    dtypes there must match what the real ops return."""
    total, heads, head_dim = 6, 4, 16
    with FakeTensorMode():
        qkv = torch.empty(total, 3, heads, head_dim, dtype=torch.bfloat16)
        cu_seqlens = torch.empty(3, dtype=torch.int32)
        causal = False
        out, lse = attn_varlen._fa4_varlen_fwd(qkv, cu_seqlens, 4, causal)
        dqkv = attn_varlen._fa4_varlen_bwd(out, out, lse, qkv, cu_seqlens, 4, causal)

    assert out.shape == (total, heads, head_dim)
    assert out.dtype == torch.bfloat16
    assert lse.shape == (heads, total)
    assert lse.dtype == torch.float32
    assert dqkv.shape == qkv.shape
    assert dqkv.dtype == qkv.dtype


@pytest.mark.skipif(not hardware_supports_fa4(), reason="Needs a Hopper+ GPU.")
class TestFA4Varlen:
    """`fa4_varlen_attn` must equal FA4's own `flash_attn_varlen_func` on
    unbound q, k, v, forward and backward, and stay opaque to torch.compile."""

    LENS = (100, 250, 162)  # packed sequence lengths
    MAX_SEQLEN = max(LENS)
    H, D = 4, 64

    def _inputs(self) -> tuple[torch.Tensor, torch.Tensor, AttnKwargs]:
        torch.manual_seed(0)
        total = sum(self.LENS)
        cu_seqlens = torch.tensor(
            [0, *itertools.accumulate(self.LENS)], dtype=torch.int32, device="cuda"
        )
        qkv = torch.randn(total, 3, self.H, self.D, device="cuda", dtype=torch.bfloat16)
        dout = torch.randn(total, self.H, self.D, device="cuda", dtype=torch.bfloat16)
        attn_kwargs = AttnKwargs(cu_seqlens=cu_seqlens, max_seqlen=self.MAX_SEQLEN)
        return qkv, dout, attn_kwargs

    @staticmethod
    def _fa4(
        qkv: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        causal: bool = False,
    ) -> torch.Tensor:
        """Reference: FA4's own varlen op on q, k, v (autograd stacks grads)."""
        q, k, v = qkv.unbind(1)
        out, _ = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            causal=causal,
        )
        return out

    @staticmethod
    def _fwd_bwd(
        fn: AttnFn,
        qkv: torch.Tensor,
        dout: torch.Tensor,
        **attn_kwargs: Unpack[AttnKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``fn``'s output and the gradient it sends back to ``qkv``."""
        qkv = qkv.detach().requires_grad_()
        out = fn(qkv, **attn_kwargs)
        out.backward(dout)
        assert qkv.grad is not None
        return out.detach(), qkv.grad

    @pytest.mark.parametrize("compiled", [False, True], ids=["eager", "compiled"])
    @pytest.mark.parametrize("causal", [False, True], ids=["full", "causal"])
    def test_matches_fa4(
        self,
        compiled: bool,  # noqa: FBT001
        causal: bool,  # noqa: FBT001
    ) -> None:
        fn: AttnFn = functools.partial(attn_varlen.fa4_varlen_attn, causal=causal)
        if compiled:
            # fullgraph: the wrapper and both ops must trace with no graph break.
            fn = torch.compile(fn, fullgraph=True, dynamic=True)
        qkv, dout, kw = self._inputs()
        ref_fn = functools.partial(self._fa4, causal=causal)
        expected = self._fwd_bwd(ref_fn, qkv, dout, **kw)
        actual = self._fwd_bwd(fn, qkv, dout, **kw)
        for got, ref in zip(actual, expected, strict=True):
            torch.testing.assert_close(got, ref, atol=2e-2, rtol=2e-2)

    @pytest.mark.parametrize("causal", [False, True], ids=["full", "causal"])
    def test_opcheck(self, causal: bool) -> None:  # noqa: FBT001
        qkv, dout, kw = self._inputs()
        fwd_args = (qkv.requires_grad_(), kw["cu_seqlens"], kw["max_seqlen"], causal)
        torch.library.opcheck(attn_varlen._fa4_varlen_fwd, fwd_args)
        out, lse = attn_varlen._fa4_varlen_fwd(*fwd_args)
        bwd_args = (dout, out.detach(), lse, qkv.detach(), *fwd_args[1:])
        torch.library.opcheck(attn_varlen._fa4_varlen_bwd, bwd_args)
