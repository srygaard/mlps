import functools
from collections.abc import Callable
from pathlib import Path

import pytest
import torch
from flash_attn.cute import flash_attn_func
from mlps_shared.imports import ImportPath
from reference_rope import apply_rotary_emb
from torch._subclasses.fake_tensor import FakeTensorMode

with ImportPath(Path(__file__).parents[1]):
    from examples import attn_triton_rope as attn_rope

hardware_supports_fa4 = attn_rope.hardware_supports_fa4
rope_tables = attn_rope.rope_tables

AttnFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


def test_fused_rope_fake_matches_kernel_layout() -> None:
    """torch.compile traces the ops through their fake impls, so the shapes and
    dtypes there must match what the real ops return."""
    batch, seqlen, heads, head_dim = 2, 6, 4, 16
    with FakeTensorMode():
        qkv = torch.empty(batch, seqlen, 3, heads, head_dim, dtype=torch.bfloat16)
        cos = torch.empty(seqlen, head_dim // 2)
        causal = False
        out, lse = attn_rope._fa4_rope_fwd(qkv, cos, cos, causal)
        dqkv = attn_rope._fa4_rope_bwd(out, out, lse, qkv, cos, cos, causal)

    assert out.shape == (batch, seqlen, heads, head_dim)
    assert out.dtype == torch.bfloat16
    assert lse.shape == (batch, heads, seqlen)
    assert lse.dtype == torch.float32
    assert dqkv.shape == qkv.shape
    assert dqkv.dtype == qkv.dtype


@pytest.mark.skipif(not hardware_supports_fa4(), reason="Needs a Hopper+ GPU.")
class TestFA4FusedRope:
    """`fa4_rope_attn` must equal RoPE (fp32 tables) followed by FA4's own
    `flash_attn_func`, forward and backward, and stay opaque to torch.compile."""

    B, S, H, D = 3, 250, 4, 64

    def _inputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        torch.manual_seed(0)
        cos, sin = rope_tables(torch.arange(self.S, device="cuda"), self.D)
        shape = (self.B, self.S, 3, self.H, self.D)
        qkv = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
        dout = torch.randn(
            self.B, self.S, self.H, self.D, device="cuda", dtype=torch.bfloat16
        )
        return qkv, cos, sin, dout

    @staticmethod
    def _rope_then_fa4(
        qkv: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        *,
        causal: bool = False,
    ) -> torch.Tensor:
        """Reference: fp32 `apply_rotary_emb`, then FA4's own op."""
        q, k, v = qkv.unbind(2)
        c, s = cos[:, None], sin[:, None]  # (S, 1, D/2), broadcast over B and H
        q = apply_rotary_emb(q.float(), c, s).bfloat16()
        k = apply_rotary_emb(k.float(), c, s).bfloat16()
        out, _ = flash_attn_func(q, k, v, causal=causal)
        return out

    @staticmethod
    def _fwd_bwd(
        fn: AttnFn,
        qkv: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        dout: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``fn``'s output and the gradient it sends back to ``qkv``."""
        qkv = qkv.detach().requires_grad_()
        out = fn(qkv, cos, sin)
        out.backward(dout)
        assert qkv.grad is not None
        return out.detach(), qkv.grad

    @pytest.mark.parametrize("compiled", [False, True], ids=["eager", "compiled"])
    @pytest.mark.parametrize("causal", [False, True], ids=["full", "causal"])
    def test_matches_rope_then_fa4(
        self,
        compiled: bool,  # noqa: FBT001
        causal: bool,  # noqa: FBT001
    ) -> None:
        fn: AttnFn = functools.partial(attn_rope.fa4_rope_attn, causal=causal)
        if compiled:
            # fullgraph: the wrapper and both ops must trace with no graph break.
            fn = torch.compile(fn, fullgraph=True)
        args = self._inputs()
        ref_fn = functools.partial(self._rope_then_fa4, causal=causal)
        expected = self._fwd_bwd(ref_fn, *args)
        actual = self._fwd_bwd(fn, *args)
        for got, ref in zip(actual, expected, strict=True):
            torch.testing.assert_close(got, ref, atol=2e-2, rtol=2e-2)

    @pytest.mark.parametrize("causal", [False, True], ids=["full", "causal"])
    def test_opcheck(self, causal: bool) -> None:  # noqa: FBT001
        qkv, cos, sin, dout = self._inputs()
        fwd_args = (qkv.requires_grad_(), cos, sin, causal)
        torch.library.opcheck(attn_rope._fa4_rope_fwd, fwd_args)
        out, lse = attn_rope._fa4_rope_fwd(*fwd_args)
        bwd_args = (dout, out.detach(), lse, qkv.detach(), *fwd_args[1:])
        torch.library.opcheck(attn_rope._fa4_rope_bwd, bwd_args)
