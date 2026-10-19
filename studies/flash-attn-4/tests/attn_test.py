import functools
from collections.abc import Callable
from pathlib import Path

import pytest
import torch
from flash_attn.cute import flash_attn_func
from mlps_shared.imports import ImportPath
from torch._subclasses.fake_tensor import FakeTensorMode

with ImportPath(Path(__file__).parents[1]):
    from examples import attn

hardware_supports_fa4 = attn.hardware_supports_fa4

AttnFn = Callable[[torch.Tensor], torch.Tensor]


def test_fake_matches_kernel_layout() -> None:
    """torch.compile traces the ops through their fake impls, so the shapes and
    dtypes there must match what the real ops return."""
    batch, seqlen, heads, head_dim = 2, 6, 4, 16
    with FakeTensorMode():
        qkv = torch.empty(batch, seqlen, 3, heads, head_dim, dtype=torch.bfloat16)
        out, lse = attn._fa4_fwd(qkv, False)  # noqa: FBT003
        dqkv = attn._fa4_bwd(out, out, lse, qkv, False)  # noqa: FBT003

    assert out.shape == (batch, seqlen, heads, head_dim)
    assert out.dtype == torch.bfloat16
    assert lse.shape == (batch, heads, seqlen)
    assert lse.dtype == torch.float32
    assert dqkv.shape == qkv.shape
    assert dqkv.dtype == qkv.dtype


@pytest.mark.skipif(not hardware_supports_fa4(), reason="Needs a Hopper+ GPU.")
class TestFA4:
    """`fa4_attn` must equal FA4's own `flash_attn_func` on unbound q, k, v,
    forward and backward, and stay opaque to torch.compile."""

    B, S, H, D = 3, 250, 4, 64

    def _inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        torch.manual_seed(0)
        shape = (self.B, self.S, 3, self.H, self.D)
        qkv = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
        dout = torch.randn(
            self.B, self.S, self.H, self.D, device="cuda", dtype=torch.bfloat16
        )
        return qkv, dout

    @staticmethod
    def _fa4(qkv: torch.Tensor, *, causal: bool = False) -> torch.Tensor:
        """Reference: FA4's own op on q, k, v (autograd stacks their grads)."""
        q, k, v = qkv.unbind(2)
        out, _ = flash_attn_func(q, k, v, causal=causal)
        return out

    @staticmethod
    def _fwd_bwd(
        fn: AttnFn,
        qkv: torch.Tensor,
        dout: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``fn``'s output and the gradient it sends back to ``qkv``."""
        qkv = qkv.detach().requires_grad_()
        out = fn(qkv)
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
        fn: AttnFn = functools.partial(attn.fa4_attn, causal=causal)
        if compiled:
            # fullgraph: the wrapper and both ops must trace with no graph break.
            fn = torch.compile(fn, fullgraph=True)
        qkv, dout = self._inputs()
        ref_fn = functools.partial(self._fa4, causal=causal)
        expected = self._fwd_bwd(ref_fn, qkv, dout)
        actual = self._fwd_bwd(fn, qkv, dout)
        for got, ref in zip(actual, expected, strict=True):
            torch.testing.assert_close(got, ref, atol=2e-2, rtol=2e-2)

    @pytest.mark.parametrize("causal", [False, True], ids=["full", "causal"])
    def test_opcheck(self, causal: bool) -> None:  # noqa: FBT001
        qkv, dout = self._inputs()
        fwd_args = (qkv.requires_grad_(), causal)
        torch.library.opcheck(attn._fa4_fwd, fwd_args)
        out, lse = attn._fa4_fwd(*fwd_args)
        bwd_args = (dout, out.detach(), lse, qkv.detach(), *fwd_args[1:])
        torch.library.opcheck(attn._fa4_bwd, bwd_args)
