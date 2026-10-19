import inspect
import itertools

import pytest
import torch
from flash_attn.cute import flash_attn_func, flash_attn_varlen_func
from flash_attn.cute.interface import _flash_attn_bwd

# Every keyword the examples pass to `_flash_attn_bwd`.
USED_KWARGS = (
    "causal",
    "cu_seqlens_q",
    "cu_seqlens_k",
    "max_seqlen_q",
    "max_seqlen_k",
    "dq",
    "dk",
    "dv",
)
# The examples pass these positionally, in this order.
POSITIONAL = ("q", "k", "v", "out", "dout", "lse")


def hardware_supports_fa4() -> bool:
    """True when the CUDA device can run FA4 (Hopper, sm90, and newer)."""
    return torch.cuda.is_available() and torch.cuda.get_device_capability() >= (9, 0)


def test_signature_accepts_what_the_examples_pass() -> None:
    params = inspect.signature(_flash_attn_bwd).parameters
    assert tuple(params)[: len(POSITIONAL)] == POSITIONAL
    missing = [name for name in USED_KWARGS if name not in params]
    assert not missing, f"_flash_attn_bwd no longer takes {missing}"


@pytest.mark.skipif(not hardware_supports_fa4(), reason="Needs a Hopper+ GPU.")
class TestOutBuffers:
    """`_flash_attn_bwd` must write dq/dk/dv into the strided views it is given
    and return those same tensors: the examples return the packed buffer behind
    the views, so if FA4 wrote elsewhere they would return garbage."""

    H, D = 4, 64

    def _check(self, qkv: torch.Tensor, **varlen: object) -> None:
        """Runs FA4 forward on ``qkv`` (slots on dim -3), then the private
        backward into a NaN-filled packed buffer, and compares to autograd."""
        qkv = qkv.detach().requires_grad_()
        fwd = flash_attn_varlen_func if varlen else flash_attn_func
        kw = {f"{key}_{side}": val for key, val in varlen.items() for side in "qk"}
        out, lse = fwd(*qkv.unbind(-3), causal=True, return_lse=True, **kw)
        dout = torch.randn_like(out)
        (expected,) = torch.autograd.grad(out, qkv, dout)

        dqkv = torch.full_like(qkv, torch.nan)
        views = dqkv.unbind(-3)
        returned = _flash_attn_bwd(
            *qkv.detach().unbind(-3),
            out.detach(),
            dout,
            lse,
            causal=True,
            dq=views[0],
            dk=views[1],
            dv=views[2],
            **kw,
        )

        for got, view in zip(returned, views, strict=True):
            assert got.data_ptr() == view.data_ptr(), "returned a new tensor"
        assert not dqkv.isnan().any(), "left part of the out-buffers unwritten"
        torch.testing.assert_close(dqkv, expected, atol=2e-2, rtol=2e-2)

    def test_fixed_length(self) -> None:
        torch.manual_seed(0)
        shape = (3, 250, 3, self.H, self.D)
        self._check(torch.randn(shape, device="cuda", dtype=torch.bfloat16))

    def test_varlen(self) -> None:
        torch.manual_seed(0)
        lens = (100, 250, 162)
        cu_seqlens = torch.tensor(
            [0, *itertools.accumulate(lens)], dtype=torch.int32, device="cuda"
        )
        shape = (sum(lens), 3, self.H, self.D)
        qkv = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
        self._check(qkv, cu_seqlens=cu_seqlens, max_seqlen=max(lens))
