from pathlib import Path
from types import ModuleType

import pytest
import torch
import triton
from mlps_shared.imports import ImportPath
from reference_rope import apply_rotary_emb
from triton.runtime.interpreter import InterpretedFunction

with ImportPath(Path(__file__).parents[1]):
    from examples import attn_triton_rope, attn_triton_rope_varlen


# Each triton example carries its own copy of the kernel.
@pytest.fixture(
    params=[attn_triton_rope, attn_triton_rope_varlen],
    ids=lambda module: module.__name__.rsplit(".", 1)[-1],
)
def example(request: pytest.FixtureRequest) -> ModuleType:
    return request.param


class TestRopeQK:
    """The triton RoPE kernel, run by triton's interpreter on CPU tensors."""

    T, H, D = 37, 15, 64  # T not a multiple of BLOCK_T, H not a power of two

    @pytest.fixture(autouse=True, params=[1, 2, 4, 8], ids=lambda b: f"BLOCK_T={b}")
    def rope(
        self,
        request: pytest.FixtureRequest,
        monkeypatch: pytest.MonkeyPatch,
        example: ModuleType,
    ) -> ModuleType:
        """Each BLOCK_T the autotuner can pick, on triton's CPU interpreter."""
        kernel = InterpretedFunction(example.rope_qk_kernel.fn)
        config = triton.Config({"BLOCK_T": request.param})
        monkeypatch.setattr(
            example,
            "rope_qk_tuned",
            triton.autotune(configs=[config], key=[], restore_value=["dst"])(kernel),
        )
        return example

    def _inputs(
        self, rope: ModuleType
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        torch.manual_seed(0)
        qkv = torch.randn(self.T, 3, self.H, self.D)
        return qkv, *rope.rope_tables(torch.arange(self.T), self.D)

    def test_forward_matches_apply_rotary_emb(self, rope: ModuleType) -> None:
        qkv, cos, sin = self._inputs(rope)
        qk = torch.empty(self.T, 2, self.H, self.D)
        rope.rope_qk(qkv, qk, cos, sin, backward=False)
        for slot in (0, 1):
            expected = apply_rotary_emb(qkv[:, slot], cos[:, None], sin[:, None])
            torch.testing.assert_close(qk[:, slot], expected)

    def test_backward_in_place_is_the_transpose(self, rope: ModuleType) -> None:
        """<rope(x), g> == <x, rope^T(g)>, and v's slot is left alone."""
        qkv, cos, sin = self._inputs(rope)
        g = torch.randn_like(qkv)
        rotated = torch.empty(self.T, 2, self.H, self.D)
        rope.rope_qk(qkv, rotated, cos, sin, backward=False)
        g_t = g.clone()
        rope.rope_qk(g_t, g_t, cos, sin, backward=True)

        torch.testing.assert_close(
            (rotated * g[:, :2]).sum(), (qkv[:, :2] * g_t[:, :2]).sum()
        )
        assert torch.equal(g_t[:, 2], g[:, 2])

    def test_one_table_shared_across_batch(self, rope: ModuleType) -> None:
        """Fixed (B, S) batches pass one (S, D/2) table; token t uses row t % S."""
        batch, seqlen = 3, 11
        torch.manual_seed(0)
        qkv = torch.randn(batch, seqlen, 3, self.H, self.D)
        cos, sin = rope.rope_tables(torch.arange(seqlen), self.D)
        qk = torch.empty(batch, seqlen, 2, self.H, self.D)
        rope.rope_qk(qkv.flatten(0, 1), qk.flatten(0, 1), cos, sin, backward=False)
        for slot in (0, 1):
            expected = apply_rotary_emb(qkv[:, :, slot], cos[:, None], sin[:, None])
            torch.testing.assert_close(qk[:, :, slot], expected)

    def test_head_dim_not_a_power_of_two(self, rope: ModuleType) -> None:
        """D/2 = 48 is padded up to 64 lanes inside the kernel and masked."""
        head_dim = 96
        torch.manual_seed(0)
        qkv = torch.randn(self.T, 3, self.H, head_dim)
        cos, sin = rope.rope_tables(torch.arange(self.T), head_dim)
        qk = torch.empty(self.T, 2, self.H, head_dim)
        rope.rope_qk(qkv, qk, cos, sin, backward=False)
        for slot in (0, 1):
            expected = apply_rotary_emb(qkv[:, slot], cos[:, None], sin[:, None])
            torch.testing.assert_close(qk[:, slot], expected)


def test_check_rope_tables_accepts_matching_tables(example: ModuleType) -> None:
    cos, sin = example.rope_tables(torch.arange(10), 64)
    example.check_rope_tables(cos, sin, rows=10, head_dim=64)


def test_check_rope_tables_rejects_wrong_rows(example: ModuleType) -> None:
    """A longer table would silently rotate with the wrong positions."""
    cos, sin = example.rope_tables(torch.arange(4096), 64)
    with pytest.raises(ValueError, match="cos/sin must be"):
        example.check_rope_tables(cos, sin, rows=1024, head_dim=64)


def test_check_rope_tables_rejects_odd_head_dim(example: ModuleType) -> None:
    cos, sin = example.rope_tables(torch.arange(10), 64)
    with pytest.raises(ValueError, match="even head_dim"):
        example.check_rope_tables(cos, sin, rows=10, head_dim=63)
