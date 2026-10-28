import torch

from croissattn import attention, quantize_nvfp4


def test_roundtrip_and_axis_recast():
    x = torch.randn(2, 3, 32, 16, dtype=torch.float32)
    q = quantize_nvfp4(x, axis=-1)
    y = q.decode()
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    assert q.packed_payload().shape[-1] == q.padded // 2
    q2 = q.recast(-2)
    assert q2.decode().shape == x.shape


def test_complete_attention_backward_is_finite():
    torch.manual_seed(4)
    q = torch.randn(1, 2, 16, 16, requires_grad=True)
    k = torch.randn_like(q, requires_grad=True)
    v = torch.randn_like(q, requires_grad=True)
    out = attention(q, k, v, chunk_size=8)
    assert out.shape == q.shape
    loss = out.square().mean()
    loss.backward()
    assert torch.isfinite(out).all()
    assert torch.isfinite(q.grad).all()
    assert torch.isfinite(k.grad).all()
    assert torch.isfinite(v.grad).all()


def test_causal_output_changes_with_prefix_only():
    torch.manual_seed(7)
    q = torch.randn(1, 1, 8, 16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    q2, k2, v2 = q.clone(), k.clone(), v.clone()
    # Keep the tensor amax and block scales unchanged so this isolates the
    # causal mask rather than a deliberately different quantization scale.
    v2[..., 5:, :] += 1e-4
    a = attention(q, k, v)
    b = attention(q2, k2, v2)
    assert torch.allclose(a[..., :5, :], b[..., :5, :], atol=1e-5, rtol=1e-5)
