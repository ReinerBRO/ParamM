from types import SimpleNamespace

import memory_router.train_transformer_router as train_router


def test_select_device_prefers_npu(monkeypatch):
    seen = {}

    monkeypatch.setattr(train_router, "torch_npu", object())
    monkeypatch.setattr(
        train_router.torch,
        "npu",
        SimpleNamespace(
            is_available=lambda: True,
            set_device=lambda device: seen.setdefault("device", device),
        ),
        raising=False,
    )
    monkeypatch.setattr(train_router.torch.cuda, "is_available", lambda: False)

    device, pin_memory = train_router._select_device()

    assert device == "npu:0"
    assert pin_memory is False
    assert seen["device"] == "npu:0"


def test_select_device_falls_back_to_cuda(monkeypatch):
    monkeypatch.setattr(train_router, "torch_npu", None)
    monkeypatch.setattr(train_router.torch.cuda, "is_available", lambda: True)

    device, pin_memory = train_router._select_device()

    assert device == "cuda"
    assert pin_memory is True


def test_select_device_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(train_router, "torch_npu", None)
    monkeypatch.setattr(train_router.torch.cuda, "is_available", lambda: False)

    device, pin_memory = train_router._select_device()

    assert device == "cpu"
    assert pin_memory is False
