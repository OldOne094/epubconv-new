import sys
import types

from epubconv import ocr
from epubconv.ocr import _detect_device


def _install_fake_paddle(monkeypatch, compiled_with_cuda: bool, device_count: int):
    fake_cuda = types.SimpleNamespace(device_count=lambda: device_count)
    fake_device = types.SimpleNamespace(cuda=fake_cuda)
    fake_paddle = types.SimpleNamespace(
        is_compiled_with_cuda=lambda: compiled_with_cuda, device=fake_device
    )
    monkeypatch.setitem(sys.modules, "paddle", fake_paddle)


def test_detect_device_returns_gpu_when_cuda_build_sees_a_device(monkeypatch):
    _install_fake_paddle(monkeypatch, compiled_with_cuda=True, device_count=1)
    assert _detect_device() == "gpu"


def test_detect_device_returns_cpu_when_not_compiled_with_cuda(monkeypatch):
    _install_fake_paddle(monkeypatch, compiled_with_cuda=False, device_count=0)
    assert _detect_device() == "cpu"


def test_detect_device_returns_cpu_when_cuda_build_sees_no_device(monkeypatch):
    _install_fake_paddle(monkeypatch, compiled_with_cuda=True, device_count=0)
    assert _detect_device() == "cpu"


def test_detect_device_falls_back_to_cpu_when_paddle_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "paddle", None)
    assert _detect_device() == "cpu"


class _FakePaddleOCR:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def predict(self, image):
        return ["recovered"]


def _install_fake_paddleocr(monkeypatch):
    monkeypatch.setitem(sys.modules, "paddleocr", types.SimpleNamespace(PaddleOCR=_FakePaddleOCR))


def test_build_engine_records_the_resolved_device(monkeypatch):
    _install_fake_paddleocr(monkeypatch)
    ocr.reset_engine()
    assert ocr.get_last_resolved_device() is None

    ocr._build_engine("ar", device="cpu")

    assert ocr.get_last_resolved_device() == "cpu"


def test_build_engine_always_disables_mkldnn_regardless_of_device(monkeypatch):
    # Regression test: PaddleX can silently fall back to CPU per-call even after
    # GPU was requested and available at construction time, and oneDNN's kernel
    # crashes on this PaddlePaddle build if mkldnn is enabled when that happens.
    _install_fake_paddleocr(monkeypatch)

    engine = ocr._build_engine("ar", device="gpu")

    assert engine.kwargs["enable_mkldnn"] is False
    assert engine.kwargs["device"] == "gpu"


def test_reset_engine_clears_the_recorded_device(monkeypatch):
    _install_fake_paddleocr(monkeypatch)
    ocr._build_engine("ar", device="gpu")
    assert ocr.get_last_resolved_device() == "gpu"

    ocr.reset_engine()

    assert ocr.get_last_resolved_device() is None


def test_get_engine_reuses_cache_for_repeated_auto_requests(monkeypatch):
    _install_fake_paddleocr(monkeypatch)
    ocr.reset_engine()

    first = ocr.get_engine("ar", device="auto")
    second = ocr.get_engine("ar", device="auto")

    assert first is second


def test_get_engine_rebuilds_when_explicit_device_differs_from_cached(monkeypatch):
    # Regression test: a user picking "CPU" in the review UI after an engine
    # already came up on GPU (or vice versa) must actually get a new engine, not
    # silently keep using the old device.
    _install_fake_paddleocr(monkeypatch)
    ocr.reset_engine()

    gpu_engine = ocr.get_engine("ar", device="gpu")
    assert ocr.get_last_resolved_device() == "gpu"

    cpu_engine = ocr.get_engine("ar", device="cpu")

    assert ocr.get_last_resolved_device() == "cpu"
    assert cpu_engine is not gpu_engine


def test_get_engine_auto_request_does_not_force_a_rebuild(monkeypatch):
    _install_fake_paddleocr(monkeypatch)
    ocr.reset_engine()

    explicit = ocr.get_engine("ar", device="cpu")
    reused = ocr.get_engine("ar", device="auto")

    assert reused is explicit
    assert ocr.get_last_resolved_device() == "cpu"


# --- _predict_recovering ---

_ONEDNN_MESSAGE = (
    "(Unimplemented) ConvertPirAttribute2RuntimeAttribute not support "
    "[pir::ArrayAttribute<pir::DoubleAttribute>]"
)


class _AlwaysWorksEngine:
    def predict(self, image):
        return ["ok"]


class _CrashesOnceEngine:
    def __init__(self):
        self.calls = 0

    def predict(self, image):
        self.calls += 1
        raise NotImplementedError(_ONEDNN_MESSAGE)


class _UnrelatedCrashEngine:
    def predict(self, image):
        raise NotImplementedError("some other unimplemented op, nothing to do with oneDNN")


def test_predict_recovering_passes_through_on_success():
    engine = _AlwaysWorksEngine()

    result, used = ocr._predict_recovering(engine, "img", "ar")

    assert result == ["ok"]
    assert used is engine


def test_predict_recovering_rebuilds_explicitly_on_cpu_after_onednn_crash(monkeypatch):
    _install_fake_paddleocr(monkeypatch)
    ocr.reset_engine()

    result, used = ocr._predict_recovering(_CrashesOnceEngine(), "img", "ar")

    assert result == ["recovered"]
    assert isinstance(used, _FakePaddleOCR)
    assert used.kwargs["device"] == "cpu"
    assert ocr.get_last_resolved_device() == "cpu"


def test_predict_recovering_reraises_unrelated_not_implemented_errors():
    import pytest

    with pytest.raises(NotImplementedError, match="nothing to do with oneDNN"):
        ocr._predict_recovering(_UnrelatedCrashEngine(), "img", "ar")
