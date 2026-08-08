import sys

from epubconv.logging_setup import _force_utf8_console


def test_force_utf8_console_reconfigures_streams_with_reconfigure(monkeypatch):
    calls = []

    class FakeStream:
        def reconfigure(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setattr(sys, "stdout", FakeStream())
    monkeypatch.setattr(sys, "stderr", FakeStream())

    _force_utf8_console()

    assert calls == [{"encoding": "utf-8", "errors": "replace"}] * 2


def test_force_utf8_console_ignores_streams_without_reconfigure(monkeypatch):
    class FakeStream:
        pass

    monkeypatch.setattr(sys, "stdout", FakeStream())
    monkeypatch.setattr(sys, "stderr", FakeStream())

    _force_utf8_console()  # must not raise
