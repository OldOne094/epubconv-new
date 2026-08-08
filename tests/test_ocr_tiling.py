import sys
import types

import numpy as np

from epubconv import ocr as ocr_module
from epubconv.models import BBox, Word
from epubconv.ocr import _merge_tile_words, plan_tile_strips, run_ocr_tiled


def word(text, y0, y1, x0=0, x1=50, conf=0.9):
    return Word(text=text, bbox=BBox(x0, y0, x1, y1), confidence=conf)


def test_merge_tile_words_keeps_non_contested_words_untouched():
    a = word("a", 0, 20)
    b = word("b", 200, 220)
    tagged = [(a, 0), (b, 1)]
    result = _merge_tile_words(tagged, boundaries=[(60, 100)])
    assert set(w.text for w in result) == {"a", "b"}


def test_merge_tile_words_picks_highest_confidence_strip_for_duplicate_line():
    # Both strips detected "the same" line (overlapping y-ranges) inside the
    # boundary zone (60, 100); strip 1's detection should win on confidence.
    weak = word("weak-reading", 65, 95, conf=0.4)
    strong = word("strong-reading", 65, 95, conf=0.95)
    tagged = [(weak, 0), (strong, 1)]

    result = _merge_tile_words(tagged, boundaries=[(60, 100)])

    assert [w.text for w in result] == ["strong-reading"]


def test_merge_tile_words_keeps_a_cluster_seen_by_only_one_strip():
    only_strip = word("only-here", 65, 95, conf=0.5)
    tagged = [(only_strip, 0)]

    result = _merge_tile_words(tagged, boundaries=[(60, 100)])

    assert [w.text for w in result] == ["only-here"]


def test_merge_tile_words_treats_different_words_on_the_confirmed_line_separately():
    # Two DIFFERENT words that both happen to be non-contested must both survive
    # (dedup must never touch words outside an overlap zone).
    a = word("first", 0, 20)
    b = word("second", 0, 20, x0=60, x1=100)
    result = _merge_tile_words([(a, 0), (b, 0)], boundaries=[(200, 240)])
    assert {w.text for w in result} == {"first", "second"}


def test_merge_tile_words_resolves_a_three_way_contested_line_to_one_winner():
    # Regression test for a real duplication bug found on an actual page: this
    # codebase's tile geometry means THREE strips can independently detect the
    # same real line at once (consecutive overlap zones overlap each other).
    # Pairwise "match the first candidate found, then stop" only pairs off two
    # of the three, leaving the third an orphan with nothing left to compare
    # against - so it survives untouched as a visible duplicate right next to
    # the winning reading, even though it geometrically overlaps both others.
    good = word("good-reading", 60, 90, conf=0.95)  # strip 0
    noisy = word("noisy-reading", 62, 88, conf=0.60)  # strip 1
    whole_line = word("whole-line-reading", 65, 85, conf=0.80)  # strip 2
    tagged = [(good, 0), (noisy, 1), (whole_line, 2)]

    result = _merge_tile_words(tagged, boundaries=[(50, 150)])

    assert [w.text for w in result] == ["good-reading"]


def test_merge_tile_words_does_not_let_a_bad_detection_bridge_two_real_lines():
    # Regression test for a real content-loss bug: a single oversized/misplaced
    # low-confidence detection from strip 0 geometrically overlaps BOTH a real
    # line from strip 0 above it and a real line from strip 1 below it. The old
    # single-linkage clustering (comparing each new word only to the *previous*
    # cluster member) chained all three into one cluster; picking "the strip
    # with the highest average confidence" for that fused cluster then kept only
    # strip 1's line and silently discarded strip 0's real line entirely - not
    # flagged as low-confidence, just gone from the output.
    line_a = word("line-a", 60, 90, conf=0.9)  # strip 0, real content
    bad_fragment = word("garbage", 75, 125, conf=0.3)  # strip 0, bad detection
    line_b = word("line-b", 110, 140, conf=0.9)  # strip 1, real content
    tagged = [(line_a, 0), (bad_fragment, 0), (line_b, 1)]

    result = _merge_tile_words(tagged, boundaries=[(50, 150)])

    texts = {w.text for w in result}
    assert "line-a" in texts, "real content from strip 0 must never be silently dropped"
    assert "line-b" in texts


def _raw(texts, scores, boxes):
    return [{"res": {"rec_texts": texts, "rec_scores": scores, "rec_boxes": boxes}}]


class FakeEngine:
    def __init__(self, responses):
        self._responses = list(responses)
        self.crop_heights = []

    def predict(self, image):
        self.crop_heights.append(image.shape[0])
        return self._responses.pop(0)


def test_run_ocr_tiled_passthrough_for_image_not_taller_than_one_strip():
    engine = FakeEngine([_raw(["مرحبا"], [0.9], [[0, 0, 10, 10]])])
    image = np.zeros((80, 50, 3), dtype=np.uint8)

    words = run_ocr_tiled(image, engine=engine, strip_height=100, overlap=40)

    assert len(engine.crop_heights) == 1
    assert [w.text for w in words] == ["مرحبا"]


def test_run_ocr_tiled_shifts_coordinates_into_full_page_space():
    # strip_height=120, overlap=40 on a 180-tall image -> strips [0,120) and
    # [80,180) (the second one hits the bottom edge and the loop stops there).
    engine = FakeEngine(
        [
            _raw(["top"], [0.9], [[0, 10, 10, 30]]),  # local y 10-30 -> global 10-30
            _raw(["bottom"], [0.9], [[0, 30, 10, 50]]),  # local y 30-50, strip starts at 80 -> global 110-130
        ]
    )
    image = np.zeros((180, 50, 3), dtype=np.uint8)

    words = run_ocr_tiled(image, engine=engine, strip_height=120, overlap=40)

    assert len(engine.crop_heights) == 2
    by_text = {w.text: w for w in words}
    assert by_text["top"].bbox.y0 == 10
    assert by_text["bottom"].bbox.y0 == 110


def test_run_ocr_tiled_stops_at_bottom_edge_without_a_redundant_trailing_strip():
    # Regression test: the loop used to keep advancing past a strip that already
    # reached the image bottom, reprocessing an already-covered tail region as an
    # "unboundaried" (never deduplicated) extra strip and duplicating its text.
    engine = FakeEngine(
        [
            _raw(["a"], [0.9], [[0, 10, 10, 30]]),
            _raw(["b"], [0.9], [[0, 10, 10, 30]]),
        ]
    )
    image = np.zeros((180, 50, 3), dtype=np.uint8)  # strip_height=120, overlap=40 -> [0,120), [80,180)

    words = run_ocr_tiled(image, engine=engine, strip_height=120, overlap=40)

    assert len(engine.crop_heights) == 2  # not 3+
    assert len(words) == 2


# --- plan_tile_strips ---


def test_plan_tile_strips_single_strip_when_image_fits():
    assert plan_tile_strips(80, strip_height=100, overlap=40) == [(0, 80)]


def test_plan_tile_strips_matches_the_grid_run_ocr_tiled_actually_uses():
    assert plan_tile_strips(180, strip_height=120, overlap=40) == [(0, 120), (80, 180)]


# --- manual_cuts ---


def test_run_ocr_tiled_with_manual_cuts_splits_exactly_there_no_overlap():
    engine = FakeEngine(
        [
            _raw(["top"], [0.9], [[0, 10, 10, 30]]),  # strip [0,100) local y 10-30 -> global 10-30
            _raw(["bottom"], [0.9], [[0, 5, 10, 25]]),  # strip [100,180) local y 5-25 -> global 105-125
        ]
    )
    image = np.zeros((180, 50, 3), dtype=np.uint8)

    words = run_ocr_tiled(image, engine=engine, manual_cuts=[100])

    assert len(engine.crop_heights) == 2
    assert engine.crop_heights == [100, 80]
    by_text = {w.text: w for w in words}
    assert by_text["top"].bbox.y0 == 10
    assert by_text["bottom"].bbox.y0 == 105


def test_run_ocr_tiled_manual_cuts_ignores_out_of_range_values():
    engine = FakeEngine([_raw(["only"], [0.9], [[0, 0, 10, 10]])])
    image = np.zeros((100, 50, 3), dtype=np.uint8)

    # 0, 100, and negative are all out of the open range (0, height) and dropped,
    # leaving no real cut -> a single strip covering the whole image.
    words = run_ocr_tiled(image, engine=engine, manual_cuts=[0, 100, -5])

    assert len(engine.crop_heights) == 1
    assert engine.crop_heights[0] == 100
    assert [w.text for w in words] == ["only"]


def test_run_ocr_tiled_manual_cuts_never_produces_duplicates_across_strips():
    # Even with words placed right at a manual cut boundary in each strip, manual
    # mode trusts the person's cuts completely and never runs dedup - so nothing
    # should vanish here, unlike the automatic-grid overlap zones.
    engine = FakeEngine(
        [
            _raw(["a"], [0.9], [[0, 95, 10, 100]]),  # strip [0,100)
            _raw(["b"], [0.9], [[0, 0, 10, 5]]),  # strip [100,180), local y0=0 -> global 100
        ]
    )
    image = np.zeros((180, 50, 3), dtype=np.uint8)

    words = run_ocr_tiled(image, engine=engine, manual_cuts=[100])

    assert {w.text for w in words} == {"a", "b"}


# --- on_progress ---


def test_run_ocr_tiled_reports_progress_per_strip():
    engine = FakeEngine(
        [
            _raw(["a"], [0.9], [[0, 0, 10, 10]]),
            _raw(["b"], [0.9], [[0, 0, 10, 10]]),
        ]
    )
    image = np.zeros((180, 50, 3), dtype=np.uint8)  # strip_height=120, overlap=40 -> 2 strips
    calls = []

    run_ocr_tiled(
        image, engine=engine, strip_height=120, overlap=40, on_progress=lambda c, t: calls.append((c, t))
    )

    assert calls == [(1, 2), (2, 2)]


def test_run_ocr_tiled_reports_progress_for_manual_cuts_too():
    engine = FakeEngine(
        [
            _raw(["a"], [0.9], [[0, 0, 10, 10]]),
            _raw(["b"], [0.9], [[0, 0, 10, 10]]),
        ]
    )
    image = np.zeros((180, 50, 3), dtype=np.uint8)
    calls = []

    run_ocr_tiled(image, engine=engine, manual_cuts=[100], on_progress=lambda c, t: calls.append((c, t)))

    assert calls == [(1, 2), (2, 2)]


def test_run_ocr_tiled_reports_progress_for_single_strip_passthrough():
    engine = FakeEngine([_raw(["a"], [0.9], [[0, 0, 10, 10]])])
    image = np.zeros((50, 50, 3), dtype=np.uint8)
    calls = []

    run_ocr_tiled(image, engine=engine, strip_height=100, on_progress=lambda c, t: calls.append((c, t)))

    assert calls == [(1, 1)]


# --- oneDNN crash recovery, propagated across the rest of a tiled page ---


class _CrashesOnceThenNeverCalledAgain:
    def __init__(self):
        self.calls = 0

    def predict(self, image):
        self.calls += 1
        raise NotImplementedError(
            "(Unimplemented) ConvertPirAttribute2RuntimeAttribute not support "
            "[pir::ArrayAttribute<pir::DoubleAttribute>]"
        )


def test_run_ocr_tiled_recovers_from_onednn_crash_and_reuses_healthy_engine(monkeypatch):
    class RecoveredPaddleOCR:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.calls = 0

        def predict(self, image):
            self.calls += 1
            return _raw(["recovered"], [0.9], [[0, 0, 10, 10]])

    monkeypatch.setitem(sys.modules, "paddleocr", types.SimpleNamespace(PaddleOCR=RecoveredPaddleOCR))
    ocr_module.reset_engine()

    crashy = _CrashesOnceThenNeverCalledAgain()
    image = np.zeros((180, 50, 3), dtype=np.uint8)  # strip_height=120, overlap=40 -> 2 strips

    words = run_ocr_tiled(image, engine=crashy, strip_height=120, overlap=40)

    assert crashy.calls == 1  # never retried on the broken engine itself
    assert [w.text for w in words] == ["recovered", "recovered"]  # both strips via the rebuilt engine
    assert ocr_module.get_last_resolved_device() == "cpu"
