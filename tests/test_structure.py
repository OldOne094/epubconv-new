from epubconv.models import BBox, BlockKind
from epubconv.structure import analyze, group_lines_into_blocks, group_words_into_lines
from epubconv.models import Word


def word(text, x0, x1, y0=0, y1=10, conf=0.9):
    return Word(text=text, bbox=BBox(x0, y0, x1, y1), confidence=conf)


def test_group_words_into_lines_orders_right_to_left():
    # Three words on the same visual line, given out of order.
    words = [
        word("left", x0=0, x1=10),
        word("right", x0=100, x1=110),
        word("middle", x0=50, x1=60),
    ]

    lines = group_words_into_lines(words)

    assert len(lines) == 1
    assert [w.text for w in lines[0].words] == ["right", "middle", "left"]


def test_group_words_into_lines_separates_by_vertical_position():
    words = [
        word("line1", x0=0, x1=10, y0=0, y1=10),
        word("line2", x0=0, x1=10, y0=50, y1=60),
    ]

    lines = group_words_into_lines(words)

    assert len(lines) == 2
    assert lines[0].bbox.y0 < lines[1].bbox.y0


def test_group_words_into_lines_empty_input():
    assert group_words_into_lines([]) == []


def test_heading_detected_for_isolated_tall_short_line():
    # A tall heading line, then a gap, then normal-height paragraph lines whose
    # boxes touch (gap ~0) as adjacent same-paragraph lines do on real pages.
    heading = word("عنوان", x0=0, x1=50, y0=0, y1=30)
    para1 = word("سطر1", x0=0, x1=50, y0=100, y1=112)
    para2 = word("سطر2", x0=0, x1=50, y0=112, y1=124)
    para3 = word("سطر3", x0=0, x1=50, y0=124, y1=136)

    lines = group_words_into_lines([heading, para1, para2, para3])
    blocks = group_lines_into_blocks(lines)

    heading_blocks = [b for b in blocks if b.kind == BlockKind.HEADING]
    paragraph_blocks = [b for b in blocks if b.kind == BlockKind.PARAGRAPH]
    assert len(heading_blocks) == 1
    assert heading_blocks[0].lines[0].text == "عنوان"
    assert len(paragraph_blocks) == 1
    assert len(paragraph_blocks[0].lines) == 3


def test_group_lines_into_blocks_still_splits_when_most_gaps_are_zero():
    # Regression test: when over half the inter-line gaps are exactly 0 (typical
    # on real pages, since adjacent same-paragraph line boxes touch), the median
    # gap itself is 0 and can't be used as the reference scale — using it used to
    # make every real paragraph break vanish, collapsing the whole page into one
    # block regardless of an obvious, much larger gap like the one at y=50->60.
    lines = group_words_into_lines(
        [
            word("a", x0=0, x1=50, y0=0, y1=10),
            word("b", x0=0, x1=50, y0=10, y1=20),
            word("c", x0=0, x1=50, y0=20, y1=30),
            word("d", x0=0, x1=50, y0=30, y1=40),
            word("e", x0=0, x1=50, y0=40, y1=50),
            word("f", x0=0, x1=50, y0=60, y1=70),
            word("g", x0=0, x1=50, y0=70, y1=80),
        ]
    )

    blocks = group_lines_into_blocks(lines)

    assert len(blocks) == 2
    assert [line.text for line in blocks[0].lines] == ["a", "b", "c", "d", "e"]
    assert [line.text for line in blocks[1].lines] == ["f", "g"]


def test_analyze_end_to_end_groups_paragraph_lines_together():
    words = [
        word("سطر1", x0=0, x1=50, y0=0, y1=12),
        word("سطر2", x0=0, x1=50, y0=12, y1=24),
    ]

    blocks = analyze(words)

    assert len(blocks) == 1
    assert blocks[0].kind == BlockKind.PARAGRAPH
    assert len(blocks[0].lines) == 2
