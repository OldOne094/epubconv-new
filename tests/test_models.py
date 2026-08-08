from epubconv.models import BBox, Block, BlockKind, Line, PageResult, PageStatus, Word


def make_word(text="مرحبا", conf=0.9, low=False):
    return Word(text=text, bbox=BBox(0, 0, 10, 10), confidence=conf, low_confidence=low)


def test_bbox_dimensions():
    box = BBox(x0=1, y0=2, x1=5, y1=9)
    assert box.width == 4
    assert box.height == 7


def test_line_text_and_confidence():
    line = Line(words=(make_word("أ", 0.8), make_word("ب", 0.6)), bbox=BBox(0, 0, 20, 10))
    assert line.text == "أ ب"
    assert line.mean_confidence == 0.7


def test_block_roundtrip_through_dict():
    line = Line(words=(make_word(low=True),), bbox=BBox(0, 0, 10, 10))
    block = Block(kind=BlockKind.HEADING, lines=(line,), level=2)

    restored = Block.from_dict(block.to_dict())

    assert restored.kind == BlockKind.HEADING
    assert restored.level == 2
    assert restored.lines[0].words[0].text == "مرحبا"
    assert restored.lines[0].words[0].low_confidence is True


def test_page_result_roundtrip_and_confidence_ratio():
    block = Block(kind=BlockKind.PARAGRAPH, lines=(Line(words=(make_word(),), bbox=BBox(0, 0, 1, 1)),))
    result = PageResult(
        index=3, status=PageStatus.OK, blocks=(block,), low_confidence_words=1, total_words=4
    )

    restored = PageResult.from_dict(result.to_dict())

    assert restored.index == 3
    assert restored.status == PageStatus.OK
    assert restored.confidence_ratio == 0.75
    assert restored.blocks[0].lines[0].words[0].text == "مرحبا"


def test_page_result_confidence_ratio_with_no_words():
    result = PageResult(index=0, status=PageStatus.OK)
    assert result.confidence_ratio == 1.0
