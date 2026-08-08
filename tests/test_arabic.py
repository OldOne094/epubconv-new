from epubconv.arabic import clean_block, clean_blocks, clean_line, flag_footnote_markers, normalize_text
from epubconv.models import BBox, Block, BlockKind, Line, Word


def make_word(text, low=False):
    return Word(text=text, bbox=BBox(0, 0, 10, 10), confidence=0.9, low_confidence=low)


def test_normalize_text_removes_tatweel():
    assert normalize_text("مرحـــبا") == "مرحبا"


def test_normalize_text_collapses_whitespace_and_strips():
    assert normalize_text("  مرحبا    بالعالم  ") == "مرحبا بالعالم"


def test_normalize_text_removes_control_chars_but_keeps_arabic():
    assert normalize_text("مرحبا\x00\x0bبالعالم") == "مرحبابالعالم"


def test_normalize_text_does_not_alter_valid_diacritics():
    text = "مَرْحَبًا"
    assert normalize_text(text) == text


def test_clean_block_drops_words_that_become_empty():
    line = Line(words=(make_word("ـــ"), make_word("مرحبا")), bbox=BBox(0, 0, 10, 10))
    block = Block(kind=BlockKind.PARAGRAPH, lines=(line,))

    cleaned = clean_block(block)

    assert len(cleaned.lines[0].words) == 1
    assert cleaned.lines[0].words[0].text == "مرحبا"


def test_clean_block_preserves_low_confidence_flag():
    line = Line(words=(make_word("مرحـبا", low=True),), bbox=BBox(0, 0, 10, 10))
    block = Block(kind=BlockKind.PARAGRAPH, lines=(line,))

    cleaned = clean_block(block)

    assert cleaned.lines[0].words[0].low_confidence is True
    assert cleaned.lines[0].words[0].text == "مرحبا"


def test_flag_footnote_markers_splits_a_bare_digit_surrounded_by_spaces():
    # Real case found on a real page: a superscript footnote marker fused
    # into the recognized line, e.g. "للمعجزة ١ كانت هنا".
    word = make_word("للمعجزة ١ كانت هنا")

    pieces = flag_footnote_markers(word)

    assert [(p.text, p.low_confidence) for p in pieces] == [
        ("للمعجزة", False),
        ("١", True),
        ("كانت هنا", False),
    ]


def test_flag_footnote_markers_splits_a_parenthesized_marker_glued_to_words():
    # Real case found on a real page: "(1)" with no whitespace on either side.
    word = make_word("شو(1)غير")

    pieces = flag_footnote_markers(word)

    assert [(p.text, p.low_confidence) for p in pieces] == [
        ("شو", False),
        ("(1)", True),
        ("غير", False),
    ]


def test_flag_footnote_markers_flags_a_standalone_marker_word():
    word = make_word("(1)")

    pieces = flag_footnote_markers(word)

    assert len(pieces) == 1
    assert pieces[0].text == "(1)"
    assert pieces[0].low_confidence is True


def test_flag_footnote_markers_ignores_a_multi_digit_page_number():
    word = make_word("39|")

    pieces = flag_footnote_markers(word)

    assert pieces == [word]


def test_flag_footnote_markers_never_alters_the_marker_text_itself():
    # Whatever the OCR actually read for the marker must survive untouched -
    # flagging, never correcting.
    word = make_word("مقدمة ٥ نهاية")

    pieces = flag_footnote_markers(word)

    assert pieces[1].text == "٥"


def test_flag_footnote_markers_preserves_confidence_on_the_non_marker_pieces():
    word = make_word("نص أول ١ نص ثاني", low=True)

    pieces = flag_footnote_markers(word)

    assert pieces[0].low_confidence is True  # carried from the original word
    assert pieces[1].low_confidence is True  # the marker itself, always flagged
    assert pieces[2].low_confidence is True


def test_clean_line_flattens_split_words_into_the_line():
    line = Line(words=(make_word("للمعجزة ١ كانت هنا"),), bbox=BBox(0, 0, 10, 10))

    cleaned = clean_line(line)

    assert [w.text for w in cleaned.words] == ["للمعجزة", "١", "كانت هنا"]


def test_clean_blocks_drops_blocks_left_with_no_lines():
    empty_line = Line(words=(make_word("ـ"),), bbox=BBox(0, 0, 10, 10))
    empty_block = Block(kind=BlockKind.PARAGRAPH, lines=(empty_line,))
    good_line = Line(words=(make_word("نص"),), bbox=BBox(0, 0, 10, 10))
    good_block = Block(kind=BlockKind.PARAGRAPH, lines=(good_line,))

    result = clean_blocks([empty_block, good_block])

    assert len(result) == 1
    assert result[0].lines[0].words[0].text == "نص"
