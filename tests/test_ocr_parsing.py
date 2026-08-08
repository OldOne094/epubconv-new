from epubconv.ocr import parse_predict_results


def test_parse_predict_results_with_rec_boxes():
    fake_result = [
        {
            "res": {
                "rec_texts": ["مرحبا", "بالعالم"],
                "rec_scores": [0.95, 0.4],
                "rec_boxes": [[0, 0, 10, 10], [20, 0, 30, 10]],
            }
        }
    ]

    words = parse_predict_results(fake_result, threshold=0.7)

    assert [w.text for w in words] == ["مرحبا", "بالعالم"]
    assert words[0].low_confidence is False
    assert words[1].low_confidence is True


def test_parse_predict_results_falls_back_to_polys():
    fake_result = [
        {
            "res": {
                "rec_texts": ["نص"],
                "rec_scores": [0.9],
                "rec_polys": [[[0, 0], [10, 0], [10, 20], [0, 20]]],
            }
        }
    ]

    words = parse_predict_results(fake_result)

    assert len(words) == 1
    assert words[0].bbox.x0 == 0
    assert words[0].bbox.x1 == 10
    assert words[0].bbox.y1 == 20


def test_parse_predict_results_skips_blank_text():
    fake_result = [
        {"res": {"rec_texts": ["", "  ", "نص"], "rec_scores": [0.9, 0.9, 0.9], "rec_boxes": [
            [0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1]
        ]}}
    ]

    words = parse_predict_results(fake_result)

    assert len(words) == 1
    assert words[0].text == "نص"


def test_parse_predict_results_handles_object_with_json_attribute():
    class FakePaddleResult:
        json = {"res": {"rec_texts": ["كلمة"], "rec_scores": [0.99], "rec_boxes": [[0, 0, 5, 5]]}}

    words = parse_predict_results([FakePaddleResult()])

    assert words[0].text == "كلمة"
