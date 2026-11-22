import numpy as np
from PIL import Image

from vehicledet.error_analysis import (
    Box,
    analyze_image,
    draw_tile,
    iou_matrix,
    render_markdown,
    save_gallery,
    select_examples,
    summarize,
)

NAMES = ["truck", "car", "tank", "plane_c", "plane_m", "heli"]


def gt(cls, x1, y1, x2, y2):
    return Box(cls, (x1, y1, x2, y2))


def pred(cls, score, x1, y1, x2, y2):
    return Box(cls, (x1, y1, x2, y2), score)


def kinds(events):
    return sorted(e.kind for e in events)


def test_iou_matrix_basic_and_empty():
    a = np.array([[0, 0, 10, 10]], dtype=float)
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]], dtype=float)
    assert iou_matrix(a, b)[0].round(3).tolist() == [1.0, 0.333, 0.0]
    assert iou_matrix(np.zeros((0, 4)), b).shape == (0, 3)


def test_true_positive_and_miss():
    ev = analyze_image(
        "a", [gt(0, 0, 0, 100, 100), gt(1, 300, 300, 400, 400)], [pred(0, 0.9, 2, 2, 100, 100)]
    )
    assert kinds(ev) == ["miss", "tp"]
    miss = next(e for e in ev if e.kind == "miss")
    assert miss.gt.cls == 1 and miss.tags == []  # 100x100 px: không nhỏ, không che, không đông


def test_class_confusion_explains_the_gt_so_it_is_not_a_miss():
    ev = analyze_image("a", [gt(4, 0, 0, 100, 100)], [pred(3, 0.9, 0, 0, 100, 100)])
    assert kinds(ev) == ["class_confusion"]
    assert ev[0].gt.cls == 4 and ev[0].pred.cls == 3


def test_duplicate_prediction_of_matched_gt():
    ev = analyze_image(
        "a", [gt(0, 0, 0, 100, 100)], [pred(0, 0.9, 0, 0, 100, 100), pred(0, 0.6, 5, 5, 100, 100)]
    )
    assert kinds(ev) == ["duplicate", "tp"]


def test_localization_error_and_background_fp():
    ev = analyze_image(
        "a",
        [gt(0, 0, 0, 100, 100)],
        [pred(0, 0.9, 60, 60, 160, 160), pred(1, 0.8, 500, 500, 560, 560)],
    )
    # IoU(pred0, gt) = 1600/18400 ~ 0.087 < 0.1 -> nền; dùng box khác để có IoU ~ 0.2
    assert "background_fp" in kinds(ev)
    ev2 = analyze_image("a", [gt(0, 0, 0, 100, 100)], [pred(0, 0.9, 50, 0, 150, 100)])
    assert kinds(ev2) == ["localization"]  # IoU = 1/3


def test_other_fp_overlaps_other_class_weakly():
    ev = analyze_image("a", [gt(0, 0, 0, 100, 100)], [pred(1, 0.9, 50, 0, 150, 100)])
    assert kinds(ev) == ["miss", "other_fp"]  # GT không được "giải thích" nên vẫn là bỏ sót


def test_miss_tags_small_occluded_crowded():
    gts = [
        gt(0, 0, 0, 20, 20),  # small (400 < 1024)
        gt(0, 200, 200, 300, 300),
        gt(0, 210, 210, 310, 310),
    ]  # chồng nhau mạnh
    gts += [gt(1, 1000 + 120 * i, 0, 1100 + 120 * i, 100) for i in range(3)]  # tổng 6 GT -> crowded
    ev = [e for e in analyze_image("a", gts, []) if e.kind == "miss"]
    tags = {e.gt.xyxy: e.tags for e in ev}
    assert "small" in tags[(0, 0, 20, 20)]
    assert "occluded" in tags[(200, 200, 300, 300)] and "occluded" in tags[(210, 210, 310, 310)]
    assert all("crowded" in t for t in tags.values())


def test_summary_counts_pairs_and_suspect_labels():
    events = []
    events += analyze_image("a", [gt(4, 0, 0, 100, 100)], [pred(3, 0.95, 0, 0, 100, 100)])
    events += analyze_image("b", [], [pred(2, 0.9, 0, 0, 50, 50), pred(2, 0.3, 200, 200, 250, 250)])
    events += analyze_image("c", [gt(0, 0, 0, 100, 100)], [pred(0, 0.9, 0, 0, 100, 100)])
    s = summarize(events, 3, NAMES)
    assert s["counts"] == {
        "tp": 1,
        "class_confusion": 1,
        "localization": 0,
        "duplicate": 0,
        "background_fp": 2,
        "other_fp": 0,
        "miss": 0,
    }
    assert s["confusion_pairs"] == [{"true": "plane_m", "predicted": "plane_c", "count": 1}]
    reasons = {(x["image"], x["reason"]) for x in s["suspect_labels"]}
    assert ("a", "high-confidence class disagreement") in reasons  # 0.95 >= 0.8
    assert ("b", "high-confidence detection without label") in reasons  # 0.9 >= 0.7
    assert len(s["suspect_labels"]) == 2  # dự đoán 0.3 không bị coi là đáng ngờ
    assert s["per_class"]["plane_m"]["confused_as_other"] == 1


def test_select_examples_is_deterministic_and_one_per_image():
    events = []
    for i in range(10):
        events += analyze_image(f"img{i}", [gt(0, 0, 0, 20, 20), gt(0, 50, 50, 70, 70)], [])
    a = select_examples(events, per_category=3, seed=1)
    b = select_examples(events, per_category=3, seed=1)
    assert [e.image for e in a["miss_small"]] == [e.image for e in b["miss_small"]]
    assert len({e.image for e in a["miss_small"]}) == len(a["miss_small"]) == 3
    assert [e.image for e in a["miss_small"]] != [
        e.image for e in select_examples(events, 3, seed=2)["miss_small"]
    ]


def test_gallery_and_markdown(tmp_path):
    img_path = tmp_path / "a.jpg"
    Image.new("RGB", (200, 150), "gray").save(img_path)
    g, p = [gt(4, 10, 10, 100, 100)], [pred(3, 0.95, 10, 10, 100, 100)]
    events = analyze_image("a.jpg", g, p)
    chosen = select_examples(events)
    listing = save_gallery(
        chosen, {"a.jpg": img_path}, {"a.jpg": g}, {"a.jpg": p}, NAMES, tmp_path / "out"
    )
    assert (tmp_path / "out" / "class_confusion.jpg").exists()
    assert (tmp_path / "out" / "suspect_label.jpg").exists()  # nhầm lớp điểm 0,95 >= 0,8
    assert listing["class_confusion"] == ["a.jpg"]
    md = render_markdown(summarize(events, 1, NAMES), listing, {"conf": 0.25})
    assert "plane_m | plane_c | 1" in md and "class_confusion.jpg" in md and "KHÔNG sửa" in md


def test_draw_tile_scales_to_height():
    tile = draw_tile(
        Image.new("RGB", (400, 200), "white"),
        [gt(0, 0, 0, 50, 50)],
        [pred(1, 0.5, 0, 0, 40, 40)],
        NAMES,
        "cap",
        height=100,
    )
    assert tile.size == (200, 114)
