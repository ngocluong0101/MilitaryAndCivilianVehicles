"""
@author: Ngoc Luong <ngocluongg0101@gmail.com>
"""

from vehicledet.data.labels import Box, compare_boxes, parse_yolo_txt, read_csv_boxes

CLASSES = {"military truck": 0, "civilian car": 1, "military tank": 2}


def parse(tmp_path, text, size=None):
    p = tmp_path / "x.txt"
    p.write_text(text)
    return parse_yolo_txt(p, 6, size)


def kinds(problems):
    return sorted(p.kind for p in problems)


def test_valid_file(tmp_path):
    boxes, problems = parse(tmp_path, "2 0.5 0.5 0.4 0.4\n\n0 0.2 0.2 0.1 0.1\n")
    assert [b.cls for b in boxes] == [2, 0]
    assert problems == []


def test_class_out_of_range_and_non_integer(tmp_path):
    boxes, problems = parse(
        tmp_path, "6 0.5 0.5 0.4 0.4\n-1 0.5 0.5 0.4 0.4\n1.5 0.5 0.5 0.4 0.4\n"
    )
    assert boxes == []
    assert kinds(problems) == ["class_out_of_range"] * 3


def test_coordinates_out_of_range(tmp_path):
    boxes, problems = parse(tmp_path, "0 1.2 0.5 0.4 0.4\n0 0.5 -0.1 0.4 0.4\n")
    assert boxes == [] and kinds(problems) == ["coord_out_of_range"] * 2


def test_zero_size_box(tmp_path):
    boxes, problems = parse(tmp_path, "0 0.5 0.5 0 0.4\n0 0.5 0.5 0.4 0.0\n")
    assert boxes == [] and kinds(problems) == ["zero_size"] * 2


def test_bad_format(tmp_path):
    boxes, problems = parse(tmp_path, "0 0.5 0.5 0.4\nabc 0.5 0.5 0.4 0.4\n0 0.5 0.5 0.4 0.4 0.9\n")
    assert boxes == [] and kinds(problems) == ["bad_format"] * 3
    assert all(p.is_error for p in problems)


def test_warnings_keep_the_box(tmp_path):
    text = "0 0.5 0.5 0.4 0.4\n0 0.5 0.5 0.4 0.4\n1 0.95 0.5 0.2 0.2\n2 0.5 0.5 0.01 0.01\n"
    boxes, problems = parse(tmp_path, text, size=(100, 100))
    assert len(boxes) == 4
    assert kinds(problems) == ["box_exceeds_image", "duplicate_box", "tiny_box"]
    assert not any(p.is_error for p in problems)


def test_compare_boxes_categories():
    a = Box(0, 0.5, 0.5, 0.4, 0.4)
    assert compare_boxes([a], [a]) == "match"
    assert compare_boxes([a], [Box(0, 0.5, 0.5, 0.4, 0.4), a]) == "count_mismatch"
    assert compare_boxes([a], [Box(1, 0.5, 0.5, 0.4, 0.4)]) == "class_mismatch"
    assert compare_boxes([a], [Box(0, 0.55, 0.5, 0.4, 0.4)]) == "minor_box_diff"  # IoU ~0.78
    assert compare_boxes([a], [Box(0, 0.9, 0.9, 0.1, 0.1)]) == "box_mismatch"


def test_compare_boxes_is_order_independent():
    a, b = Box(0, 0.2, 0.2, 0.2, 0.2), Box(1, 0.7, 0.7, 0.2, 0.2)
    assert compare_boxes([a, b], [b, a]) == "match"


def test_read_csv_boxes_normalizes_and_dedupes(tmp_path):
    header = "filename,width,height,class,xmin,ymin,xmax,ymax\n"
    row = "a.jpg,200,100,military tank,50,20,150,80\n"
    (tmp_path / "t1.csv").write_text(header + row + "b.jpg,10,10,spaceship,0,0,5,5\n")
    (tmp_path / "t2.csv").write_text(header + row)  # dòng lặp giữa hai file
    boxes, sizes, unknown = read_csv_boxes([tmp_path / "t1.csv", tmp_path / "t2.csv"], CLASSES)
    assert boxes["a.jpg"] == [Box(2, 0.5, 0.5, 0.5, 0.6)]
    assert sizes["a.jpg"] == {(200, 100)}
    assert boxes["b.jpg"] == [] and unknown == {"spaceship": 1}
