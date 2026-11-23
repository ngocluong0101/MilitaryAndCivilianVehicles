"""
@author: Ngoc Luong <ngocluongg0101@gmail.com>
"""

import pytest
from PIL import Image

from vehicledet.data.audit import box_size_category, run_audit, scan_dataset
from vehicledet.data.labels import Box

CSV_HEADER = "filename,width,height,class,xmin,ymin,xmax,ymax\n"


@pytest.fixture
def real_dataset(tmp_path):
    """Dataset giả có ảnh thật 100x100.

    a khớp CSV | b lệch số bbox | c nền, ngoài CSV | d nhãn lỗi | e ảnh hỏng | f thiếu TXT
    | g chỉ trong CSV
    """
    root = tmp_path / "ds"
    images, txt, csvd = (
        root / "Images",
        root / "Labels" / "TXT format",
        root / "Labels" / "CSV Format",
    )
    for d in (images, txt, csvd):
        d.mkdir(parents=True)
    for s in "abcdf":
        Image.new("RGB", (100, 100), "gray").save(images / f"{s}.jpg")
    (images / "e.jpg").write_bytes(b"not an image")
    txt_content = {
        "a": "2 0.5 0.5 0.5 0.5\n",
        "b": "0 0.3 0.3 0.2 0.2\n1 0.7 0.7 0.2 0.2\n",
        "c": "",
        "d": "9 0.5 0.5 0.5 0.5\n",
        "e": "1 0.5 0.5 0.5 0.5\n",
    }
    for k, v in txt_content.items():
        (txt / f"{k}.txt").write_text(v)
    rows = [
        "a.jpg,100,100,military tank,25,25,75,75",
        "b.jpg,100,100,military truck,20,20,40,40",
        "g.jpg,100,100,civilian car,25,25,75,75",
    ]
    (csvd / "train_labels.csv").write_text(CSV_HEADER + "\n".join(rows) + "\n")
    (csvd / "test_labels.csv").write_text(CSV_HEADER + rows[0] + "\n")
    return root


def test_scan_flags_unreadable_and_label_errors(real_dataset):
    rec = scan_dataset(real_dataset)
    assert not rec["e.jpg"].readable and rec["e.jpg"].read_error
    assert rec["d.jpg"].has_error and rec["d.jpg"].boxes == []
    assert rec["a.jpg"].size == (100, 100) and len(rec["a.jpg"].boxes) == 1
    assert not rec["f.jpg"].has_txt


def test_run_audit_counts(real_dataset):
    a = run_audit(real_dataset)
    assert a["images"]["total"] == 6
    assert a["images"]["unreadable"] == 1
    assert a["images"]["no_txt_file"] == 1
    assert a["images"]["with_label_error"] == 1
    assert a["images"]["txt_empty_or_no_valid_box"] == 2  # c (rỗng) và d (bbox duy nhất lỗi)
    assert a["label_problems"]["counts"]["class_out_of_range"] == 1
    assert a["boxes"]["per_class"]["military tank"] == 1
    assert a["boxes"]["per_class"]["military truck"] == 1


def test_run_audit_csv_vs_txt(real_dataset):
    st = run_audit(real_dataset)["csv_vs_txt"]["status_counts"]
    assert st["match"] == 1  # a
    assert st["count_mismatch"] == 1  # b: CSV 1 bbox, TXT 2 bbox
    assert st["csv_image_missing_on_disk"] == 1  # g
    assert st["txt_only_not_in_csv_empty"] == 2  # c (rỗng) và d (bbox duy nhất bị lỗi)
    assert st["no_txt_and_not_in_csv"] == 1  # f
    assert st["txt_only_not_in_csv_labeled"] == 1  # e (có nhãn, không có trong CSV)


def test_csv_missing_image_matched_to_renamed_file(real_dataset):
    # g.jpg (CSV) có cùng bbox/lớp với e.jpg (TXT, không có trong CSV) -> nghi bị đổi tên
    r = run_audit(real_dataset)["csv_vs_txt"]
    assert r["csv_missing_probably_renamed"] == {"g.jpg": "e.jpg"}


def test_box_size_category_uses_pixel_area():
    size = (640, 480)
    assert box_size_category(Box(0, 0.5, 0.5, 20 / 640, 20 / 480), size) == "small"
    assert box_size_category(Box(0, 0.5, 0.5, 50 / 640, 50 / 480), size) == "medium"
    assert box_size_category(Box(0, 0.5, 0.5, 200 / 640, 200 / 480), size) == "large"
