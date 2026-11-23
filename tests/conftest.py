"""
@author: Ngoc Luong <ngocluongg0101@gmail.com>
"""

import pytest

CSV_HEADER = "filename,width,height,class,xmin,ymin,xmax,ymax\n"


def _row(name: str) -> str:
    return f"{name},100,100,military tank,10,10,50,50\n"


@pytest.fixture
def fake_dataset(tmp_path):
    """Dataset giả theo cấu trúc Mendeley.

    Ảnh a..f.jpg trên đĩa. test_labels.csv: d, e. train_labels.csv: a, b, c, d (lặp với test),
    g (thiếu file). f không có trong CSV nào; c không có file nhãn TXT.
    """
    root = tmp_path / "dataset with spaces"
    images = root / "Images"
    txt = root / "Labels" / "TXT format"
    csv_dir = root / "Labels" / "CSV Format"
    for d in (images, txt, csv_dir):
        d.mkdir(parents=True)

    for stem in "abcdef":
        (images / f"{stem}.jpg").write_bytes(b"\xff\xd8fake")
        if stem != "c":
            (txt / f"{stem}.txt").write_text("2 0.5 0.5 0.4 0.4\n")

    (csv_dir / "train_labels.csv").write_text(
        CSV_HEADER + "".join(_row(f"{s}.jpg") for s in "abcdg")
    )
    (csv_dir / "test_labels.csv").write_text(CSV_HEADER + "".join(_row(f"{s}.jpg") for s in "de"))
    return root
