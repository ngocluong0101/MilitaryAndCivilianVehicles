| # | Model | Split | imgsz | mAP50 | mAP50-95 | Lớp yếu nhất (AP50-95) | Thời gian train | Epoch | MLflow run |
|---|---|---|---|---|---|---|---|---|---|
| exp1_yolov5su_legacy_640 | yolov5su | legacy (val) | 640 | 0.792 | 0.540 | civilian aircraft (0.459) | 15 phút | 15 | `c2ad559e` |
| exp2_yolov5su_v2_640 | yolov5su | v2 (val) | 640 | 0.755 | 0.459 | civilian aircraft (0.380) | 14 phút | 15 | `adebc1d7` |
| exp3a_yolo11n_v2_640 | yolo11n | v2 (val) | 640 | 0.771 | 0.468 | civilian aircraft (0.402) | 9.0 phút | 15 | `ea2b6a7e` |
| exp3b_yolo11s_v2_640 | yolo11s | v2 (val) | 640 | 0.770 | 0.465 | civilian aircraft (0.383) | 15 phút | 15 | `96484233` |
| exp3c_yolo11n_v2_640_30ep | yolo11n | v2 (val) | 640 | 0.788 | 0.480 | civilian aircraft (0.422) | 15 phút | 30 | `dd0d8145` |
| exp6_yolo11n_v2_640_30ep_cls1 | yolo11n | v2 (val) | 640 | 0.781 | 0.469 | civilian aircraft (0.401) | 15 phút | 30 | `369051d5` |
| TEST: exp3c_yolo11n_v2_640_30ep | yolo11n | v2 (test) | 640 | 0.844 | 0.531 | military aircraft (0.430) | 15 phút | 30 | `dd0d8145` |
