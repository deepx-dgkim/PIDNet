# PIDNet-S Cityscapes Accuracy Check

이 저장소는 공식 PIDNet-S Cityscapes checkpoint에서 export한 ONNX를 Cityscapes 데이터로 빠르게 검증하기 위한 최소 구성입니다. 원본 구현은 [XuJiacong/PIDNet](https://github.com/XuJiacong/PIDNet)을 기준으로 했고, Cityscapes 전처리는 PIDNet의 ImageNet mean/std 및 19개 trainId 매핑을 따릅니다.

## 설치

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
python -m pip install datasets
```

GPU에서 ONNX Runtime을 쓰려면 CPU 패키지 대신 환경에 맞는 `onnxruntime-gpu`를 설치하세요.

```bash
python -m pip uninstall -y onnxruntime
python -m pip install onnxruntime-gpu
```

## Cityscapes 데이터 준비

이 저장소의 빠른 accuracy check는 `cityscapes_small` 폴더를 사용합니다. `scripts/download_cityscapes_small.py`는 Hugging Face의 `Chris1/cityscapes` validation split을 streaming으로 읽어서 500장을 아래 구조로 저장합니다.

```text
cityscapes_small/
  images/
    000000.png
    ...
  masks/
    000000_mask.png
    ...
```

처음 한 번만 다음처럼 준비하면 됩니다.

```bash
python scripts/download_cityscapes_small.py
```

스크립트 기본값은 `OUT_DIR = "cityscapes_small"`, `SPLIT = "validation"`, `NUM_SAMPLES = 500`입니다. 다른 개수나 split이 필요하면 스크립트 상단 값을 바꾼 뒤 다시 실행하세요. 저장된 mask는 Cityscapes label ID 형식이므로, 평가 스크립트의 기본값인 `--label-format auto`가 PIDNet의 19-class trainId로 변환합니다.


## PIDNet-S Cityscapes ONNX Export

공식 PIDNet-S Cityscapes checkpoint가 있으면 다음처럼 ONNX를 다시 만들 수 있습니다. 공식 repo는 `PIDNet_S_Cityscapes_val.pt`, `PIDNet_S_Cityscapes_test.pt`를 Cityscapes PIDNet-S 가중치로 제공합니다.

이 저장소의 export 스크립트는 입력 더미 텐서를 `1x3x1024x2048`로 만들고 `dynamic_axes`를 쓰지 않으므로 ONNX 입력/출력의 batch size가 `1`로 고정됩니다. 다른 해상도로 고정하고 싶으면 `--height`, `--width`만 바꿔서 다시 export하세요.

```bash
git clone https://github.com/XuJiacong/PIDNet external/PIDNet

# PyTorch와 ONNX export 의존성 설치
python -m pip install torch onnx
```

현재 디렉터리에 내려받은 공식 `.pt` 파일이 있다면 그대로 지정해서 export할 수 있습니다. 아래 google drive 에서 다운로드 가능합니다.
아래 google driver 에서 `PIDNet_S_Cityscapes_val.pt` 와 `PIDNet_S_Cityscapes_test.pt` 를 다운로드 합니다. 
https://drive.google.com/drive/folders/0BySIOtxxULinfjlGdGFiT3NQVUdLVDBxWnhhTjB4VXNBRkFOa281WHlkektYY2VBcWVZb1k?resourcekey=0-w0JIXUekD-FCW-Rm1Z-HfQ&usp=sharing

```bash
# val checkpoint -> ONNX
python scripts/export_pidnet_s_cityscapes_onnx.py \
  --pidnet-repo external/PIDNet \
  --checkpoint PIDNet_S_Cityscapes_val.pt \
  --output pidnet_s_cityscapes_val.onnx

# test checkpoint -> ONNX
python scripts/export_pidnet_s_cityscapes_onnx.py \
  --pidnet-repo external/PIDNet \
  --checkpoint PIDNet_S_Cityscapes_test.pt \
  --output pidnet_s_cityscapes_test.onnx
```


## cityscapes_small Accuracy 확인

이미 `cityscapes_small` 폴더에 500장 데이터가 준비되어 있으면 바로 실행할 수 있습니다. 스크립트는 기본적으로 아래 구조를 읽습니다.

```text
cityscapes_small/
  images/
    000000.png
    ...
  masks/
    000000_mask.png
    ...
```

공식 PIDNet-S Cityscapes val checkpoint에서 export한 ONNX는 공정한 validation 지표 확인용으로 사용합니다.

```bash
python scripts/eval_cityscapes_onnx.py \
  --model pidnet_s_cityscapes_val.onnx \
  --dataset-root cityscapes_small \
  --save-json metrics/pidnet_s_cityscapes_val_small.json
```

공식 PIDNet-S Cityscapes test checkpoint에서 export한 ONNX는 데모/시각화 품질 확인용으로 사용합니다.

```bash
python scripts/eval_cityscapes_onnx.py \
  --model pidnet_s_cityscapes_test.onnx \
  --dataset-root cityscapes_small \
  --save-json metrics/pidnet_s_cityscapes_test_small.json
```

현재 `cityscapes_small` 500장 기준 측정 결과:

| ONNX | mIoU | Pixel Accuracy | Mean Accuracy |
|---|---:|---:|---:|
| `pidnet_s_cityscapes_val.onnx` | 76.56% | 95.33% | 84.43% |
| `pidnet_s_cityscapes_test.onnx` | 84.84% | 96.78% | 91.54% |

`PIDNet_S_Cityscapes_test.pt`는 train+val로 학습된 test 제출용 가중치일 수 있으므로, `cityscapes_small`이 val 이미지 기반이면 수치가 높게 나올 수 있습니다.

일부만 빠르게 확인:

```bash
python scripts/eval_cityscapes_onnx.py \
  --model pidnet_s_cityscapes_val.onnx \
  --dataset-root cityscapes_small \
  --limit 20
```

`cityscapes_small/masks`의 값은 Cityscapes 원본 label ID 형식이라 기본값인 `--label-format auto`가 label ID를 PIDNet의 19-class trainId로 변환합니다. 만약 다른 데이터셋에서 이미 trainId `0..18, 255` 형태의 마스크를 쓰면 다음 옵션을 추가하세요.

```bash
python scripts/eval_cityscapes_onnx.py \
  --model pidnet_s_cityscapes_val.onnx \
  --dataset-root cityscapes_small \
  --label-format trainid
```

## ONNX Demo

`demo_onnx.py`는 `pidnet_s_cityscapes_val.onnx`를 기본으로 로드하고, 입력으로 단일 이미지 파일, 이미지 폴더, 비디오 파일을 받을 수 있습니다. 추론 결과는 OpenCV `imshow` 창에 실시간으로 표시하며 따로 저장하지 않습니다. 이미지 폴더나 비디오 입력에서는 이전 결과를 지우고 새 프레임의 segmentation 결과만 계속 갱신합니다.

```bash
# 단일 이미지
python scripts/demo_onnx.py external/PIDNet/samples/frankfurt_000000_002196_leftImg8bit.png

# 이미지 폴더
python scripts/demo_onnx.py cityscapes_small/images

# 비디오 파일
python scripts/demo_onnx.py input_video.mp4
```

기본 화면은 원본 이미지 위에 Cityscapes segmentation 색상을 overlay합니다. mask만 보거나 원본/overlay를 나란히 보고 싶으면 다음 옵션을 사용할 수 있습니다.

```bash
python scripts/demo_onnx.py cityscapes_small/images --view mask
python scripts/demo_onnx.py input_video.mp4 --view side-by-side
```

다른 ONNX를 사용하려면 `--model`을 지정하세요.

```bash
python scripts/demo_onnx.py cityscapes_small/images \
  --model pidnet_s_cityscapes_test.onnx
```

실행 중 `q` 또는 `Esc`를 누르면 종료됩니다. `imshow`를 사용하므로 GUI 표시가 가능한 환경이 필요합니다.

## DXNN Accuracy 확인

DXNN 모델은 `dxnn/` 폴더의 `.dxnn` 파일을 사용합니다. 실행 환경에는 DEEPX DXRT Python 패키지의 `dx_engine` 모듈이 설치되어 있어야 합니다.

```bash
python -c "from dx_engine import InferenceEngine; print('dx_engine ok')"
```

`cityscapes_small` 데이터가 준비되어 있으면 다음처럼 DXNN accuracy를 확인할 수 있습니다.

```bash
# val DXNN
python scripts/eval_cityscapes_dxnn.py \
  --model dxnn/pidnet_s_cityscapes_val.dxnn \
  --dataset-root cityscapes_small \
  --save-json metrics/pidnet_s_cityscapes_val_dxnn_small.json

# test DXNN
python scripts/eval_cityscapes_dxnn.py \
  --model dxnn/pidnet_s_cityscapes_test.dxnn \
  --dataset-root cityscapes_small \
  --input-color rgb \
  --save-json metrics/pidnet_s_cityscapes_test_dxnn_small.json
```

일부 이미지만 빠르게 확인하려면 `--limit`을 추가하세요.

```bash
python scripts/eval_cityscapes_dxnn.py \
  --model dxnn/pidnet_s_cityscapes_val.dxnn \
  --dataset-root cityscapes_small \
  --limit 20
```

현재 `cityscapes_small` 500장 기준 DXNN 측정 결과:

| DXNN | mIoU | Pixel Accuracy | Mean Accuracy |
|---|---:|---:|---:|
| `pidnet_s_cityscapes_val.dxnn` | 29.53% | 75.54% | 37.84% |
| `pidnet_s_cityscapes_test.dxnn` | 40.00% | 79.55% | 46.69% |
| `pidnet_s_cityscapes_val_calib100.dxnn` | 29.53% | 75.65% | 37.82% |

예측 trainId mask를 PNG로 저장하려면 `--save-preds`를 지정합니다.

```bash
python scripts/eval_cityscapes_dxnn.py \
  --model dxnn/pidnet_s_cityscapes_val.dxnn \
  --dataset-root cityscapes_small \
  --save-preds outputs/dxnn_val_preds
```

## DXNN Demo

`demo_dxnn.py`는 `dxnn/pidnet_s_cityscapes_val.dxnn`을 기본으로 로드하고, ONNX demo와 동일하게 단일 이미지 파일, 이미지 폴더, 비디오 파일을 입력으로 받을 수 있습니다. 추론 결과는 OpenCV `imshow` 창에 실시간으로 표시합니다.

```bash
# 단일 이미지
python scripts/demo_dxnn.py external/PIDNet/samples/frankfurt_000000_002196_leftImg8bit.png

# 이미지 폴더
python scripts/demo_dxnn.py cityscapes_small/images

# 비디오 파일
python scripts/demo_dxnn.py input_video.mp4
```

화면 출력 방식은 `--view`로 바꿀 수 있습니다.

```bash
python scripts/demo_dxnn.py cityscapes_small/images --view mask
python scripts/demo_dxnn.py input_video.mp4 --view side-by-side
```

다른 DXNN 모델을 사용할 때는 `--model`을 지정하세요. `pidnet_s_cityscapes_test.dxnn`은 기존 측정과 동일하게 `--input-color rgb`를 함께 지정합니다.

```bash
python scripts/demo_dxnn.py cityscapes_small/images \
  --model dxnn/pidnet_s_cityscapes_test.dxnn \
  --input-color rgb
```

실행 중 `q` 또는 `Esc`를 누르면 종료됩니다. `imshow`를 사용하므로 GUI 표시가 가능한 환경이 필요합니다.
