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

이 저장소의 빠른 accuracy check는 `cityscapes_small` 폴더를 사용합니다. `scripts/download_cityspace_small.py`는 Hugging Face의 `Chris1/cityscapes` validation split을 streaming으로 읽어서 500장을 아래 구조로 저장합니다.

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
python scripts/download_cityspace_small.py
```

스크립트 기본값은 `OUT_DIR = "cityscapes_small"`, `SPLIT = "validation"`, `NUM_SAMPLES = 500`입니다. 다른 개수나 split이 필요하면 스크립트 상단 값을 바꾼 뒤 다시 실행하세요. 저장된 mask는 Cityscapes label ID 형식이므로, 평가 스크립트의 기본값인 `--label-format auto`가 PIDNet의 19-class trainId로 변환합니다.


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

| ONNX | mIoU | Pixel Accuracy | Mean Accuracy | 비고 |
|---|---:|---:|---:|---|
| `pidnet_s_cityscapes_val.onnx` | 76.56% | 95.33% | 84.43% | 공정한 validation 기준 |
| `pidnet_s_cityscapes_test.onnx` | 84.84% | 96.78% | 91.54% | 데모/시각화용 권장 |

`PIDNet_S_Cityscapes_test.pt`는 train+val로 학습된 test 제출용 가중치일 수 있으므로, `cityscapes_small`이 val 이미지 기반이면 수치가 높게 나올 수 있습니다. 따라서 README나 리포트에서 정확도를 말할 때는 `pidnet_s_cityscapes_val.onnx` 결과를 대표값으로 쓰고, 데모 앱은 `pidnet_s_cityscapes_test.onnx`를 쓰는 것을 권장합니다.

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

출력 JSON에는 `mIoU`, `pixel_accuracy`, `mean_accuracy`, 클래스별 IoU, FPS가 포함됩니다. ONNX 입력 크기가 고정되어 있으면 스크립트가 자동으로 읽고, 동적 입력이면 원본 Cityscapes 크기인 `1024x2048`을 그대로 넣습니다. 입력 크기를 직접 맞춰야 하는 경우:

```bash
python scripts/eval_cityscapes_onnx.py \
  --model pidnet_s_cityscapes_val.onnx \
  --dataset-root cityscapes_small \
  --input-size 1024 2048
```

## 참고

- PIDNet 공식 README는 Cityscapes 데이터를 `data/cityscapes`에 풀고 `tools/eval.py`로 Cityscapes val 평가를 수행하는 흐름을 안내합니다.
- Cityscapes 공식 스크립트는 `csDownload`, `csCreateTrainIdLabelImgs`, 평가 도구를 제공합니다.
- 이 스크립트는 `_gtFine_labelTrainIds.png`가 있으면 그대로 사용하고, 없으면 `_gtFine_labelIds.png`를 PIDNet/Cityscapes 19-class trainId로 변환합니다.

## PIDNet-S Cityscapes ONNX Export

공식 PIDNet-S Cityscapes checkpoint가 있으면 다음처럼 ONNX를 다시 만들 수 있습니다. 공식 repo는 `PIDNet_S_Cityscapes_val.pt`, `PIDNet_S_Cityscapes_test.pt`를 Cityscapes PIDNet-S 가중치로 제공합니다.

이 저장소의 export 스크립트는 입력 더미 텐서를 `1x3x1024x2048`로 만들고 `dynamic_axes`를 쓰지 않으므로 ONNX 입력/출력의 batch size가 `1`로 고정됩니다. 다른 해상도로 고정하고 싶으면 `--height`, `--width`만 바꿔서 다시 export하세요.

```bash
git clone https://github.com/XuJiacong/PIDNet external/PIDNet

# PyTorch와 ONNX export 의존성 설치
python -m pip install torch onnx
```

현재 디렉터리에 내려받은 공식 `.pt` 파일이 있다면 그대로 지정해서 export할 수 있습니다. 아래 google drive 에서 다운로드 가능합니다.
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

ONNX input shape에서 batch가 `1`로 고정됐는지 확인:

```bash
python - <<'PY'
import onnx

model = onnx.load("pidnet_s_cityscapes_val.onnx")
for value in list(model.graph.input) + list(model.graph.output):
    dims = [
        dim.dim_value if dim.dim_value else dim.dim_param
        for dim in value.type.tensor_type.shape.dim
    ]
    print(value.name, dims)
PY
```

생성된 ONNX를 `cityscapes_small`로 확인:

```bash
python scripts/eval_cityscapes_onnx.py \
  --model pidnet_s_cityscapes_val.onnx \
  --dataset-root cityscapes_small \
  --save-json metrics/pidnet_s_cityscapes_val_small.json
```
