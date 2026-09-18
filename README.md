# PIDNet on DEEPX NPU — Evaluation Example

Example project for evaluating [PIDNet](https://github.com/XuJiacong/PIDNet) on a
DEEPX NPU: export an official PIDNet-S Cityscapes checkpoint to ONNX with ArgMax /
Softmax baked into the graph (so the NPU does that work, not the host), then run
demo and accuracy scripts for both the ONNX model and the DXNN model compiled from it.

## Model provenance

- Pretrained model: official [XuJiacong/PIDNet](https://github.com/XuJiacong/PIDNet)
  repo, PIDNet-S Cityscapes checkpoint (Google Drive link in that repo's README).
- **Use `PIDNet_S_Cityscapes_test.pt` or `PIDNet_S_Cityscapes_val.pt` — not
  `PIDNet_S_Simple_Cityscapes_test.pt`.** The "Simple" checkpoint is missing the
  `pag3`/`pag4`/`dfm` fusion modules (~8% of the model). Loading it leaves those
  layers randomly initialized and produces plausible-looking but wrong segmentation,
  with no error or warning. After loading any checkpoint, verify
  `matched == len(model.state_dict())`.
- `models/` vendors only the 3 files needed to load that checkpoint
  (`__init__.py`, `pidnet.py`, `model_utils.py`), copied from the official repo
  (MIT license, see `models/LICENSE`).

## Repository layout

```text
PIDNet/
├── export_argmax_softmax_onnx.py # Export: checkpoint -> ONNX with ArgMax + Softmax baked in
├── models/                      # PIDNet-S model definition, vendored from XuJiacong/PIDNet
│   ├── __init__.py
│   ├── pidnet.py
│   ├── model_utils.py
│   └── LICENSE                  # MIT license covering these three vendored files
├── pidnet_demo_onnx.py          # ONNX demo (image / image folder / video)
├── pidnet_demo_dxnn.py          # DXNN demo (video, async NPU pipeline)
├── download_cityscapes_small.py # Fetches a small Cityscapes validation subset
├── eval_cityscapes_onnx.py      # mIoU / pixel accuracy / mean accuracy for an ONNX model
├── eval_cityscapes_dxnn.py      # Same, for a compiled DXNN model
└── compare_onnx_dxnn.py         # ONNX vs. DXNN output consistency (cosine similarity)
```

## 1. Install dependencies

```bash
python3 -m venv venv && source venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

`torch`/`onnxscript` are only needed for step 2 (large CUDA download). DXNN steps
need the DEEPX DXRT package (`dx_engine`) and a connected NPU.

## 2. Export the checkpoint to ONNX

Download `PIDNet_S_Cityscapes_test.pt` (**not** `PIDNet_S_Simple_Cityscapes_test.pt`
— see Model provenance above) and place it in the repo.

```bash
python export_argmax_softmax_onnx.py \
  --p PIDNet_S_Cityscapes_test.pt \
  --height 512 --width 512 \
  --batch_size 1 \
  --o pretrained_dynamic.onnx
```

Adds two output heads at 1/8 resolution: `labels` (ArgMax) and `masks` (Softmax).

## 3. Run the ONNX demo

```bash
python pidnet_demo_onnx.py assets/images/000341.png --model pretrained_dynamic.onnx
python pidnet_demo_onnx.py assets/videos/pidnet-test.mp4 --model pretrained_dynamic.onnx
```

Flags: `--output {labels,masks}`, `--view {overlay,mask,side-by-side}`, `--alpha`,
`--max-display-size`. `--help` for the full list.

## 4. Evaluate ONNX accuracy on Cityscapes

```bash
python download_cityscapes_small.py
python eval_cityscapes_onnx.py \
  --model pretrained_dynamic.onnx \
  --dataset-root cityscapes_small \
  --output-index -1 \
  --save-json metrics/pretrained_dynamic_onnx.json
```

`--output-index -1` selects the `masks` head.

## 5. Compile to DXNN

Out of scope for this repository — done manually by the DEEPX engineering team.
The compiled model must keep the same `labels`/`masks` outputs at the same
1/8-resolution shapes for the scripts below to work unchanged.

## 6. Run the DXNN demo

```bash
python pidnet_demo_dxnn.py assets/videos/pidnet-test.mp4 --model pretrained_dynamic_handmade.dxnn
```

Same flags as step 3 (`--output`, `--view`, `--alpha`, `--max-display-size`,
`--window-name`). Video only for now — no single-image input.

## 7. Evaluate DXNN accuracy on Cityscapes

```bash
python eval_cityscapes_dxnn.py \
  --model pretrained_dynamic_handmade.dxnn \
  --dataset-root cityscapes_small \
  --output-index -1 \
  --input-color rgb \
  --save-json metrics/pidnet_argmax_softmax_dxnn.json
```

`--input-color rgb` is required — the default `bgr` silently gives a wrong,
misleadingly low score. Compare against step 4's metrics for the NPU accuracy delta.

## 8. ONNX vs. DXNN output consistency

```bash
python compare_onnx_dxnn.py \
  --onnx-model pretrained_dynamic.onnx \
  --dxnn-model pretrained_dynamic_handmade.dxnn \
  --dataset-root cityscapes_small \
  --save-json metrics/onnx_vs_dxnn.json
```

Reports per-pixel cosine similarity and argmax agreement between the two `masks`
outputs (mean/min/`--threshold`-based ratio). Use `--images-dir assets/images` for a
quick smoke test without the full dataset.
