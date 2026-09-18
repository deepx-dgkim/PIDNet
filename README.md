# PIDNet on DEEPX NPU — Evaluation Example

This repository is an example project for evaluating [PIDNet](https://github.com/XuJiacong/PIDNet)
on a DEEPX NPU. It takes an official PIDNet-S Cityscapes checkpoint, exports it to
ONNX with the post-processing (ArgMax / Softmax) moved into the graph so the NPU does
that work instead of the host CPU, and provides demo and accuracy-evaluation scripts
for both the ONNX model and the DXNN model compiled from it.

## Model provenance

The pretrained model comes from the official
[XuJiacong/PIDNet](https://github.com/XuJiacong/PIDNet) repository — specifically the
PIDNet-S Cityscapes checkpoint (`PIDNet_S_Cityscapes_val.pt` or
`PIDNet_S_Cityscapes_test.pt`), distributed from that repository's README via Google
Drive. `modify.py` needs the PIDNet-S model *definition* (not the full training repo)
to load that checkpoint and export it; this project vendors only the three files that
definition actually requires — `models/__init__.py`, `models/pidnet.py`, and
`models/model_utils.py`, copied as-is from `models/` in the official repository —
instead of a full clone. They are pure PyTorch (`torch`, `torch.nn`,
`torch.nn.functional` only) with no dependency on the rest of that repository
(training tools, configs, datasets, etc.).

> **Checkpoint sanity check:** the official architecture includes the `pag3`, `pag4`,
> and `dfm` boundary/detail fusion modules. If you load a checkpoint that is missing
> those (for example because it was trained with a different, "simplified" variant of
> PIDNet), PyTorch's `strict=False` loading will silently leave them randomly
> initialized and the exported model will produce plausible-looking but wrong
> segmentation. Always confirm `matched == len(model.state_dict())` when loading a new
> checkpoint before trusting anything exported from it. See `ANALYSIS.md` for a worked
> example of this exact failure and how it was diagnosed.

## Repository layout

```text
PIDNet/
├── modify.py                    # Export: checkpoint -> ONNX with ArgMax + Softmax baked in
├── models/                      # PIDNet-S model definition, vendored from XuJiacong/PIDNet
│   ├── __init__.py
│   ├── pidnet.py
│   └── model_utils.py
├── pidnet_demo_onnx.py          # ONNX demo (image / image folder / video)
├── pidnet_demo_dxnn.py          # DXNN demo (video, async NPU pipeline)
├── download_cityscapes_small.py # Fetches a small Cityscapes validation subset
├── eval_cityscapes_onnx.py      # mIoU / pixel accuracy / mean accuracy for an ONNX model
├── eval_cityscapes_dxnn.py      # Same, for a compiled DXNN model
├── compare_onnx_dxnn.py         # ONNX vs. DXNN output consistency (cosine similarity)
└── ANALYSIS.md                  # Root-cause writeup of a checkpoint/architecture mismatch found during development
```

## 1. Install dependencies

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

`requirements.txt` includes `torch` and `onnxscript`, needed only for the export step
(step 2) — they pull in a sizeable CUDA toolchain, so expect a multi-GB download.
DXNN-related steps additionally require the DEEPX DXRT Python package (`dx_engine`)
and a connected DEEPX NPU.

## 2. Export the checkpoint to ONNX

Download `PIDNet_S_Cityscapes_test.pt` from the
official repository's Google Drive link and place it somewhere in this repo (for
example at the repo root).

`modify.py` wraps the PIDNet-S backbone with the two output layers this project
needs the NPU to compute, instead of doing them on the host after inference:

```python
logits = backbone(x)                  # [B, 19, H, W]
labels = torch.argmax(logits, dim=1)  # int64  [B, H, W]      -> NPU-side ArgMax
masks  = F.softmax(logits, dim=1)     # float32 [B, 19, H, W] -> NPU-side Softmax
```

Export with a fixed **batch size of 1** and a fixed **512×512 input**:

```bash
python modify.py \
  --p PIDNet_S_Cityscapes_test.pt \
  --height 512 --width 512 \
  --batch_size 1 \
  --o pretrained_dynamic.onnx
```

## 3. Run the ONNX demo

`pidnet_demo_onnx.py` accepts an image file or a video file:

```bash
python pidnet_demo_onnx.py assets/images/000341.png --model pretrained_dynamic.onnx
python pidnet_demo_onnx.py assets/videos/pidnet-test.mp4 --model pretrained_dynamic.onnx
```

Useful flags: `--output {labels,masks}` selects which baked-in head to visualize,
`--view {overlay,mask,side-by-side}` controls the visualization, `--alpha` controls
overlay opacity, `--max-display-size` caps the preview window size. Run
`python pidnet_demo_onnx.py --help` for the full list.

## 4. Evaluate ONNX accuracy on Cityscapes

Fetch a small Cityscapes validation subset once:

```bash
python download_cityscapes_small.py
```

Then evaluate mIoU / pixel accuracy / mean accuracy. `--output-index -1` selects the
`masks` (Softmax) head, whose per-class scores are equivalent to raw logits for
argmax-based accuracy metrics:

```bash
python eval_cityscapes_onnx.py \
  --model pretrained_dynamic.onnx \
  --dataset-root cityscapes_small \
  --output-index -1 \
  --save-json metrics/pretrained_dynamic_onnx.json
```

## 5. Compile to DXNN

Compiling the ONNX model into a `.dxnn` file for the DEEPX NPU is **out of scope for
this repository** — it is done manually by the DEEPX engineering team using internal
compilation tooling, not by a script here. The compiled model is expected to expose
the same two outputs as the ONNX model, `labels` and `masks`, at the same 1/8-resolution
shapes, so that the demo and evaluation scripts below work against it unchanged.

## 6. Run the DXNN demo

`pidnet_demo_dxnn.py` runs the compiled DXNN model through an async NPU inference pipeline
(separate reader / submit / wait / postprocess threads) for maximum throughput:

```bash
python pidnet_demo_dxnn.py assets/videos/pidnet-test.mp4 --model pidnet_argmax_softmax.dxnn
```

It shares the same flag names as `pidnet_demo_onnx.py` (`--output`, `--view`,
`--alpha`, `--max-display-size`, `--window-name`) for consistency. Run
`python pidnet_demo_dxnn.py --help` for the full list.

> **Video only, for now.** `pidnet_demo_dxnn.py`'s async pipeline is currently built around
> a single video stream and does not accept a single image file. An image-input DXNN
> demo is planned but not yet implemented in this repository.

## 7. Evaluate DXNN accuracy on Cityscapes

Same dataset, same metrics, against the compiled model:

```bash
python eval_cityscapes_dxnn.py \
  --model pidnet_argmax_softmax.dxnn \
  --dataset-root cityscapes_small \
  --output-index -1 \
  --input-color rgb \
  --save-json metrics/pidnet_argmax_softmax_dxnn.json
```

`--input-color rgb` is required here: `eval_cityscapes_dxnn.py` defaults to
`bgr`, but every model exported by `modify.py` (and `pidnet_demo_dxnn.py`) expects RGB
input, matching PIDNet's original training-time preprocessing. Leaving this at the
default `bgr` silently feeds the wrong color order and produces a misleadingly low
score that reflects a preprocessing mismatch, not model quality.

Compare the resulting `metrics/*.json` files (mIoU, pixel accuracy, mean accuracy)
against the ONNX evaluation from step 4 to quantify any accuracy drop introduced by
NPU compilation (quantization, in particular).

## 8. ONNX vs. DXNN output consistency (cosine similarity)

Beyond dataset-level mIoU, it is useful to check that the ONNX model and the compiled
DXNN model produce numerically consistent per-pixel outputs on the same input, since
mIoU alone can hide a shift that happens to still pick the same argmax class most of
the time. `compare_onnx_dxnn.py` does this by comparing each model's `masks`
(Softmax) output at the network's native 1/8-resolution stride, before any
upsampling:

```bash
python compare_onnx_dxnn.py \
  --onnx-model pretrained_dynamic.onnx \
  --dxnn-model pidnet_argmax_softmax.dxnn \
  --dataset-root cityscapes_small \
  --save-json metrics/onnx_vs_dxnn.json
```

For each image, it computes:
- **Per-pixel cosine similarity** between the two models' 19-class score vectors at
  every spatial location (`cos_sim = dot(a, b) / (norm(a) * norm(b))`), pooled across
  the whole dataset rather than reported as a single number — mean, min, and the
  fraction of pixels below `--threshold` (default `0.98`) — since quantization error
  is not uniform across classes or regions.
- **Per-pixel argmax agreement**, as a cross-check against the mIoU delta from step 7:
  a large cosine-similarity drop with a small argmax-agreement drop usually means
  quantization is shifting *confidence* without yet flipping the predicted class;
  investigate before it does.

Use `--images-dir` to point at a plain folder of images instead of a
`cityscapes_small`-shaped dataset (for a quick smoke test, e.g. `--images-dir
assets/images`). `--dxnn-input-color` defaults to `rgb` for the same reason
`--input-color rgb` is required in step 7.
