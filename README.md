# compsr

Residual CNNs that restore the luma of an image after it has been downscaled, AV1-compressed, decoded, and Lanczos-upscaled back to the original resolution.

The network does not replace the upscaler. It sits after it, at the display resolution, and predicts a residual added to the upscaled luma. Chroma stays on the codec's ordinary 4:2:0 path. That split matches a real decoder: the display process already upscales, and a small luma CNN is cheap enough to consider as a post-filter.

The five networks are the architectures in the original Colab notebooks (`notebooks/SRCompA.ipynb` through `SRCompE.ipynb`). This tree reimplements them in PyTorch, trains them under one sample budget, and reports MSE, PSNR, and SSIM on the hosted test patches. PyTorch is the port because the models are ordinary convolutions (8k–25k parameters) and the layer list is small enough to match Keras parameter counts exactly. The notebooks stay in `notebooks/` unchanged.

## Problem

A typical delivery chain reduces resolution, encodes with a modern codec, and lets the receiver upscale with a linear filter. Two degradations stack: the low-pass of the downscale, and the quantization noise of the codec. Classic super-resolution often studies bicubic downscale at 2× or more and ignores compression. Here the downscale ratios are modest, including fractional ones, because that is where a local CNN still has a chance to recover detail, and the codec is part of the observation model.

The ratios are 2:1, 8:5, and 4:3. The codec is AV1. The README that shipped with the notebooks calls the operating points QP 20, 30, 40, and 50 on the AV1 0–63 scale: 20 is near-lossless, 50 is the strongest quantization of the four. (One sentence in the original README labels QP 50 as high quality; the dataset list and the measured noise levels treat it as the low-quality point.)

Each network is a map from upscaled luma to restored luma at the same resolution, so one set of weights can be selected from the ratio and the quantizer of the bitstream that was just decoded.

## Data pipeline

Source images are DIV2K. The original split used here is:

- DIV2K 1–800, randomly divided 80/20 into train and validation (the original script's seed was not published).
- DIV2K 801–900 as the test set. Images 901–1000 have no public full-resolution pair, so they are unused.

For each ratio `r` and quality `Q` the published procedure is:

1. Convert RGB to YUV 4:2:0. The CNN sees only Y. The released reference patches occupy code values 16–235, which is limited-range luma.
2. Lanczos downscale with window parameter `a = 5`.
3. Encode the downscaled frame with libaom AV1 at quality `Q`, then decode.
4. Lanczos upscale with `a = 5` back to the original luma size.
5. Draw patches whose reference Y has `max - min >= 8`. Train and validation patches are 48×48. Test patches are 64×64.
6. Store `data` (reference Y), `noisy_data` (upscaled reconstruction), and `patchsize` in a compressed `.npz`.

Hosted archives live under `https://storage.googleapis.com/srcompdata/Ratio_{2by1|8by5|4by3}/` with names `DIV2K_{train,valid,test}_{ratio}_{qp}_{48x48|64x64}.npz`. Every URL and sha256 is in `compsr/checksums.json`. `compsr download` checks the hash before training.

The 2:1 and 8:5 archives are distinct: noisy patches do not repeat across QP, and the 2:1 QP 50 test baseline MSE matches the number printed in the notebooks (about 0.002260 on the `[0, 1]` scale). The 4:3 archives are not usable. All eight 4:3 train/validation files share one sha256, all four 4:3 test files share another, the stored arrays are 64×64 even when the filename and `patchsize` field say 48, and the `(reference, noisy)` multiset is the 2:1 QP 50 test split. Training on them would report a fake ratio. `load_split` refuses entries marked `"usable": false`. `results/dataset_audit.json` is the check that produced that conclusion. `compsr generate` is the reconstruction of the README procedure for anyone who wants a real 4:3 set; it was not run on all of DIV2K here, and it does not claim bit-exact agreement with the hosted files (libaom version, `cpu-used`, matrix, crop, and the original 640/160 seed were not published).

## Architectures

Every network is fully convolutional, takes one luma channel in `[0, 1]`, and returns `input + residual`. The last convolution is `tanh`, so the residual is in `(-1, 1)` before the skip add. Training loss is MSE against the reference **before** 8-bit rounding. Evaluation rounds and clamps, matching `round_postpredict` in the notebooks.

Padding is TensorFlow `SAME` (an odd pad puts the extra pixel on the bottom and right). Weights use Keras Glorot uniform and zero bias. `SeparableConv2D` is a depthwise kernel with no bias, then a pointwise 1×1 with bias, then the activation. Stride-2 branches are bilinear-upsampled by 2 and concatenated with the full-resolution branch. Keras-flops did not count that upsample or the activations; this repo uses the same convention so the FLOP numbers below are comparable: `2 * MAC + bias add`, plus one add per pixel for the residual. Activations are free in that count.

| Model | Idea | Params | FLOPs / pixel | Adam lr |
| --- | --- | ---: | ---: | ---: |
| A | Dense 5×5 / 3×3 stack, channels 32-28-24-20-16-16-1 | 24953 | 49770 | 1e-3 |
| B | Spatially separable stack. The opening 5×5 is `(5, 1)` then `(1, 5)` | 20327 | 40381 | 1e-3 |
| C | Full-resolution 20-18-16 path plus a stride-2 40-28-24 path, 5×5 head | 24607 | 23307 | 1e-3 |
| D | Same two-path shape as C, both paths spatially separable | 17367 | 16455 | 2e-3 |
| E | Two-path network with depthwise-separable 5×5 layers, 3×3 head | 8277 | 9028 | 2e-3 |

These counts are the Keras summaries saved in the notebooks and are checked by `tests/test_models.py`.

Model B needs a note. The notebook **source** currently writes the first layer as a dense 5×5, which is 15,815 parameters. The executed summary, the FLOP print, and the original README all say 20,327 and 40,381, and they describe B as the spatially separable network. The separable front-end is what those numbers are. Later separable pairs in the same notebook are `(k, 1)` with a linear activation followed by `(1, k)` with the nonlinearity; B uses that pattern for the opening 5×5 as well, with `tanh` on the second factor.

Learning rates are the ones written next to each `generate_cnn` (1e-3 for A–C, 2e-3 for D–E). They are part of the architecture recipe. The sample budget is shared; the step size is not.

## Training setup

Optimizer is Adam with Keras defaults: betas `(0.9, 0.999)`, `eps = 1e-7`, no weight decay. Batch size is 256 for every model. Each epoch shuffles with a `torch.Generator` seeded by the run seed. Weight init is seeded the same way.

The hosted training splits hold about 271k patches of 48×48. A full epoch of model A is on the order of ten minutes on the 4-core CPU used for this repo, and B is slower, so a 5×8 grid of full passes does not fit a short training budget. Every reported run therefore uses the same subset:

- 16,384 training patches, `subset_seed = 0` (the same index set for every model and seed)
- 4,096 validation patches, same subset seed, used only to log validation MSE
- the **entire** test split for reported MSE, PSNR, and SSIM

Seeds 0, 1, and 2 are trained for the headline setting, ratio 2:1 at QP 50, which is the only setting the original notebooks scored. Every other reported setting uses seed 0. Epoch count is the same for every run; the value used is in `results/protocol.json` and `configs/train.yaml`.

The original Colab logs are not on this budget. They cover only 2:1 at QP 50, one run each, and the saved epoch counts disagree across models (A's log shows 20 while its cell now says 40, B and C show 20, D shows 40, E's source says 30 and stores no epoch log). The test-cell outputs in those notebooks print MSE reductions of 6.509912759065628% (A), 7.026189565658569% (B), 7.0015788078308105% (C), 7.105708122253418% (D), and 6.9490790367126465% (E). A–D share a printed baseline MSE of 0.0022601327; E prints 0.0022601304. Those figures are the notebook logs, not outputs of this trainer.

## Metrics

All scores compare against the reference luma, not against the upscaled input.

- **MSE** is the mean over every test pixel after scaling code values by `1/255`. Restored pixels are rounded to 8-bit and clamped first. Baseline pixels are the archived uint8 reconstruction.
- **PSNR** is `10 log10(1 / MSE)` with peak 1.0 on that scale (peak 255 on code values). It is one dataset-wide PSNR, not the mean of per-patch PSNRs. Gain is restored PSNR minus baseline PSNR, in dB.
- **SSIM** is Wang et al. 2004: Gaussian window, sigma 1.5, 11 taps, population covariance (`use_sample_covariance=False`), `data_range=1`. The reported number is the mean of per-patch SSIM. `tests/test_metrics.py` checks it against scikit-image.

MSE reduction percent is `(MSE_baseline - MSE_restored) / MSE_baseline * 100`, the quantity the notebooks printed.

## Results

Reported runs use the budget in `configs/train.yaml`: 6 epochs, batch 256, 16,384 training patches (`subset_seed` 0), full test split. Seeds 0, 1, and 2 are trained for 2:1 at QP 50. Other settings use seed 0. `python -m compsr report` writes `results/metrics.csv`, `results/table.md`, and the plots from `results/runs/*.json`.

<!-- RESULTS -->
<!-- /RESULTS -->

A star marks the highest mean PSNR gain at that QP. Where three seeds exist, the cell is mean ± sample standard deviation (`ddof=1`). Plots:

- `results/psnr_gain_vs_params.png` — headline setting, PSNR gain against parameter count
- `results/psnr_gain_vs_qp.png` — PSNR gain against AV1 QP

The epoch count comes from a probe of A and E on 2:1 QP 50, seed 0, saved in `results/pilot_2by1_qp50.json`. That probe is not part of the results table. It used 8 epochs and, between epochs, a 2,048-patch slice of the test set. Epoch 6 already contained most of the probe gain while both models were still moving, so the grid stops at 6 for every model.

## Efficiency

Parameter counts and FLOPs/pixel are properties of the graph; the table in Architectures is the keras-flops-compatible count. `python -m compsr bench` writes `results/efficiency.json`: patch throughput and the latency of one luma frame. `python -m compsr export-onnx` writes `results/onnx/model_{A-E}.onnx` (gitignored) and `results/onnx_report.json`. The legacy tracer freezes `SAME` padding and the bilinear resize to the example resolution, so each graph matches PyTorch at that height and width (any batch) and is not a dynamic-shape model.

<!-- EFFICIENCY -->
<!-- /EFFICIENCY -->

## Reproduce

```bash
pip install -r requirements.txt
python -m compsr download --data-dir data
python -m compsr train-grid --config configs/train.yaml
python -m compsr bench --output results/efficiency.json
python -m compsr export-onnx --report results/onnx_report.json
python -m compsr report --results-dir results
python -m compsr audit --data-dir data --check-arrays --output results/dataset_audit.json
```

Train one cell:

```bash
python -m compsr train --model C --ratio 8by5 --qp 30 --seed 0
```

Rebuild a patch archive from RGB frames (needs `ffmpeg` built with libaom):

```bash
python -m compsr generate \
  --images-dir /path/to/DIV2K_train_HR \
  --split train --split-seed 0 \
  --ratio 4by3 --qp 30 \
  --patch-size 48 --patches-per-image 16 \
  --output data/generated_4by3_qp30_train.npz
```

`generate` crops the top-left to a size that is an integer ratio and even in both luma and the 4:2:0 downscale, converts full-range RGB to limited-range YUV with the selected matrix (`bt709` by default), Lanczos-filters with `param0=5`, and encodes one still AV1 frame at `-crf Q -b:v 0`. `--cpu-used` changes the bitstream. CI does not run this path; `tests/test_generate.py` checks the geometry and the ffmpeg arguments.

Tests and lint:

```bash
ruff check .
pytest -q
```

GitHub Actions (`.github/workflows/ci.yml`) runs those two commands on Python 3.12. It does not download the archives.

## Tradeoffs and limitations

- **Capacity versus work.** A is the dense stack: most parameters and most FLOPs per pixel. C spends similar parameters on a stride-2 branch and cuts FLOPs roughly in half. E is the small depthwise model, about one fifth the FLOPs of A. Which one wins on PSNR is an empirical question answered by the table, not by the parameter count. D and E were given a higher Adam step in the notebooks; under a shared step they might rank differently.
- **The CNN is a post-filter, not a joint upscaler.** Fractional ratios stay in the Lanczos stage so one fully convolutional net can serve every ratio. A network that upscales inside the graph could do better and would need a different design per ratio.
- **Y only.** Chroma errors and chroma upscaling are untouched. PSNR here is luma PSNR on extracted patches, not a full-frame YUV or VMAF score.
- **Patch metrics.** Test patches were kept only when the reference was contrasty (`range >= 8`), so the average is not a random crop of DIV2K and not a full-frame score.
- **Training subset.** Reported gains are for 16,384 patches and the epoch count in `results/protocol.json`, not for a 20-epoch pass over all ~271k patches. Absolute gains are not comparable to the original Colab logs. Relative ranking inside this budget is.
- **4:3 is missing** because the hosted files are copies of the 2:1 QP 50 test set. The generator is there; an end-to-end DIV2K encode was not run.
- **Single machine, CPU.** Throughput numbers are from that CPU. They will not match a GPU or a different oneDNN build.
- **ONNX** matches PyTorch at the exported resolution. Other resolutions need another export. The max absolute error is in `results/onnx_report.json`.
- **No perceptual loss, no quantization-aware training, no bitstream side information.** The net never sees QP as an input. The training set is already separated by QP, which is the deployment story in the original notebooks: pick the checkpoint whose `(ratio, QP)` is closest to the frame just decoded.

## Layout

| Path | Role |
| --- | --- |
| `compsr/models.py` | Networks A–E, parameter and FLOP counts |
| `compsr/data.py` | Manifest, checksummed download, patch subsets |
| `compsr/train.py` | Seeded Adam/MSE loop |
| `compsr/evaluate.py` | Full-test MSE, PSNR, SSIM |
| `compsr/generate.py` | DIV2K → Lanczos → AV1 → patches |
| `compsr/checksums.json` | sha256 for the 36 hosted archives |
| `configs/train.yaml` | Shared budget and the settings that are actually trained |
| `notebooks/` | Original Colab notebooks |
| `results/` | Run JSON, `metrics.csv`, plots, efficiency, audit |

## License

MIT. See `LICENSE`.
