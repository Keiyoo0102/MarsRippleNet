# MarsRippleNet

**Automated semantic mapping and quantitative characterization of Martian aeolian wind ripples from HiRISE imagery**

This repository contains the segmentation implementation for the study *Integrating semantic segmentation and frequency-domain analysis for automated mapping and quantitative characterization of Martian wind ripples*. It provides the Mars-RippleNet model, related segmentation modules, data-loading utilities, and training/evaluation scripts.

## Research workflow in the study

1. **Semantic segmentation.** Mars-RippleNet predicts four labels: masked (0), straight (1), sinuous (2), and complex (3). Masked pixels are excluded from class-average metrics.
2. **Physical parameter extraction.** The study applies 2D-FFT with a Tukey window and sub-pixel peak localization to estimate ripple spacing and orientation. Against manual measurements, the reported R² values are 0.870 for spacing and 0.718 for orientation. Only straight and sinuous ripples are used for the primary physical-parameter assessment.
3. **Dune-scale analysis.** Four representative barchan dunes are examined for ripple spacing, orientation dispersion, morphology composition, and geomorphic complexity across zones. The wavelength comparison reports 9,795 valid, overlapping-window FFT spacing estimates and a median of 2.013 m. These estimates are spatially correlated and are not 9,795 independent ripples.

The paper describes this full workflow. The current GitHub repository provides the semantic-segmentation code; it does **not** contain the study’s complete 2D-FFT extraction or dune-zoning analysis pipeline.

## Mars-RippleNet architecture

For a single-channel 64×64 input, the EfficientNet-B0 encoder produces features at the following resolutions:

| Encoder feature | Channels | Spatial size |
|:--|--:|--:|
| Input | 1 | 64×64 |
| Downsampling output 1 | 16 | 32×32 |
| Downsampling output 2 | 24 | 16×16 |
| Downsampling output 3 | 40 | 8×8 |
| Downsampling output 4 | 112 | 4×4 |
| Downsampling output 5 | 320 | 2×2 |

The five U-Net-style decoder blocks upsample and concatenate the corresponding encoder skip feature where available. Channel counts below show decoder input plus skip input, followed by decoder output.

| Decoder block | Upsampling | Input + skip → output channels | Dual-Branch Block |
|:--|:--:|:--:|:--:|
| 0 | 2×2 → 4×4 | 320 + 112 → 256 | No |
| 1 | 4×4 → 8×8 | 256 + 40 → 128 | No |
| 2 | 8×8 → 16×16 | 128 + 24 → 64 | Yes, at 16×16 |
| 3 | 16×16 → 32×32 | 64 + 16 → 32 | Yes, at 32×32 |
| 4 | 32×32 → 64×64 | 32 + 0 → 16 | No |

A 1×1 segmentation head maps the final 16-channel 64×64 decoder feature to four output classes. Thus, the model uses two Dual-Branch Blocks, after upsampling and skip concatenation in decoder blocks 2 and 3.

### Dual-Branch Block: SPM, DWR, then SAFM

Each Dual-Branch Block processes the same decoder feature in two parallel branches:

- **SPM (Strip Pooling Module):** pools along horizontal and vertical strips, forms an attention map, and returns an attention-gated feature with a residual connection.
- **DWR (Dilation-wise Residual Module):** reduces channels, applies parallel depth-wise 3×3 convolutions with dilation rates 1, 3, and 5, concatenates their outputs, and fuses them with a 1×1 convolution. In the Dual-Branch Block, the DWR-internal residual is disabled.
- **SAFM (Spatial Adaptive Feature-fusion Module):** concatenates the SPM and DWR outputs, predicts a spatial sigmoid gate, and computes a weighted fusion of the two branches. SAFM is the fusion unit inside the Dual-Branch Block; it is not the name of the whole block. The block then adds the original input as an outer residual.

The decoder order is upsampling → skip concatenation (when present) → Dual-Branch Block at the two configured indices → two decoder convolutions. See the [model implementation](https://github.com/Keiyoo0102/MarsRippleNet/blob/main/models/mars_ripple_net.py), [SPM](https://github.com/Keiyoo0102/MarsRippleNet/blob/main/models/spm_module.py), [DWR](https://github.com/Keiyoo0102/MarsRippleNet/blob/main/models/dwr_module.py), and [SAFM](https://github.com/Keiyoo0102/MarsRippleNet/blob/main/models/safm_module.py).

## Segmentation results

The following values are reported in the manuscript’s independent-test-set comparison (Table 4); they are reproduced here as reported and were not recomputed for this README.

| Model | Parameters (M) | OA (%) | mIoU | F1-score |
|:--|--:|--:|--:|--:|
| Mars-RippleNet | 6.40 | 81.36 | 0.6382 | 0.7617 |
| EfficientNet-UNet | 6.25 | 76.77 | 0.6327 | 0.7600 |
| DeepLabV3+ | 26.67 | 76.60 | 0.6298 | 0.7563 |
| SegFormer | 24.72 | 75.80 | 0.6186 | 0.7492 |
| U-Net | 31.04 | 59.24 | 0.5472 | 0.6810 |
| VM-UNet | 3.51 | 57.87 | 0.5007 | 0.6512 |

Compared with EfficientNet-UNet, Mars-RippleNet improves OA by 4.59 percentage points and mIoU by 0.0055 in absolute terms. Its reported straight, sinuous, and complex IoUs are 0.5756, 0.4155, and 0.9234, respectively. The parameter count increases by 0.15 M (about 2.4%).

### Ablation and input-size results

| Table 5 model | Parameters (M) | OA (%) | Test mIoU | F1-score |
|:--|--:|--:|--:|--:|
| BaseNet | 6.25 | 76.77 | 0.6327 | 0.7600 |
| BaseNet + SPM | 6.27 | 77.14 | 0.6380 | 0.7637 |
| BaseNet + DWR | 6.36 | 76.34 | 0.6275 | 0.7578 |
| Mars-RippleNet | 6.40 | 81.36 | 0.6382 | 0.7617 |

| Input tile size | BaseNet mIoU | Mars-RippleNet mIoU | Difference |
|:--|--:|--:|--:|
| 32×32 | 0.4646 | 0.4979 | +0.0333 |
| 48×48 | 0.6045 | 0.6284 | +0.0239 |
| 64×64 | 0.6213 | 0.6382 | +0.0169 |

Mars-RippleNet improves over BaseNet at each tested input size, but the gains are modest and segmentation accuracy remains sensitive to input size. These results do not establish strong cross-scale robustness or a globally optimal input configuration.

## Code and data availability

### Code provided here

The repository includes the segmentation model and modules, model training scripts, a segmentation evaluation script, data-loading utilities, configurations, and loss functions. It does not provide the complete 2D-FFT parameter-extraction or dune-zonal-analysis code, pretrained/final model checkpoints, or a complete offline-augmented training set. Consequently, the repository and the supporting-data deposit alone do not reproduce every training experiment end to end.

The training/evaluation scripts expect local data under DATA/Label/Augmented_Final, organized by input size (32, 48, and 64 pixels) and train/validation/test images and masks. This is an expected runtime layout, not a claim that the complete Augmented_Final directory is distributed here or on Zenodo. The evaluation script also requires a locally available checkpoint.

### Research Data Availability

Supporting research data are publicly available from [Zenodo](https://doi.org/10.5281/zenodo.23730927), under the title *Supporting Data for Automated Martian Wind Ripple Mapping and Quantitative Characterization*. The Zenodo record currently lists these five ZIP files:

- **Base_Annotations.zip** — base annotated image patches and labels at the documented patch scales.
- **Model_Evaluation.zip** — available model-evaluation records.
- **FFT_Validation.zip** — reference measurements and 2D-FFT validation records.
- **Dune_Zonal_Statistics.zip** — available dune-zone statistics.
- **Wavelength_Distribution.zip** — window-level wavelength estimates and KDE configuration.

This is a compact set of supporting data, not the full offline-augmented training dataset. The complete Augmented_Final directory is not included. Original HiRISE products should be obtained from the [NASA PDS HiRISE archive](https://hirise-pds.lpl.arizona.edu/PDS/) and are not relicensed by this repository. Published third-party reference data retain their own source terms.

### Running the segmentation scripts

Install dependencies with:

    pip install -r requirements.txt

The repository provides these entry points:

    python scripts/train_mars_ripple_net.py
    python scripts/evaluate_final.py

They require the local data layout and, for evaluation, a model checkpoint described above. Training was not run as part of this documentation update.

## Licenses

- **Software code in this repository:** MIT License, see [LICENSE](LICENSE).
- **Original research-derived data:** CC BY 4.0 only to the extent specified by the published Zenodo record and the authors’ licensing rights; see [DATA_LICENSE.md](DATA_LICENSE.md) and the [Zenodo record](https://doi.org/10.5281/zenodo.23730927).
- **Third-party HiRISE imagery and reference data:** not relicensed here; refer to NASA PDS and each applicable source’s terms.
