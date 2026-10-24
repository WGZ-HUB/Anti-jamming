# Condition-Guided Latent Diffusion for Radar Anti-Jamming

This repository contains the inference implementation and compact examples for the proposed condition-guided latent diffusion radar anti-jamming method.

## Included scenarios

- `ISRJ`: interrupted-sampling repeater jamming
- `NFMJ`: noise frequency-modulated jamming
- `ISRJ_NFMJ`: compound ISRJ and NFMJ
- `FALSE`: dense repeater and multiple false-target jamming
- `EXTENDED_COMPOUND`: extended target under compound jamming

Each scenario includes one complex-valued MATLAB example with `echo`, `ref`, and `sig`, and the best inference-only VAE and diffusion checkpoints.

## Installation

Python 3.10 and a CUDA-enabled PyTorch environment are recommended.

```bash
pip install -r requirements.txt
```

## Inference

Run the provided compound-jamming example with one-step DDIM inference:

```bash
python inference.py --scenario ISRJ_NFMJ --ddim-steps 1
```

Run another included scenario:

```bash
python inference.py --scenario FALSE --ddim-steps 1
```

Use a custom MATLAB input containing `echo` and `ref`; `sig` is optional and is used only to report evaluation metrics:

```bash
python inference.py --scenario ISRJ_NFMJ --input path/to/sample.mat
```

The recovered complex target echo is saved as `outputs/<SCENARIO>/prediction.mat`. When `sig` is available, `metrics.json` also reports input/output SINR, SINRIF, RMSE, and correlation.

## Example data format

Each signal is a complex-valued array with shape `[N, 6400]`:

- `echo`: jammed observation
- `ref`: transmitted reference waveform used as the condition
- `sig`: clean target echo used only for evaluation

## Repository structure

```text
.
|-- inference.py
|-- requirements.txt
|-- VAE/vae/                    # STFT-VAE and TF-GridNet modules
|-- TSE/model/tse/              # condition encoder and uDiT backbone
|-- TSE/config/tse-base.yaml
|-- checkpoints/<SCENARIO>/     # best inference-only checkpoints
|-- examples/<SCENARIO>/sample.mat
`-- outputs/
```

## Reproducibility notes

- The released inference uses deterministic VAE encoding.
- The default setting uses one DDIM step, `eta=0`, and random seed 2024.
- The decoder uses the known transmitted reference scale; it does not use the target label.
- Outputs follow the unit-peak convention of the released datasets. Pass `--preserve-scale` to disable it.
- The target label `sig` is never used as an inference input.
- All paths are relative to the repository root.
