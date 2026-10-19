# UniDiff-NMR

UniDiff-NMR performs nuclear magnetic resonance (NMR) reconstruction, denoising, and decoupling using two-dimensional diffusion priors. The Plug-and-Play (PnP) framework combines a 2D UNet diffusion prior with task-specific data constraints for NUS reconstruction, DOSY denoising, pure-shift, and virtual-decoupling applications.

The project uses a 2D model throughout its workflows. The 3D NUS notebook and 4D NUS script also apply the 2D prior to slices or data blocks; the dimensionality of the data differs from that of the network. One-dimensional training and notebooks are outside the current workflow.

## Project Structure

```text
UniDiff-NMR/
├── config/
│   ├── sr_sr3_64_64.json          # 64 × 64 model configuration
│   └── sr_sr3_256_256_val.json    # 256 × 256 inference configuration
├── core/                         # Logging, metrics, and experiment utilities
├── data/
│   ├── dataset_2D.py             # Simulated 2D data generation
│   └── test_real_data_ps/         # Pure-shift data processing utilities
├── model/
│   ├── sr3_modules/
│   │   ├── unet_2D.py            # 2D UNet
│   │   ├── diffusion_pnp_2D.py   # PnP diffusion model
│   │   ├── diffusion_2D.py
│   │   └── diffusion64.py
│   ├── networks.py               # Network construction
│   ├── model.py                  # Training, sampling, and reconstruction
│   └── base_model.py
├── experiments/                  # Generated experiment logs, weights, and results
├── results/                      # 4D reconstruction arrays and timing reports
├── train_2D.py                    # 2D training entry point
├── test_nus_2D.ipynb              # 2D NUS reconstruction
├── test_nus_3D.ipynb               # 3D NUS reconstruction with 64 × 64 slices
├── test_denosing.ipynb            # DOSY denoising (filename as stored)
├── test_PS_real.ipynb             # Pure-shift reconstruction
├── test_virtual_decoupling.ipynb  # Virtual decoupling
├── test_nus_4D_real.py            # 4D NUS experiment script
├── requirement.txt
├── LICENSE
└── README.md
```

## Environment Setup

Use Python 3.10 and install a PyTorch build compatible with your local CUDA environment.

```bash
cd UniDiff-NMR
conda create -n unidiff-nmr python=3.10 -y
conda activate unidiff-nmr

# After installing a suitable PyTorch build, install the remaining dependencies.
pip install -r requirement.txt
pip install scipy matplotlib nmrglue PyYAML jupyterlab ipykernel
python -m ipykernel install --user --name unidiff-nmr --display-name "UniDiff-NMR"
```

The dependency file is named `requirement.txt`. Workflows involving NMRPipe processing require NMRPipe and a C shell, such as `tcsh`, to be installed separately. Examples that use an external sampling-mask generator also require the corresponding `libPG_cb.so` library.

## Model Configuration

| Configuration | Model Size | Main Uses |
| --- | --- | --- |
| `config/sr_sr3_64_64.json` | 64 × 64 | 2D training and 3D/4D NUS experiments |
| `config/sr_sr3_256_256_val.json` | 256 × 256 | 2D NUS, DOSY, pure-shift, and decoupling experiments |

Key configuration fields:

- `gpu_ids`: GPU device indices.
- `model.unet`: Network channels, channel multipliers, attention, and residual blocks.
- `model.diffusion`: Input dimensions, channels, and conditional generation settings.
- `model.beta_schedule`: Noise schedules for training and inference.
- `path.resume_state`: Checkpoint prefix.
- `train`: Training iterations, learning rate, validation frequency, and checkpoint frequency.

Checkpoints must match the network architecture and input dimensions. Set `resume_state` to the checkpoint prefix without `_gen.pth`, for example:

```json
"resume_state": "experiments/my_run/checkpoint/I1000000_E100"
```

Inference loads `I1000000_E100_gen.pth`. Resuming training also requires the corresponding `I1000000_E100_opt.pth`. Set `resume_state` to `null` to train from scratch.

## Pretrained Weights

Download the pretrained model weights manually from [Google Drive](https://drive.google.com/drive/folders/1orOH96FaDM25iG_pM4tmn_OtPPCfKIXO).

| Model | Configuration | Checkpoint |
| --- | --- | --- |
| 64 × 64 prior | `config/sr_sr3_64_64.json` | `I250000_E25_gen.pth` |
| 256 × 256 prior | `config/sr_sr3_256_256_val.json` | `I910000_E91_gen.pth` |

Place the downloaded files under the project root, preserving their original filenames:

```text
checkpoints/
├── 64/
│   └── I250000_E25_gen.pth
└── 256/
    └── I910000_E91_gen.pth
```

Set `path.resume_state` in the corresponding configuration to the local checkpoint prefix, **without** `_gen.pth`.

For `config/sr_sr3_64_64.json`:

```json
"resume_state": "checkpoints/64/I250000_E25"
```

For `config/sr_sr3_256_256_val.json`:

```json
"resume_state": "checkpoints/256/I910000_E91"
```

Replace the existing machine-specific absolute paths before running inference. The code loads the downloaded local files.

These `*_gen.pth` files are used for inference. Resuming training also requires the corresponding `*_opt.pth` file with the same prefix. Set `path.resume_state` to `null` to train from scratch.

## 2D Training

The training entry point is `train_2D.py`, with simulated data generated by `data/dataset_2D.py`. Configure the data, sampling masks, GPU devices, and checkpoint paths for your environment, then run:

```bash
python train_2D.py -c config/sr_sr3_64_64.json -p train -gpu 0
```

The current simulated training data uses 64 × 64 patches. Training records the loss, periodically generates validation samples, and saves model and optimizer states. Log, TensorBoard, checkpoint, and result paths are managed through the configuration and experiment utilities.

## Reconstruction and Data Processing

Start Jupyter from the project root and select the **UniDiff-NMR** kernel:

```bash
jupyter lab
```

Configure the input data, sampling masks, pretrained checkpoints, GPU IDs, and output paths for the selected task, then run the relevant notebook cells in order. The checked-in configurations and some notebook cells contain machine-specific paths that must be changed for your environment. Raw data, MAT files, pretrained weights, and external mask-generation libraries must be prepared separately when they are not included in your checkout.

| Application | Notebook | Configuration (under `config/`) |
| --- | --- | --- |
| 2D NUS reconstruction | `test_nus_2D.ipynb` | `sr_sr3_256_256_val.json` |
| 3D NUS reconstruction | `test_nus_3D.ipynb` | `sr_sr3_64_64.json` |
| 4D NUS reconstruction | `test_nus_4D_real.py` | `sr_sr3_64_64.json` |
| DOSY denoising | `test_denosing.ipynb` | `sr_sr3_256_256_val.json` |
| Pure-shift | `test_PS_real.ipynb` | `sr_sr3_256_256_val.json` |
| Virtual decoupling (VD) | `test_virtual_decoupling.ipynb` | `sr_sr3_256_256_val.json` |

### NUS Reconstruction

`test_nus_2D.ipynb` covers data loading, undersampling-mask generation, observation preparation, blockwise reconstruction, and result assembly.

`test_nus_3D.ipynb` applies the 2D 64 × 64 prior to slices of 3D data. Its workflow includes NMRPipe data loading, real and imaginary component processing, slice reconstruction, and volume assembly. The standalone `test_nus_4D_real.py` script supports 4D NUS experiments.

For 2D NUS, prepare `data/GB1_DComplex_Dephase_Part.mat` with the `LR` variable and update the mask generator's `lib_path` to your local `libPG_cb.so`. For 3D NUS, prepare the selected dataset under `data/data_3D/` and configure the slice, mask, and result paths in the notebook.

For 4D NUS, prepare these MAT files:

- `data/Hnoesy_4D.mat`, containing the `data` variable.
- `data/Hnoesy_4D_Mask_0.1.mat`, containing the `mask` variable.

After setting the 64 × 64 checkpoint prefix, run:

```bash
python test_nus_4D_real.py -c config/sr_sr3_64_64.json -p val -gpu 0 --c_batch 8
```

`--c_batch` controls the number of slices processed in each diffusion call; reduce it if GPU memory is insufficient. The script writes four reconstructed arrays (`recon_spec1_*.npy` through `recon_spec4_*.npy`) and reconstruction timing reports (`reconstruction_timing_*.json` and `reconstruction_timing_*.csv`) to `results/`.

### Denoising, Pure-Shift, and Virtual Decoupling

`test_denosing.ipynb` applies the 2D prior to DOSY denoising (`pnp_mode='dosy'`). Update the input path to your `GSP_data.mat` file; the notebook includes visualization and MAT exports for noisy, denoised, and clean spectra.

`test_PS_real.ipynb` supports pure-shift experiments, with related processing utilities in `data/test_real_data_ps/`. Prepare the Varian `fid` and `procpar` files under `data/PS_real/<name>/PSYCHE.fid/` and select the sample name in the notebook.

`test_virtual_decoupling.ipynb` provides the virtual-decoupling workflow (`pnp_mode='vd'`) using `data/CON_Decouple.mat`. Select the visualization and export cells needed for your experiment.

## Acknowledgements

This project extends the SR3/DDPM implementation from [Image Super-Resolution via Iterative Refinement](https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement). The original implementation also draws on WaveGrad and diffusion-model implementations by rosinality and lucidrains.

## License

This project retains the [Apache License 2.0](LICENSE).
