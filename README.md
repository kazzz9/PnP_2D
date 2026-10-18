# PnP-NMR 2D

基于二维扩散先验的核磁共振（NMR）信号重建、去噪与去耦项目。通过 Plug-and-Play（PnP）方法，将二维 UNet 扩散先验与不同实验任务的数据约束结合，用于 NUS、DOSY、Pure-Shift 和虚拟去耦等应用。

本项目以二维模型为统一入口。3D、4D NUS notebook 同样使用二维先验处理切片或数据块；数据的维度不代表网络的维度。一维训练与 notebook 不属于当前使用流程。

## 项目结构

```text
PnP_2D/
├── config/
│   ├── sr_sr3_64_64.json          # 64 × 64 模型配置
│   └── sr_sr3_256_256_val.json    # 256 × 256 推理配置
├── core/                         # 日志、指标和实验管理
├── data/
│   └── dataset_2D.py             # 二维模拟数据生成
├── model/
│   ├── sr3_modules/
│   │   ├── unet_2D.py            # 二维 UNet
│   │   ├── diffusion_pnp_2D.py   # PnP 扩散模型
│   │   ├── diffusion_2D.py
│   │   └── diffusion64.py
│   ├── networks.py               # 二维网络构建入口
│   ├── model.py                  # 训练、采样与重建接口
│   └── base_model.py
├── experiments/                  # 实验日志、权重与结果
├── test_real_data_ps/             # Pure-Shift 数据处理工具
├── train_2D.py                    # 二维训练入口
├── test_*.ipynb                   # 各应用的实验 notebook
├── test_nus_4D_real.py            # 4D NUS 实验脚本
├── requirement.txt
├── LICENSE
└── README.md
```

## 环境配置

推荐使用 Python 3.10，并根据本机 CUDA 环境安装对应的 PyTorch。

```bash
cd PnP_2D
conda create -n pnp-2d python=3.10 -y
conda activate pnp-2d

# 安装适合本机环境的 PyTorch 后，安装其余依赖
pip install -r requirement.txt
pip install scipy matplotlib nmrglue PyYAML jupyterlab ipykernel
python -m ipykernel install --user --name pnp-2d --display-name "PnP-NMR 2D"
```

依赖文件名为 `requirement.txt`。涉及 NMRPipe 数据处理的流程需另行安装 NMRPipe 和 C shell（如 `tcsh`）；使用外部采样掩膜生成器的示例需准备相应的 `libPG_cb.so`。

## 模型与配置

`model/networks.py` 仅构建二维模型：

- `define_G(opt)`：供 `model/model.py` 调用的统一入口。
- `define_G_2D(opt)`：构建 `sr3_modules/unet_2D.py` 中的 UNet，并封装为 `diffusion_pnp_2D.GaussianDiffusion`。

| 配置 | 模型尺寸 | 主要用途 |
| --- | --- | --- |
| `config/sr_sr3_64_64.json` | 64 × 64 | 二维训练及 3D、4D NUS 实验 |
| `config/sr_sr3_256_256_val.json` | 256 × 256 | 二维 NUS、DOSY、Pure-Shift 和去耦实验 |

主要配置项：

- `gpu_ids`：使用的 GPU 编号。
- `model.unet`：网络通道数、通道倍率、注意力和残差块等参数。
- `model.diffusion`：输入尺寸、通道数和条件生成设置。
- `model.beta_schedule`：训练与推理的噪声调度。
- `path.resume_state`：模型权重前缀。
- `train`：训练步数、学习率、验证频率和保存频率。

模型权重应与网络结构和输入尺寸匹配。`resume_state` 使用不带 `_gen.pth` 的前缀，例如：

```json
"resume_state": "experiments/my_run/checkpoint/I1000000_E100"
```

推理加载 `I1000000_E100_gen.pth`；恢复训练还需对应的 `I1000000_E100_opt.pth`。从头训练时将 `resume_state` 设置为 `null`。

## 二维训练

训练入口为 `train_2D.py`，模拟数据由 `data/dataset_2D.py` 生成。根据实验环境设置数据、采样掩膜、GPU 和权重路径后，使用：

```bash
python train_2D.py -c config/sr_sr3_64_64.json -p train
```

训练过程记录损失，定期生成验证样本，并保存模型与优化器状态。日志、TensorBoard、权重及结果路径由配置和实验管理模块设置。

## 重建与数据处理

在项目根目录启动 Jupyter，选择 **PnP-NMR 2D** 内核：

```bash
jupyter lab
```

根据所选任务配置输入数据、采样掩膜、预训练权重及结果保存路径，再按顺序运行 notebook 中的相关单元。

| 应用 | Notebook | 配置文件（位于 `config/`） |
| --- | --- | --- |
| 2D NUS 重建 | `test_nus_2D.ipynb` | `sr_sr3_256_256_val.json` |
| 3D NUS 重建 | `test_nus3D.ipynb` | `sr_sr3_64_64.json` |
| 4D NUS 重建 | `test_nus_4D.ipynb` | `sr_sr3_64_64.json` |
| 2D DOSY | `test_dosy_2D.ipynb` | `sr_sr3_256_256_val.json` |
| Pure-Shift | `test_PS_real.ipynb` | `sr_sr3_256_256_val.json` |
| 虚拟去耦（VD） | `test_VD_real.ipynb` | `sr_sr3_256_256_val.json` |
| CA 去耦 | `test_CaVD_real.ipynb` | `sr_sr3_256_256_val.json` |

### NUS 重建

`test_nus_2D.ipynb` 包含数据读取、欠采样掩膜生成、观测数据准备、分块重建与结果合并等步骤。

`test_nus3D.ipynb` 使用二维 64 × 64 先验处理三维数据切片，包含 NMRPipe 数据读取、实虚部处理、切片重建和体数据合并流程。`test_nus_4D.ipynb` 与 `test_nus_4D_real.py` 用于四维 NUS 实验。

### DOSY、Pure-Shift 与去耦

`test_dosy_2D.ipynb` 使用二维先验进行 DOSY 处理，并提供结果可视化与后处理单元。

`test_PS_real.ipynb` 用于 Pure-Shift 实验，相关数据处理工具位于 `test_real_data_ps/`。

`test_VD_real.ipynb` 和 `test_CaVD_real.ipynb` 分别提供虚拟去耦及 CA 去耦流程。按实验需要选择结果展示和 MAT 文件导出单元。

## 致谢

本项目基于 [Image Super-Resolution via Iterative Refinement](https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement) 的 SR3/DDPM 实现进行扩展。原实现亦参考了 WaveGrad，以及 rosinality 和 lucidrains 的扩散模型实现。

## 许可证

本项目保留 [Apache License 2.0](LICENSE)。
