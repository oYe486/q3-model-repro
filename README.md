# 问题三：模型复现核心材料

本目录是 `DLF-main` 当前最终模型的独立副本。原目录的代码、参数和数据均未修改。提交包只保存核心代码、配置、参数差分、阈值和核验结果；约 421 MB 的复训初始化权重及约 948 MiB 的附件 2 对齐特征须从原始材料或发布地址取得。**仅凭这个小包，无法在没有上述大文件时独立复现结果。**

## 1. 文件与参数

| 路径 | 作用 |
| --- | --- |
| `trains/`、`run.py`、`data_loader.py`、`config.py` | 完整的当前模型结构、训练目标和附件 2 数据加载；与原目录逐文件相同 |
| `utils/`、`infer_attachment4.py`、`tests/` | 验证、解释、附件 4 推理及核验代码 |
| `config/config.json`、`requirements.txt` | 模型配置和 Python 依赖 |
| `weights/final_delta.pth` | 最终模型相对 `transfer_init/model.pth` **精确变化的 10 个张量**，约 0.22 MB；其余 359 个张量完全相同 |
| `weights/manifest.json`、`setup_weights.py` | 大权重的来源、SHA256 和自动下载／无损还原程序 |
| `pretrained/bert-base-uncased/` | BERT 配置和分词表；大权重在运行准备阶段从已校验的初始化权重生成 |
| `outputs/validation/` | 当前 728 条验证结果及三分类阈值 |
| `outputs/attachment4/` | 当前 20 条无标签样本的结构化预测和核验摘要；媒体文件不在此包 |
| `docs/`、`Agents.md`、`LICENSE` | 模型机制、证据边界、原工程说明及许可证 |

保存的参数是**无损差分**，不是量化或近似替代。`setup_weights.py` 按原模型状态字典还原后，逐张量校验的 SHA256 为 `4f443ec5361b29e1b0b2e4e1d4734a45451bc1429a79836ff49c6d7088772f68`。重新保存 `.pth` 时 ZIP 元数据会改变，因此文件字节 SHA256 可能不同，但所有参数名称、形状、类型和数值与原最终模型一致。

## 2. 运行环境

已核验：Windows、Python 3.9.13、PyTorch 1.13.0+cu117（CUDA 11.7）、Transformers 4.33.1、NumPy 1.23.5、SciPy 1.9.1、scikit-learn 1.0.2、pandas 1.4.4、OpenCV 4.8.1、Whisper 20231117。其他依赖见 `requirements.txt`。本包不含虚拟环境。

在本目录新建 Python 3.9 环境，先按硬件安装 PyTorch 1.13，再执行：

```powershell
python -m pip install -r requirements.txt
```

验证与推理可用 CPU，重新训练原配置使用 CUDA GPU。视频证据重算还需要 FFmpeg（`imageio-ffmpeg` 提供可执行程序）和 Whisper `base.en` 权重；模型数值预测本身不依赖视频或 Whisper。

## 3. 数据集来源与处理规则

将原题附件 2 的 `aligned_50.pkl` 放在本目录**同级**的 `datasets/附件2-数据集特征文件/` 中。预期文件大小 `993842861` 字节，SHA256 为 `66e867aa74bc70a844e806e5571e371c9abb4a35f9e2887ce9b4d97ff2cb8fcd`。不得用 `unaligned_50.pkl` 替代，也不得重新划分训练／验证集。`config/config.json` 的 `dataset_root_dir=".."` 正是此目录布局。

`data_loader.py` 按原文件中的 `train` 与 `valid` 划分读取。文本取 `text_bert`，每条为 `[3,50]`，依次是 token ID、attention mask、segment ID；语音取 `[50,74]`，视觉取 `[50,35]` 的已对齐预提取特征。回归标签来自 `regression_labels`。文本的首个 `[CLS]` 与最后一个有效 `[SEP]` 不作为音视频词位置；音视频全零行视为缺失观测，音频 `-inf` 置零。掩码、有效位置和批处理细节直接见 `data_loader.py`。本模型不重新从原视频提取训练特征。

附件 4 仅用于无标签推理。若重算 20 条预测及媒体证据，将原题附件 4 的**对齐版本** `01.pkl` 至 `20.pkl` 与 `videos/01.mp4` 至 `20.mp4` 放在同级 `datasets/附件4-可解释专项视频样本与特征文件/对齐版本/`。每个特征文件必须包含 `id`、`raw_text`、`text_bert`、`audio`、`vision`，且不得含标签。附件 4 不参与训练、模型选择或三分类阈值选取。不可将未对齐版本输入当前检查点。音视频时间／帧仅作候选定位，规则和限制见 `docs/attachment4_evidence.md`。

## 4. 获取权重并还原最终模型

原始 `transfer_init/model.pth` 大小 `441549829` 字节，SHA256 为 `1c5d5877d91ceb62aec044aabe940ddaa35d9f0bf6a3f814df5db566fd58e38c`。可从现有 `DLF-main/checkpoints/transfer_init/model.pth` 读取，或使用发布在 GitHub Release 的下载地址。**发布地址在上传完成前为空；设置后 `python setup_weights.py` 会自动下载并校验。**也可以显式指定：

```powershell
python setup_weights.py --source "C:\path\to\transfer_init\model.pth"
# 或：python setup_weights.py --url "https://github.com/OWNER/REPO/releases/download/TAG/transfer_init_model.pth"
```

脚本只读取来源文件，在本目录生成 `checkpoints/transfer_init/model.pth`、`checkpoints/final/model.pth` 和供 Hugging Face 加载的 `pretrained/bert-base-uncased/pytorch_model.bin`。还原后约需 1.3 GB 本地磁盘空间；这些生成物**不得放回 50 MB 提交包或普通 Git 历史**。脚本会拒绝大小或哈希不符的初始化权重，并校验还原参数的完整数值哈希。

## 5. 复现命令

以下命令均在本目录运行。先完成第 3、4 节，再执行：

```powershell
python -m unittest discover -s tests -v
python run.py --check-config
python -m utils.validation --gate-temperatures 1.0
python -m utils.verify_coalition_online
python -m utils.verify_coalition_masks
```

当前固定模型在附件 2 的 728 条验证样本上：MAE `0.55647`、macro-F1 `0.61582`、准确率 `0.63049`。三分类阈值为 `lower=-0.05437684431672096`、`upper=0.29138344526290894`；分数 `< lower` 为负向 0，`> upper` 为正向 2，其余为中性 1。阈值和分类指标取自同一个验证集，不能当作独立测试集成绩。`outputs/validation/valid_thresholds.json` 保存完整指标与配置。

若要从已发布的初始化权重**重新训练**当前融合头，请保留一份最终权重，然后执行：

```powershell
python run.py --seed 1111
python -m utils.validation --gate-temperatures 1.0
```

训练会改写 `checkpoints/final/model.pth` 和验证结果。`run.py` 的最终配置为 `adaptive_main=True`、`coalition_aware=True`、`train_fusion_only=True`、随机组合损失系数 `0.2`、batch size `16`、梯度累积 `10`、Adam 学习率 `3e-4`、最多 `5` 轮、验证损失早停 `2` 轮、门控温度 `1.0`。具体随机性、损失、冻结参数与网络结构见 `Agents.md` 和源码。不同硬件与 CUDA 内核可能造成重新训练的微小数值差异；使用还原的固定权重可复算现有结果。

附件 4 的完整媒体证据重算命令：

```powershell
python infer_attachment4.py --asr-model base.en
python -m utils.verify_attachment4
```

`DLF.from_final_checkpoint(root=".", device="cpu")` 可直接载入固定模型及阈值。调用 `model(..., return_explanation=True)` 得到强度、类别、融合门控和完整模态组合 Shapley 解释。现有 20 条结构化结果位于 `outputs/attachment4/predictions_20.jsonl`。

## 6. 提交体积与发布方式

本目录小包不含附件数据、MP4、`.venv`、Whisper 权重、完整 BERT 权重或两个 421 MB 检查点。建议将**代码包**作为普通 Git 文件，将 `transfer_init_model.pth` 作为 GitHub Release asset；然后把该 asset 的固定版本 URL 写入 `weights/manifest.json` 的 `release_asset_url`。`setup_weights.py` 可自动下载并以 SHA256 防止拿错版本。完整模型不应直接提交到普通 Git 仓库。

如果大权重没有公开、可持续访问的下载位置，本小包只能依赖另行提供原始 `transfer_init/model.pth`，不能声称已实现独立在线复现。
