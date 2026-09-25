# 问题三：自适应主模态多模态情感分析

本目录是基于 [DLF（AAAI 2025）](https://arxiv.org/abs/2412.12225) 整理的**最终版本**，只保留附件 2 训练／验证和附件 4 逐样本推理、证据核验所需的代码与产物。它不包含问题一、二的实现，也不包含历史消融实验。清理前的完整快照位于同级目录 `../DLF-main-backup-before-cleanup-20260924/`。

## 方法与输出

模型在文本、语音、视觉的**解耦后模态特有空间**计算可用性感知的门控权重，由此形成样本自适应融合查询；最终只使用一个融合头。为使缺失模态的解释有训练依据，融合头在附件 2 训练集上见过随机非空模态组合。推理时计算三模态的全部 8 种组合，并对当前预测类别的决策边际进行精确 Shapley 分解。

- `fusion_main_modality`：门控权重最高的模态，是模型的融合路由选择。
- `main_modality`：对**本次模型预测**正向支持最大的完整模态；三模态有符号支持见 `shapley_decision_support`，正向份额见 `positive_support_share`。无明确正向优势时仍返回候选，但标记 `main_uncertain`。
- `shapley_content_support` 与 `shapley_route_effect`：固定原融合权重与重新路由两种干预的差别，用于区分内容和路由效应；不能当作现实世界中的因果效应。
- 连续情感强度为 `output_logit`；负向／中性／正向分别编码为 `0/1/2`，用附件 2 验证集选择的双阈值划分。附件 4 不参与训练、模型选择或阈值选择。

上述“主要参考模态”是**模型内的决策解释**，不是已验证的客观真实最重要模态。验证集标签仅用于离线评价，不进入附件 4 推理。

## 目录

| 路径 | 用途 |
|---|---|
| `trains/singleTask/model/DLF.py` | 在原 DLF 模型文件内实现自适应融合、8 组合归因、检查点加载与解释输出 |
| `run.py` | **当前最终模型的直接训练入口**；无需其他训练脚本转调 |
| `config.py`、`data_loader.py` | 原项目配置与附件 2 对齐数据加载 |
| `utils/validation.py`、`utils/verify_coalition_*.py` | 附件 2 验证选阈值、在线一致性与缺失模态核验 |
| `infer_attachment4.py`、`utils/attachment4_data.py`、`utils/evidence_*.py`、`utils/media_alignment.py` | 附件 4 无标签推理与关键证据定位 |
| `utils/verify_attachment4.py`、`utils/inspect_attachment4.py`、`tests/` | 附件 4 逐条核验、文件结构检查和单元测试 |
| `checkpoints/final/model.pth` | 当前最终模型参数；推理必需 |
| `checkpoints/transfer_init/model.pth` | 从头复训当前融合头所需的初始化参数；仅推理不需要 |
| `outputs/validation/` | 验证阈值、指标与 728 条逐样本记录 |
| `outputs/attachment4/`、`outputs/attachment4_structure/`、`docs/` | 附件 4 结果、结构清单和核验报告 |
| `pretrained/bert-base-uncased/` | 当前本地 BERT 权重与分词器；可按原 DLF 方式重新下载 |
| `.venv/` | 本机运行环境，不属于核心代码提交内容 |

附件 2 特征默认位于同级 `../datasets/附件2-数据集特征文件/aligned_50.pkl`；附件 4 对齐特征与视频默认位于 `../datasets/附件4-可解释专项视频样本与特征文件/对齐版本/`。路径设定分别见 `config/config.json` 和 `infer_attachment4.py`。**不要把未对齐版附件 4 特征直接输入该对齐检查点。**

## 环境与运行

已验证环境为 Windows、Python 3.9、PyTorch 1.13、CUDA 11.7。`requirements.txt` 列出除 PyTorch 外的项目依赖；先按相应硬件安装 PyTorch，再安装该文件。当前目录已有 `.venv/python.exe`，以下命令均在本目录运行。若重新安装或提交核心代码，`.venv/` 与本地 BERT 权重无需打包；但运行时须下载同款 `bert-base-uncased` 权重与分词器，或相应修改 `config/config.json` 的 `pretrained` 路径。Whisper `base.en` 首次使用也需取得其权重。

```powershell
.\.venv\python.exe -m unittest discover -s tests -v
.\.venv\python.exe run.py --check-config
.\.venv\python.exe -m utils.validation --gate-temperatures 1.0
.\.venv\python.exe -m utils.verify_coalition_online
.\.venv\python.exe -m utils.verify_coalition_masks
.\.venv\python.exe infer_attachment4.py --asr-model base.en
.\.venv\python.exe -m utils.verify_attachment4
```

若需要**重新训练**，先确保 `checkpoints/transfer_init/model.pth` 和附件 2 训练集存在，再直接运行 `run.py`（可选 `--seed`）。训练会改写 `checkpoints/final/model.pth`；随后必须重新运行验证、全部核验与附件 4 推理。当前结果已固定，无需为使用现有模型重新训练；`run.py --check-config` 只检查输入，不改动权重。

直接使用原模型类 `trains.singleTask.model.DLF.DLF`：`DLF.from_final_checkpoint()` 读取 `checkpoints/final/model.pth` 与 `outputs/validation/valid_thresholds.json`，调用 `model(..., return_explanation=True)` 输出连续强度、三分类、融合主模态和 Shapley 解释；常规 `model(...)` 保持原训练／原始推理接口。阈值默认为 `-0.0543768443`、`0.2913834453`，即分数低于前者为负向、高于后者为正向，其余为中性。

## 当前核验与边界

附件 2 验证集 728 条：MAE `0.55647`、macro-F1 `0.61582`、准确率 `0.63049`。这些三分类指标与阈值来自**同一验证集**，不是独立测试集估计。对 633 条有明确标签误差增量胜者的样本，解释主模态与该离线代理一致率为 `56.87%`，低于始终选择文本的 `62.09%`；因此不能宣称已找到真实最重要模态。

附件 4 的 20 条对齐样本均通过预测、Shapley、局部遮挡和证据记录的算法回放核验；18 条主证据可精确回指**提供的原文**片段，2 条音频主证据具有原视频时间**候选**。语音时间与视觉帧通过 MP4 音轨识别、逐词匹配和实际解码得到；代码以可解码帧数而非容器标称帧数判断视频范围，并在全文与音轨匹配率低于 `0.70` 时禁止媒体定位。逐样本输出的 `error_attribution` 标记了 `13` 的视觉特征缺失、`07`／`18` 的文本截断和 `15` 的文本—媒体不一致。原始附件 4 文件不修改；数据仍缺少原始特征行到时间／帧的权威映射，故不能声称音视频定位已完成严格的特征来源核验。附件 4 无真实标签，这些异常是可能的输入／证据误差来源，不能直接当作已证实的分类错误原因。详见 [附件 4 证据报告](docs/attachment4_evidence.md)及[提交措辞与定位边界](docs/attachment4_location_boundary.md)。

原始 DLF 代码、模型设计与论文归属原作者；本目录保留其 `LICENSE`。若引用基础方法，请引用 Wang 等人的 DLF 论文。
