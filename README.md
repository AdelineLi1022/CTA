# CTA：ISLES'24 稀疏 LVO 定位研究

按「单类别定位 → CoW 解剖引导 → 人工审核后的 BA/TICA/M1 分类」组织的研究代码。输入为 Hugging Face Parquet 或已经提取的 NIfTI。检测后端使用 **nnDetection**；本项目负责数据适配、空间检查、代理框生成、患者划分、运行接口、预测坐标转换和距离评估。

**当前验证范围：合成 NIfTI/Parquet 端到端测试。尚未读取你的真实分片、安装 GPU nnDetection 或训练模型。** 本地数据目录是 `F:\isles24_parguet`，云端不能直接访问这个 Windows 路径。

不把 149 个分片解释为 149 例，不假设所有空标注都是阴性，不根据最近的 CoW 标签自动生成血管节段真值。固定框是定位代理标签，不能用于血栓体积测量。本项目实现的是 ISLES'24 方法适配，不承诺复现原论文的指标或独立测试结果。

## 1. 在 Windows 上检查你的 Parquet

在本项目根目录打开 PowerShell，建议使用 Python 3.12：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[test]"
.\.venv\Scripts\cta-lvo.exe inspect "F:\isles24_parguet" --output "F:\isles24_work\parquet_schema.json"
```

`parquet_schema.json` 包含每个分片的 Arrow 字段、嵌套结构和行数，不加载影像内容。**下一步需要根据该文件确认实际 CTA、LVO、CoW 字段和编码方式。** Hugging Face 上不同版本字段可能不同，示例映射不是已经验证的 ISLES24 schema。

Linux / 当前云环境：

```bash
python -m venv /workspace/cta-venv
/workspace/cta-venv/bin/python -m pip install -e '.[test]'
/workspace/cta-venv/bin/pytest -q
```

如果以后需要直接下载 HF 数据，可用：

```bash
cta-lvo download OWNER/DATASET --revision DATASET_COMMIT_SHA --output /data/isles24_parquet
```

请填写真实仓库 ID 和固定 revision；私有/受限数据通过本机 Hugging Face 的正常认证方式授权，不把 token 写进项目。当前云环境访问 Hugging Face 返回 HTTP 403，本地 Parquet 流程无需这个网络请求。

## 2. 字段映射与空间要求

复制 `configs/hf_mapping.example.json` 为自己的映射配置。值是字段路径，嵌套字段使用 `a.b`。输入影像支持 NIfTI 文件路径、NIfTI 原始/压缩字节，以及 HF 的 `{bytes, path}` 对象；**不自动猜测 ZIP、DICOM、NumPy 数组的编码或空间**。文件相对路径以 Parquet 根目录为基准。

`space` 和 `annotation_status` 可以映射到元数据列；也支持 `{"constant": "ncct"}` 这种显式常量，但只有你已根据数据说明和影像检查确认时才能使用。状态为：

- `positive`：明确阳性，必须有非空 LVO mask。
- `negative`：经过可靠标注/审核确认阴性，LVO mask 可为空或不存在。
- `unknown`：没有可靠阳性/阴性结论；统计保留，训练和评估排除。

不能为了让导入通过，给全部病例填写 `negative` 或在原始 CTA 上填写 `space=ncct`。

```powershell
.\.venv\Scripts\cta-lvo.exe import-parquet "F:\isles24_parguet" --mapping configs\my_mapping.json --output "F:\isles24_work\imported"
```

输出目录必须不存在，避免覆盖原始数据或已导入文件。失败的部分输出会保留以便诊断；修改映射后选择一个新目录重试。按 batch 流式读取分片，每次一行。重复患者会报错，需要先明确重复扫描/多次就诊的处理规则。

如果 HF 版本只有**原始 CTA**，需要另行获取并核实配准 CTA；本代码不下载或伪造 `space-ncct_cta`，不复制 mask affine 来假装配准。标准输入为同一患者、同一 NCCT 参考空间中的 CTA、LVO mask、CoW mask，3D、毫米单位，shape/affine 一致。网格一致只是必要条件，仍须做影像叠加的人工 QC，不能证明解剖配准正确。

也可跳过 Parquet 导入，直接建立 manifest：

```json
{
  "cases": [
    {
      "patient_id": "sub001",
      "space": "ncct",
      "annotation_status": "positive",
      "cta": "sub001/space-ncct_cta.nii.gz",
      "lvo": "sub001/lvo_mask.nii.gz",
      "cow": "sub001/cow_segmentation.nii.gz"
    }
  ]
}
```

manifest 中相对路径基于 manifest 所在目录。患者 ID 使用字母、数字、下划线或连字符，避免 nnDetection 路径解析问题。

## 3. 统计病例并冻结划分

以下命令可在 PowerShell 激活虚拟环境后执行：

```powershell
cta-lvo audit "F:\isles24_work\imported\manifest.json" --output "F:\isles24_work\audit.json"
cta-lvo split "F:\isles24_work\imported\manifest.json" --seed 24 --output "F:\isles24_work\splits.json"
```

`audit.json` 逐例包含阳性/阴性/未知状态、shape、spacing、LVO 26 邻接连通域数量、中心 voxel 和 RAS 毫米坐标、各 CoW 标签体素数、每个 LVO 中心距最近 CoW 体素的物理距离及候选标签。最近 CoW 类别只是候选。每个连通域暂作为一个定位靶点，连通域是否代表独立闭塞需要人工核查。

默认按状态分层，患者级约 70/15/15 划分；固定 seed，同一患者不能跨组。小样本组可能没有验证/测试病例，训练接口会拒绝空 train/val。只有患者编号真正对应独立患者时，该划分才能防止泄漏；多次就诊需先统一患者 ID。所有对比实验复用同一个 `splits.json`。框大小、ROI margin、距离阈值和置信度阈值均需预先指定或仅用验证集选择。

## 4. 阶段一：单类别 nnDetection

```powershell
cta-lvo export "F:\isles24_work\imported\manifest.json" --splits "F:\isles24_work\splits.json" --output "F:\isles24_work\det_data\Task100_ISLES24LVO" --box-mm 10 --roi full
```

`10 mm` 是可修改的研究超参数，**不是已核实的原论文框尺寸**。以稀疏连通域中心构造沿影像 voxel 轴的物理尺寸立方代理框，转换成 nnDetection 所需的实例标签图和 `{instances: {实例号: 类别号}}`。离散化会向外取整，图像边缘处会截断。框重叠报错，要求核查连通域，不会静默覆盖实例。

导出结构：

```text
Task100_ISLES24LVO/
  dataset.json
  raw_splitted/
    imagesTr/   # train + val CTA，case_0000.nii.gz
    labelsTr/   # 代理实例图 + case.json
    imagesTs/   # 独立 test CTA（若存在）
    labelsTs/   # 仅供评估
  preprocessed/splits_final.pkl
  splits_final.json
  proxy_metadata.json
```

类别从 0 开始，实例图背景为 0、目标从 1 开始。`splits_final.pkl` 是导出器生成的可信本地文件，放在上游实际读取的 `preprocessed/` 下；仅训练 fold 0，避免上游自动创建其他随机划分。nnDetection 的辅助分割监督也来自代理框，不代表真实血管/血栓分割。

### Linux GPU 上运行 nnDetection

Windows 负责整理数据即可；nnDetection 上游面向 Linux，含 CUDA 扩展，不能只在本项目安装一个 PyTorch 就算完成安装。训练请使用单独 Linux GPU 环境，按[上游安装说明](https://github.com/MIC-DKFZ/nnDetection#installation)安装匹配的 PyTorch、torchvision、CUDA、nnDetection，并运行：

```bash
python -c 'import torch; import nndet._C; assert torch.cuda.is_available()'
```

本项目导出格式和命令接口参考上游提交 `97a58f3110b71caf1b4bcc1851e67cf11e987fc5`。其依赖包含较老的 Lightning / SimpleITK / nnU-Net，应与本项目 Python 3.12 数据环境隔离；尚未验证 GPU 环境的依赖组合。预处理规划也需要 GPU，显存预算根据目标分辨率确定，不在 CPU 云环境尝试重型安装。

将整个 `det_data/Task100_ISLES24LVO` 复制到 GPU 主机，使用上游命令即可：

```bash
export det_data=/data/det_data
export det_models=/data/det_models
export OMP_NUM_THREADS=1
export det_num_threads=4
nndet_prep 100 --full_check -np 4 -npp 4
# 根据实际生成的 plan 确认 D3V001_3d 路径。
nndet_unpack /data/det_data/Task100_ISLES24LVO/preprocessed/D3V001_3d/imagesTr 4
nndet_train 100 -o exp.fold=0 --sweep
nndet_predict 100 RetinaUNetV001_D3V001_3d --fold 0
```

如果同一 GPU Python 环境也能安装本项目，等价封装为：

```bash
cta-lvo nndet prep /data/det_data/Task100_ISLES24LVO --models /data/det_models
cta-lvo nndet unpack /data/det_data/Task100_ISLES24LVO --models /data/det_models --plan D3V001_3d
cta-lvo nndet train /data/det_data/Task100_ISLES24LVO --models /data/det_models
cta-lvo nndet predict /data/det_data/Task100_ISLES24LVO --models /data/det_models
```

封装使用子进程保留退出状态，不掩盖预处理/训练失败；模型名和 plan 可通过参数调整。当前云环境没有验证这些 GPU 命令执行成功。

## 5. 阶段二：CoW 引导与真实推理

使用相同 splits 导出第二个任务：

```powershell
cta-lvo export "F:\isles24_work\imported\manifest.json" --splits "F:\isles24_work\splits.json" --output "F:\isles24_work\det_data\Task101_CoWOracle" --roi cow-oracle --margin-mm 20 --box-mm 10
```

CoW 前景 bounding box 向外扩展物理 margin，裁剪同时更新 affine。若 ROI 漏掉任何 LVO 中心会报错；应报告这类病例、在训练集上选择合理 margin，并冻结后重新实验。不会根据测试病例 LVO 真值扩大裁剪范围。人工 CoW mask 的结果属于 **oracle 实验**，应与自动 CoW 推理分开报告。

自动 CoW 分割模型/权重不包含在本仓库。接入外部分割模型产生同空间 mask，在 manifest 中设置 `cow_source: "automatic"`，再用 `--roi cow-auto` 训练/验证自动 ROI。

对**无 LVO 标注的新 CTA**，可准备独立推理任务，保持任务名与训练任务一致、使用不同父目录：

```bash
cta-lvo inference-input /data/new_manifest.json \
  --source-task /data/det_data/Task101_CoWOracle \
  --output /data/new_inputs/Task101_CoWOracle --roi cow-auto --margin-mm 20
cta-lvo nndet predict /data/new_inputs/Task101_CoWOracle --models /data/det_models
```

新 manifest 只需 `patient_id`、`space`、`cta`；CoW 模式另需 `cow` 和 `cow_source=automatic`。`inference-input` 不读取 LVO，即使 manifest 中有 LVO 路径也会忽略；CoW 模式拒绝人工真值来源。它仅校验声明，不能代替你审核 mask 来源。无 CoW 模型时可用阶段一 full-CTA 模型及 `--roi full`。训练和推理的 ROI 类型/CoW margin 必须一致。

## 6. 预测转换、定位与 FROC 评估

nnDetection 原始 restored 预测是本地 pickle，且坐标基于 SimpleITK 数组顺序。转换器读取 `*_boxes.pkl`，校验 `restore=True`，将 `(a1,b1,a2,b2,c1,c2)` 框中心从 z/y/x 转为 NIfTI x/y/z，再通过**实际导出 CTA affine**转成 RAS mm（含 CoW 裁剪偏移）。

```bash
cta-lvo convert-predictions /data/det_data/Task100_ISLES24LVO \
  /data/det_models/Task100_ISLES24LVO/MODEL/FOLD/test_predictions \
  --trusted-local --output /data/predictions.json
cta-lvo evaluate /data/audit.json /data/predictions.json \
  --splits /data/splits.json --subset test --distance-mm 10 \
  --score-threshold 0.5 --froc-thresholds 0.1,0.3,0.5,0.7,0.9 --output /data/metrics.json
```

请替换实际 MODEL/FOLD 目录。只对自己生成的可信本地 pickle 使用 `--trusted-local`；不可加载来源不明的 pickle。其他模型可直接提供 JSON：

```json
[
  {"patient_id": "sub001", "center_mm": [12.0, -15.0, 50.0], "score": 0.9}
]
```

同一病例预测按置信度排序，在物理距离阈值内与最近未匹配靶点一对一匹配。重复检测计为 FP；未输出预测的评估病例仍进入分母并计漏检；评估集合以 splits 为准，拒绝混入其他患者预测。

输出靶点定位敏感度、阳性病例定位敏感度、FP/例、匹配点平均距离、多置信度阈值的 FROC 工作点，以及有**经确认阴性病例**时的病例特异度。FROC 工作点不是插值后的官方挑战评分。默认评估类别无关定位；不要把代理框 IoU 或条件分类准确率解释成真实血栓范围/总体临床准确率。请在验证集选择阈值后冻结到测试集；没有独立阴性队列时不能声称验证了临床报警特异度。

## 7. 阶段三：审核后的血管节段分类

按 `audit.json` 中的连通域顺序，对每例每个靶点人工确认 BA / TICA / M1。创建审核文件：

```json
{
  "sub001": {
    "1": {"segment": "M1", "reviewed": true}
  }
}
```

`cta-lvo export ... --vessel-labels reviewed_segments.json` 会导出 BA=0、TICA=1、M1=2 多类别实例标签，训练命令与前面一致。遗漏审核或 `reviewed != true` 会报错；不支持的血管节段需要另行定义研究范围，不能强塞为这三类。

`cta-lvo evaluate ... --vessel-labels reviewed_segments.json` 额外报告**成功定位且有审核标签的靶点**上的条件分类准确率和混淆计数。漏检仍在定位指标中计入 FN，不能以条件分类准确率代替联合检测分类性能。代理框生成参数/连通域定义变化后必须重新核对审核编号。

## 测试覆盖

```bash
pytest -q
```

覆盖合成 HF bytes 结构的导入、CLI audit、异向性物理坐标、grid 不匹配拒绝、未知标注隔离、多连通域/重叠框、裁剪偏移、固定患者划分、独立 test 导出、nnDetection fold/类别格式、restored 框轴顺序转换、一对一评估、审核分类，以及不需要 LVO 的推理准备。未包括真实数据适配验证、GPU nnDetection 运行、自动 CoW 分割或论文性能复现。
