# FaceCluster-FM

Windows 单机离线头像包分层分类排序工具：按人脸身份聚簇 → 按属性硬分区排序 → 批量重命名输出，并提供 SigLIP2 发型三模式与 PySide6 桌面 GUI。

- **离线优先**：全部推理在本地完成（DirectML GPU 加速，CPU 兜底），仅模型下载需联网一次
- **缓存驱动**：特征缓存以图片内容哈希（blake2b）寻址，重跑/换键序/切模式秒级重放，零重复计算
- **可回滚**：重命名支持 copy 副本（默认）与 inplace 两阶段改名 + 日志回滚

## 功能总览

| 里程碑 | 能力 | 入口 |
|---|---|---|
| M0 标注管线 | 人脸检测+识别、发区/皮肤掩码、三属性（发色/肤色 0-19 级/发长）、pHash | `annotate` / `report` |
| M1 身份聚簇 | 代表点法聚簇（防传递串簇）、medoid、簇属性众数、坏脸门控落尾 | `cluster` |
| M2 分区排序 | 键序层级硬分区、段内贪心最近邻链、F 值报告、批量重命名 | `order` / `rename` |
| M3 发型+GUI | SigLIP2 嵌入三模式（固定清单/聚类/示例标注）+ 桌面端 | `hairstyle` / `gui` |

## 环境要求

- Windows 10/11，Python 3.12
- GPU：任何支持 DirectML 的显卡（可选，无卡自动走 CPU）
- 磁盘：模型约 1.5GB（视觉塔 355MB 必须；文本塔 1.1GB 仅发型模式 A 需要）

## 安装

```powershell
git clone https://github.com/huangzhanhao/FaceCluster-FM.git
cd FaceCluster-FM
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

下载发型模型（联网一次）：

```powershell
.venv\Scripts\python -m facecluster.cli fetch-model          # SigLIP2 视觉塔 ~355MB
.venv\Scripts\python -m facecluster.cli fetch-model --text   # 文本塔 ~1.1GB，仅模式 A 需要
```

下载失败自动回退 hf-mirror.com 镜像；也可设 `HF_ENDPOINT` 后重试。

## 快速开始

### 命令行全流程

```powershell
# 1. 标注：扫描 样本/ 目录，产出 out/annotations.jsonl
.venv\Scripts\python -m facecluster.cli annotate

# 2. 身份聚簇 → out/clusters.json
.venv\Scripts\python -m facecluster.cli cluster

# 3. 分区排序（默认键序 肤色→发长→发色）→ out/plan.json + F 值报告
.venv\Scripts\python -m facecluster.cli order

# 4. 批量重命名（copy 副本模式，输出到 样本_sorted/）
.venv\Scripts\python -m facecluster.cli rename --dry-run   # 先预演
.venv\Scripts\python -m facecluster.cli rename             # 实际执行
```

### 图形界面

```powershell
.venv\Scripts\python -m facecluster.gui
```

- **键序页**：勾选/拖拽分区键，改动自动触发重排（纯内存秒级），F 值实时刷新
- **发型页**：三模式切换（嵌入缓存共用），一键应用为分区键
- **聚簇页**：medoid 代表图网格，多选合并 / 单选拆分，改完自动重排
- **批量重命名**：前后名对照表，冲突/漂移红标，copy 与 inplace 可选

## 发型三模式

发型分组基于 SigLIP2 ViT-B/16 发区嵌入（非发区置灰，768 维），嵌入只算一次，三模式切换零重算：

| 模式 | 做法 | 适用 |
|---|---|---|
| A 固定清单 | 20 类中文展示 ← 英文提示词零样本匹配 | 想要稳定命名类目 |
| B 嵌入聚类（默认） | k-means++，K 自动按轮廓系数（2~12）搜索 | 不确定有哪几类发型 |
| C 示例标注 | 给几张示例图，其余按最近锚点分配 | 想完全自定义组语义 |

生成 `out/hairstyle.json` 后，把「发型」加入分区键即可参与排序（GUI 中一键应用，或 CLI）：

```powershell
.venv\Scripts\python -m facecluster.cli order --keys 肤色,发长,发型
```

## 数据流与工件

```
样本/*.png ──annotate──▶ out/annotations.jsonl（逐图记录，嵌入/掩码进缓存）
                │
                └─cluster──▶ out/clusters.json（簇号 1..N 连续，尾部：坏脸/无人脸/失败）
                        │
        hairstyle ──▶ out/hairstyle.json（图级发型标签，作键时按簇叠加）
                        │
                └─order────▶ out/plan.json（唯一改名依据：位次/段路径/新名/sim/F 值）
                        │            └─▶ out/sort_report.html + sort_plan.csv
                        └─rename───▶ 样本_sorted/（copy）或原地（inplace）
                                     out/rename_journal.json（inplace 回滚依据）
```

- 缓存位于 `~/.facecluster/cache/features/{内容哈希}.npz`（嵌入、掩码、发型嵌入），改名后依然命中
- 模型位于 `~/.facecluster/models/`（SigLIP2）与 `~/.insightface/models/`（buffalo_s）

## 重命名安全机制

- **copy 模式（默认）**：原包不动，输出到同级 `{目录名}_sorted/`；目标名被占用且内容不同时拒绝执行
- **inplace 模式**：两阶段改名（先全部改临时名，再改最终名，防互相覆盖），journal 先行落盘；中途失败自动逆向回滚
- **漂移检测**：执行前按 size/mtime 快检，与 plan 不符的文件跳过不动
- 回滚：`rename --rollback`

## 项目结构

```
facecluster/
├── cli.py            # CLI 入口（annotate/cluster/order/rename/hairstyle/fetch-model）
├── config.py         # 全部可调参数 + 分区键注册表
├── cluster.py        # M1 身份聚簇
├── partition.py      # M2 硬分区 + 排序
├── rename.py         # M2 重命名/回滚
├── hairstyle.py      # M3 SigLIP2 发型三模式
├── adjust.py         # GUI 人工合并/拆分簇
├── modelhub.py       # 模型下载（镜像回退）
├── annotate/         # M0 标注管线（人脸/掩码/属性/哈希）
└── gui/              # PySide6 桌面端（app/workers/panels/table_model/thumbs）
tools/                # 接触表、GPU 自检
docs/                 # 需求定稿、M0 PoC 报告
```

## 调试

仓库已含 `.vscode/launch.json`：GUI 桌面端与全部 CLI 子命令的调试配置，解释器指向 `.venv`，在 Trae / VS Code 的「运行和调试」面板直接选用。

## 已知边界

- 混合光照下深浅肤色的残余混淆记录于 [docs/M0-PoC报告.md](docs/M0-PoC报告.md)，发型模式 B 的组语义需人工看图核对（GUI 聚簇页/发型预览）
- 人工合并/拆分簇（GUI）是对当前 clusters.json 的会话级修正，重跑 `cluster` 会覆盖
