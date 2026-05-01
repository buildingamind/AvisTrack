# AvisTrack 多-Chamber 存储架构重构

## Context

**问题：** 当前 pipeline (`configs/wave2_collective.yaml`) 假设 sample / 标注 / 训练数据集 / 模型权重全部和原始视频在同一个 `drive.root` 下。这个假设在新场景下被打破：

- 每个 chamber type 有多个物理 chamber，每个 chamber 一块独立外置盘
- 同 chamber type 要训练一个统一模型 → 需要跨多块盘汇总数据
- 训练机一次只能挂一块盘
- 老 wave 没有 `camera_rois.json` / `valid_ranges.json` / `time_calibration.json`，且路径混乱
- repo 是 pipeline 代码，不放任何数据

**目标：** 把"源数据存储"和"训练工作区"在架构上分开，让标注 / 训练 / eval 不再依赖任何源盘在线，并补足老 wave 的元数据补全工具链。

**用户已对齐的关键决策：**
- Workspace 物理位置：始终插着的外置 SSD
- 老 wave 要参与训练，需要专门补元数据工具
- 标注流程：外部 CVAT → 导出 YOLO txt → pipeline 只消费

---

## 1. 存储两层架构

```
┌─ 源盘 (chamber drive，间歇性挂载，每个 chamber 一块) ───────────┐
│ {chamber_root}/                                                  │
│   ├── _avistrack_source.yaml          ← 标记: 我是哪个 chamber  │
│   │      chamber_id: collective_104A                             │
│   │      chamber_type: collective                                │
│   │      drive_uuid: <volume-serial>                             │
│   ├── {wave_id}/                       ← 每个 wave 一个文件夹    │
│   │   ├── 00_raw_videos/...                                      │
│   │   └── 02_Chamber_Metadata/                                   │
│   │       ├── camera_rois.json                                   │
│   │       ├── valid_ranges.json                                  │
│   │       ├── time_calibration.json                              │
│   │       └── ocr_roi.json                                       │
│   └── _avistrack_added/                ← 给老 wave 补的元数据    │
│       └── {legacy_wave_id}/                                      │
│           ├── video_index.json         ← 扫描出来的视频清单      │
│           ├── camera_rois.json                                   │
│           ├── valid_ranges.json                                  │
│           └── time_calibration.json                              │
└──────────────────────────────────────────────────────────────────┘
                          ↓ sample 时把 clip 写入工作区
┌─ 中央工作区 (外置 SSD，始终在线) ───────────────────────────────┐
│ {workspace_root}/                                                │
│   └── {chamber_type}/                  ← 每个 chamber type 一个 │
│       ├── workspace.yaml                                         │
│       ├── sources.yaml                  ← 注册的所有源盘          │
│       ├── clips/                        ← 原始库存,永远只增不删  │
│       │   └── {chamber_id}/{wave_id}/*.mp4                       │
│       ├── annotations/                  ← 原始库存                │
│       │   └── {chamber_id}/{wave_id}/*.txt    (YOLO 格式)        │
│       ├── manifests/                                             │
│       │   └── all_clips.csv     (唯一真相 clip 清单 + 元信息)    │
│       ├── datasets/                    ← 多份 dataset 并存,各从  │
│       │   │                               clips/+annotations/ 筛 │
│       │   ├── full_v1/                                           │
│       │   │   ├── recipe.yaml    (包含/排除/split/seed)         │
│       │   │   ├── manifest.csv   (实际入选的 clip)              │
│       │   │   ├── images/{train,val,test}/  (软链接)            │
│       │   │   ├── labels/{train,val,test}/  (软链接)            │
│       │   │   └── data.yaml                                     │
│       │   ├── chambers_1and2_only/                              │
│       │   └── exclude_legacy_v1/                                │
│       └── models/                                                │
│           ├── index.csv         (全实验台账,自动维护)           │
│           └── {experiment_name}/                                 │
│               ├── meta.json     (dataset_name + recipe_hash +   │
│               │                  git_sha 永久绑定)              │
│               ├── snapshots/    (训练开始那刻的不可变副本)      │
│               │   ├── experiment.yaml                           │
│               │   ├── recipe.yaml                               │
│               │   ├── data.yaml                                 │
│               │   └── uncommitted.diff (如果 git_dirty)         │
│               ├── phase1/                                        │
│               │   ├── runs/{model_name}/weights/best.pt + ...   │
│               │   ├── leaderboard.csv (各 run 指标聚合)         │
│               │   └── winner.json                               │
│               ├── phase2/...                                     │
│               ├── phase3/...                                     │
│               ├── final/                                         │
│               │   ├── best.pt                                   │
│               │   └── source.json (来自哪个 phase/run)          │
│               └── eval/                                          │
│                   └── {test_dataset_name}/                       │
│                       ├── per_clip_results.csv                  │
│                       ├── summary.csv                           │
│                       └── eval_config.yaml                      │
└──────────────────────────────────────────────────────────────────┘

推理输出 (cli/run_batch.py 写入,无 _meta.json 拒绝输出):
{batch_output_root}/{batch_run_name}/
├── _meta.json   (weights, experiment_name, chamber_id, wave_id,
│                 git_sha, started_at, ended_at, tracker_config)
├── {video}.parquet ...
└── log.txt
```

**核心原则：**
- clip 进了工作区就和源盘脱钩，标注/训练/eval 全程不需要源盘
- 每条 clip 用 `chamber_id + wave_id + 原始视频路径` 做溯源（写在 `manifests/all_clips.csv`）
- 源盘上 `_avistrack_source.yaml` + workspace 上 `sources.yaml` 双向锚定，挂载新盘自动识别
- **`clips/` + `annotations/` 是不可变库存；`datasets/` 是从库存筛出的视图**（用 recipe.yaml 描述筛选规则）。同一 chamber type 可同时存在多个 dataset 并各自训练，互不污染
- 每个 model 的 `meta.json` 永久绑定它用的 `dataset_name` + recipe_hash + git_sha，回头可重现
- **完整 lineage 链**: `recipe.yaml → datasets/{name}/ → models/{exp}/ → eval/{test}/ + batch_outputs/{run}/_meta.json`，任意 artifact 可双向追溯
- **训练快照不可变**: 训练开始时把 `experiment.yaml + recipe.yaml + data.yaml + git diff` 拷到 `models/{exp}/snapshots/`，源文件之后被改不影响溯源
- **experiment_name 不允许重名覆盖**: `run_pipeline.py` 启动时检测到已存在直接报错

---

## 2. 配置层重构

### 旧（要废弃）
```yaml
drive:
  root: "/media/woodlab/104-A/Wave2"
  raw_videos:    "{root}/00_raw_videos"
  dataset:       "{root}/01_Dataset_MOT_Format"
  ...
```
单一 `drive.root` 模板把所有路径绑死。

### 新

**(a) 工作区配置**（中央工作区里一份，描述这个 chamber type 的全局信息）
```yaml
# {workspace_root}/{chamber_type}/workspace.yaml
chamber_type: collective
workspace:
  root: "{workspace_root}/{chamber_type}"        # 在 loader 里解析
  clips:        "{root}/clips"
  annotations:  "{root}/annotations"
  manifests:    "{root}/manifests"
  dataset:      "{root}/dataset"
  models:       "{root}/models"
chamber:
  n_subjects:  9
  fps:         30
  target_size: [640, 640]
```

**(b) 源盘注册**（中央工作区里一份）
```yaml
# {workspace_root}/{chamber_type}/sources.yaml
chamber_type: collective
chambers:
  - chamber_id: collective_104A
    drive_uuid: "ABCD-1234"            # 自动 probe，重挂能识别
    drive_label: "104-A"
    waves:
      - wave_id: wave2
        layout: structured
        wave_subpath: "Wave2"
        raw_videos_subpath: "{wave_subpath}/00_raw_videos"
        metadata_subpath: "{wave_subpath}/02_Chamber_Metadata"
      - wave_id: wave1_legacy
        layout: legacy
        wave_subpath: "Wave1_OldDump"
        raw_videos_glob: "**/*.mp4"
        metadata_subpath: "_avistrack_added/wave1_legacy"
```

**(c) 实验配置**（repo 里 / 工作区里都行，描述训练或推理实验）
```yaml
# train/experiments/W2_collective_phase1.yaml
chamber_type: collective
workspace_yaml: "{workspace_root}/collective/workspace.yaml"
sources_yaml:   "{workspace_root}/collective/sources.yaml"
experiment_name: W2_collective_phase1
training:
  epochs: 300
  models: [yolov8n, yolov8s, yolov8m, ...]
```

**Loader 改动 (`avistrack/config/loader.py`)：**
- 当前 `_resolve_placeholders()` 只处理 `{root}` → `drive.root`
- 改为支持三个独立模板变量：`{workspace_root}` / `{chamber_root}` / `{wave_subpath}`
- 加 sanity check：`workspace_root` 不能是 git repo 内路径
- `chamber_root` 在加载源盘配置时通过 drive_uuid 自动 probe 到当前挂载点（如果命中）；命中不到给清晰错误，提示哪些操作需要哪个 chamber 在线

---

## 3. 文件清单

### 新建文件

| 路径 | 作用 |
|---|---|
| `configs/chamber_type_sources_template.yaml` | sources.yaml 的模板（用户起草过，我们正式化） |
| `configs/workspace_template.yaml` | workspace.yaml 的模板 |
| `tools/init_chamber_workspace.py` | 创建一个 chamber type 的工作区目录结构 + workspace.yaml + 空 sources.yaml |
| `tools/register_chamber_source.py` | 检测当前挂载的盘，写入 `_avistrack_source.yaml` 到源盘 + 把它登记到 workspace 的 sources.yaml |
| `tools/scan_legacy_wave.py` | 老 wave 专用：扫盘找视频 → video_index.json，然后引导用户用 `pick_rois` / `edit_valid_ranges` 在 `_avistrack_added/{wave}/` 下补元数据 |
| `tools/import_annotations.py` | 把 CVAT 导出的 YOLO txt（按 clip 命名）拷到 `{workspace}/{chamber_type}/annotations/{chamber}/{wave}/` |
| `tools/build_dataset.py` | 入参 `--recipe path/to/recipe.yaml`：按 recipe 从 `clips/` + `annotations/` 筛选,生成 `datasets/{name}/` (含 manifest.csv + 软链接 images/labels + data.yaml)。同一 chamber type 可建多个 dataset |
| `tools/list_clips.py` | 浏览 `manifests/all_clips.csv` 帮写 recipe (按 chamber/wave/标注状态过滤查看) |
| `configs/recipe_template.yaml` | recipe.yaml 模板 (include/exclude/split 字段) |
| `tools/list_experiments.py` | 列 `models/index.csv`,可按 chamber_type/dataset_name/日期过滤,默认按时间倒序 |
| `tools/show_lineage.py` | 输入任意 artifact (best.pt / *.parquet / eval/summary.csv),输出从 recipe → dataset → experiment → eval/output 完整链 |
| `tools/compare_experiments.py` | 并排比较多个 experiment 的 dataset 差异 + phase 指标 |
| `tools/rebuild_index.py` | 从 `models/*/meta.json` 扫描重建 `index.csv` (索引坏了的兜底) |
| `avistrack/lineage.py` | 公共 lineage 工具库: hash recipe.yaml、读写 meta.json、git_sha+dirty 探测、index.csv 增量维护 |
| `docs/chamber_type_workspace.md` | 文档：架构图 + 标准操作流程（新盘怎么接入、老 wave 怎么处理、怎么开新实验） |

### 修改文件

| 路径 | 改动 |
|---|---|
| `avistrack/config/loader.py` (lines 37-39 附近的 `_resolve_placeholders`) | 支持 `{workspace_root}` / `{chamber_root}` / `{wave_subpath}`；加 repo 路径 sanity check；加 chamber drive UUID probe |
| `tools/sample_clips.py` | 入参改为 `--workspace-yaml + --sources-yaml + --chamber-id + --wave-id`；输出 clip 到 `{workspace}/{chamber_type}/clips/{chamber}/{wave}/`；同步 append 到 `manifests/all_clips.csv` |
| `tools/edit_valid_ranges.py` | 写到 chamber 元数据目录（`{chamber_root}/{wave}/02_Chamber_Metadata/` 或 `_avistrack_added/{wave}/`），不再去工作区 |
| `tools/pick_rois.py` | 同上 |
| `train/run_train.py` | data.yaml 路径来自 `datasets/{dataset_name}/data.yaml`,输出到 `models/{experiment}/phase{N}/`;启动时写 meta.json + snapshots/;删除所有 `drive.root` 引用 |
| `train/run_pipeline.py` | 三阶段 phase 之间的 best.pt 路径都从 `models/{exp}/phase{N}/` 内解析;每个 phase 完成时写 `leaderboard.csv` + `winner.json`;最后产出 `final/best.pt` + `source.json`;完成后 append 一行到 `models/index.csv` |
| `eval/run_eval.py` | 测试集从 `datasets/{dataset_name}/test/` 读,weights 从 `models/{exp}/final/best.pt` 读,输出到 `models/{exp}/eval/{test_dataset_name}/` 并写 `eval_config.yaml` |
| `cli/run_batch.py` | 推理时 **同时**需要 `chamber_root`(读视频)+ `workspace_root`(读权重);**强制**写 `_meta.json` 否则拒绝输出;输出可选写工作区或源盘 |

---

## 4. 标准操作流程（文档化到 `docs/chamber_type_workspace.md`）

**第一次设置 chamber type：**
```
python tools/init_chamber_workspace.py --workspace-root /media/wkspc \
    --chamber-type collective
```

**接入一块新源盘（每块新盘一次）：**
```
# 插上新盘
python tools/register_chamber_source.py --workspace-root /media/wkspc \
    --chamber-type collective --chamber-id collective_104A
# 工具会自动 probe drive UUID，写两侧 yaml
```

**结构化新 wave：**
```
# 视频已经在 {chamber_root}/{wave}/00_raw_videos/，元数据自带或手动放
python tools/sample_clips.py --workspace-root /media/wkspc \
    --chamber-type collective --chamber-id collective_104A --wave-id wave2
```

**老 wave（无元数据）：**
```
python tools/scan_legacy_wave.py --chamber-id collective_104A --wave-id wave1_legacy
# 工具扫盘，提示运行:
#   tools/pick_rois.py --legacy --chamber-id ... --wave-id ...
#   tools/edit_valid_ranges.py --legacy --chamber-id ... --wave-id ...
# 然后:
python tools/sample_clips.py ... --chamber-id collective_104A --wave-id wave1_legacy
```

**标注（CVAT 导出后）：**
```
python tools/import_annotations.py --workspace-root /media/wkspc \
    --chamber-type collective --cvat-export ./cvat_export.zip
```

**构建数据集 + 训练：**
```
# 先写一份 recipe.yaml (从 configs/recipe_template.yaml 复制改)
# 例如: datasets_recipes/full_v1.yaml 描述用哪些 chamber/wave/split 比例
python tools/build_dataset.py --workspace-root /media/wkspc --chamber-type collective \
    --recipe datasets_recipes/full_v1.yaml
# 生成 {workspace}/collective/datasets/full_v1/

# 实验配置里指定 dataset_name,训练就跑这份 dataset
python train/run_pipeline.py --experiment train/experiments/W2_collective_phase1.yaml
# 实验配置含: dataset_name: full_v1
# 输出落到 {workspace}/collective/models/W2_collective_phase1/, meta.json 记录 dataset_name + git sha
```

想换数据组合训练？写新 recipe.yaml → build → 新建 experiment yaml 指向新 dataset_name，互不污染。

源盘从此可以拔掉。整个训练 + eval 不需要任何源盘在线。

---

## 5. 实施阶段建议（每阶段独立可验证）

| 阶段 | 内容 | 验证 |
|---|---|---|
| **A. 配置层** | 改 `loader.py` 支持新模板变量；加模板 yaml；加 repo 路径 sanity check | 单测 + 加载现有 wave2_collective.yaml 不应破坏（暂时通过 chamber.root 别名兼容） |
| **B. 工作区骨架** | `init_chamber_workspace.py` + `register_chamber_source.py` | 跑一次：能在外置 SSD 上初始化目录树 + 注册当前 wave2 盘 |
| **C. Sample 迁移** | `sample_clips.py` 入参改，输出到工作区，写 manifest | 用 wave2 那块盘跑，clip 落在工作区，manifest 行数对得上 |
| **D. 标注 + 数据集构建** | `import_annotations.py` + `build_dataset.py --recipe` + `list_clips.py` + `recipe_template.yaml` | 拷一份现有 W2_iou096_v1 标注进来,写 recipe 跑 build,生成的 data.yaml 跑 ultralytics yolo val 不报错;再写一份只包含 chambers_1and2 的 recipe,build 出第二个 dataset 共存 |
| **E. 训练 + eval 切换 + lineage** | `run_train.py` / `run_pipeline.py` / `run_eval.py` 改读 workspace;`avistrack/lineage.py` + `list_experiments.py` + `show_lineage.py` + `rebuild_index.py` | 跑一次 phase1 几个 epoch,确认权重落在 `models/{exp}/phase1/`,`meta.json` + `snapshots/` 齐全;run_eval 后 `eval/{test}/` 有 summary;`list_experiments.py` 列出该实验;`show_lineage.py final/best.pt` 输出从 recipe 到 model 的完整链 |
| **F. 老 wave 工具** | `scan_legacy_wave.py` + `pick_rois.py --legacy` + `edit_valid_ranges.py --legacy` | 找一块老盘跑通：扫盘 → 补元数据 → sample → 加入 dataset |
| **G. 批量推理 + 输出 lineage** | `cli/run_batch.py` 双 root 支持 + 强制写 `_meta.json` | 拿训好的模型对一块源盘的视频跑 → parquet 输出落到工作区,同目录 `_meta.json` 有 weights/experiment_name/git_sha;`show_lineage.py output.parquet` 能从 parquet 反查到 experiment 和 dataset |

阶段之间 PR 可独立合入，旧 wave2 工作流在 A-E 期间通过兼容层不受影响（loader 检测到老式 `drive` section 就走老路径）。E 完成后即可删除兼容层。

---

## 6. 关键注意事项

1. **drive UUID probe**（Windows / Linux 双平台）：Windows 用 `wmic logicaldisk get VolumeSerialNumber`，Linux 用 `blkid` / `/dev/disk/by-uuid/`，loader 抽象掉差异
2. **clip 文件名要含 chamber_id + wave_id**（避免不同源盘的 clip 在工作区撞名）
3. **manifests/all_clips.csv 是溯源真相**：列至少包含 `clip_path, chamber_id, wave_id, source_video, source_drive_uuid, sampled_at`
4. **CVAT 导入命名约定**：`import_annotations.py` 必须能从 txt 文件名匹配回 clip（用 clip basename），不匹配的 txt 报错而非静默
5. **老 wave 的 `wave_id` 命名**：建议 `wave{N}_legacy` 区分，避免和结构化 wave 冲突
6. **build_dataset 的切分**：默认按 `recipe.split.stratify` (chamber/wave/none) 分层抽样而不是全局随机，保证 train/val/test 都包含每类样本（防止某 chamber 全进 val 导致泛化评估偏差）。recipe 提供 `seed` 字段保证可重现
7. **datasets/ 用软链接节省空间**：Linux `ln -s`,Windows `mklink /J` (junction)。需要在 `build_dataset.py` 里抽象掉平台差异。万一软链接不可用 (跨设备/权限)，退化为 hardlink → copy 三级 fallback
8. **不要把 workspace 路径硬编码进 repo**：所有指向 workspace 的路径必须从 yaml 来；CI / 测试用 tmpdir
