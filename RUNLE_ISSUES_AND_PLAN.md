# Runle Issues & Fix Plan (AvisTrack)

> **目的**：AvisTrack（分析管线）侧待办清单。与 `ChamberBroadcaster/RUNLE_ISSUES_AND_PLAN.md` 格式一致，跨 repo 决策互相引用。
>
> **上次更新**：2026-04-19

---

## 目录

| Section | 数量 |
|---|---|
| [已修复](#已修复) | — |
| [未修复](#未修复) | A1 |
| [批次实施规划](#批次实施规划) | 批次 A — 视频输入校验 |

---

## 已修复

（暂无）

---

## 未修复

### A1 — 分析入口必须拒读未 remux 的录像段

- **上游背景**：ChamberBroadcaster 录像为抗崩溃保留 fragmented MP4（见 `ChamberBroadcaster/RUNLE_ISSUES_AND_PLAN.md` Q1 / Q21）。录制结束后必须跑 ChamberBroadcaster 侧 `scripts/remux_after.py` 转封装才算完成。未 remux 的段：
  - Windows 自带播放器拖不动进度条
  - 剪辑工具随机访问慢
  - inference 体验差 / 边缘情况可能读到半帧
- **症状**：AVISTrack 若静默接受未 remux 文件 → 跑出来的 tracking 结果可能混入帧读取异常的伪样本，事后溯源代价高。
- **要求**：所有视频读取入口（dataset loader / sample clip 工具 / evaluation 脚本 / CLI / 任何 `cv2.VideoCapture` / `decord` / `ffmpeg-python` 调用点）在打开文件前必须通过统一 validator，做 **3 条 hard-fail 检查**。任一成立 → `raise`，**不 warn / 不跳过 / 不静默降级**：

  1. **文件名以 `FRAGMENTED_` 开头** → 这个段还没被 `scripts/remux_after.py` 处理过。
  2. **文件所在目录存在 `!!_PENDING_REMUX_README.txt`** → 整个目录里至少有一段未处理；即便当前这段看起来没 `FRAGMENTED_` 前缀（例如用户手工改过名），也拒绝整目录以免不一致。
  3. **同目录 `timestamp_calibration.jsonl` 里对应 `video_file` 最新状态是 `remux_status=pending` 且没有后续 `reason=remux_complete` 行** → 记账本层面未完成。

- **为什么三条都要（OR 关系）**：
  - 只查文件名：用户手工改名可绕过
  - 只查 txt：孤立复制的单个文件不带 txt
  - 只查 jsonl：老录像 / 非标准路径可能没 jsonl
  - 三条互补，任一成立即拒

- **错误消息模板**（英文，便于跨成员复制排障）：
  ```
  Refusing to read {path}: recording segment has not been remuxed.
  Trigger: {which of the 3 checks fired}
  Fix:
    conda activate chamber_broadcaster
    cd <ChamberBroadcaster repo>
    python scripts/remux_after.py "{parent_dir}"
  Background: AvisTrack/RUNLE_ISSUES_AND_PLAN.md#a1
             ChamberBroadcaster/RUNLE_ISSUES_AND_PLAN.md (Q1 / Q21)
  ```

- **不做降级选项**（例如 `ALLOW_FRAGMENTED=1` 环境变量）：contract 的价值来自刚性。任何"临时跳过"开关都会变成默认开着 → 伪样本重新潜入。要用 fragmented 文件做调试的人可以手动 remux 一份。

- **文件**（待定，首次实现时枚举）：统一 helper `validate_recording_path(path)`，所有视频读入口强制调用。

---

## 批次实施规划

### 批次 A — 视频输入校验（A1）

1. **枚举视频读入口**：grep 本 repo 找 `cv2.VideoCapture` / `decord` / `ffmpeg-python` / `imageio` / 任何自定义 video reader 用法；dataset loaders、sample clip 工具、evaluation 脚本、CLI 入口都列出来。
2. **新建 helper** `validate_recording_path(path: Path) -> None`：
   - 实现 3 条 hard-fail 检查
   - 失败抛 `RecordingNotRemuxedError(path, trigger, fix_cmd)`，异常类继承 `RuntimeError`，`__str__` 走上面的错误消息模板
3. **所有入口强制通过 helper**，删除重复 / 不完整的检查。
4. **集成测试**：
   - 构造 `FRAGMENTED_dummy.mp4` → 被拒（trigger: filename）
   - 构造 `dummy.mp4` + 同目录 `!!_PENDING_REMUX_README.txt` → 被拒（trigger: pending txt）
   - 构造 `dummy.mp4` + 同目录 `timestamp_calibration.jsonl` 含 `{"video_file": "dummy.mp4", "remux_status": "pending"}` 无后续 `remux_complete` → 被拒（trigger: jsonl pending）
   - 同上但 jsonl 后续有一行 `{"video_file": "dummy.mp4", "reason": "remux_complete"}` → 通过
   - 合法 `dummy.mp4` 无 txt / jsonl 清洁 → 通过

### 决策记录

- **A1**（2026-04-19）：ChamberBroadcaster Q21 的 04-19 版决策选择"broadcaster 落 marker + 独立脚本 remux"（详见上游 md）。分析侧用 3 条 hard-fail 检查兜底：文件名 / txt 存在 / jsonl status。三条 OR，任一成立即拒。错误消息指向上游 `scripts/remux_after.py`。拒绝提供"跳过校验"的降级开关 —— contract 的价值来自刚性。

---

### 关键文件索引

| 问题 | 核心文件 |
|---|---|
| A1 | 本 repo 所有视频读入口（批次 A 首次实现时枚举）; 新建 `validate_recording_path(path)` helper; 新异常 `RecordingNotRemuxedError` |
