# 运动触发、空场景降频与 VAAPI 实验

2026-10-21。分支 `experiment/motion-vaapi` 从 `experiment/combined-optimizations` 的 `14d0df0` 创建。全部变体沿用两检测模型联合编译及原身份模型；神经网络计算只使用 TPU。

## 预定义对照

运动实验：1280×720、1920×1080；空画面、进入/离开、静止单人、缓慢淡入、光照变化。每种 60 帧、5 FPS 源时间戳、重复 3 轮，顺序随机，四组共 7,200 帧。每组预热 5 次完整推理。

- combined：上一分支的跟踪、身份缓存与联合模型。
- motion：空画面先检查低分辨率变化，最长 0.5 秒后强制重新检测。
- idle：仅将空画面完整检测间隔延长至 2 秒，**没有运动唤醒**；用于测量发现延迟的代价。
- motion_idle：运动唤醒与 2 秒空画面强制检测同时开启。

变化图最大 320×240，像素差阈值 18，变化占比阈值 0.002。与最后一次 TPU 确认的空帧比较，使缓慢变化能够累积。已有人员继续使用原时序策略，不因静止而删除，不扩大身份缓存期限。断流、尺寸、时间戳和身份库变化会使缓存失效；空画面复用仍检查 TPU 故障状态。

记录平均/P95、TPU 调用次数、逐帧输出、相对 combined 的人数/身份差异、首次发现延迟与离开后旧结果持续时间。均为公开图合成，不能据此声称真实视频准确率提高。

解码实验：FFmpeg 生成 1080p 和 4K 的 H.264 testsrc2，各 150 帧、25 FPS、CRF 23、veryfast、单线程编码。对比 OpenCV、FFmpeg CPU、FFmpeg VAAPI，以及两种 FFmpeg 路径缩放至 720p。每组重复 3 轮；不支持的路径保留错误记录，不回退 CPU。

整体实验：公开图合成空画面、进入/离开、静止单人 1080p H.264，各 150 帧、25 FPS。三种解码后端分别结合 combined、motion_idle；每 5 个源帧分析一次，每组重复 3 轮。共解码 8,100 帧、分析 1,620 帧。记录整段回放墙钟时间、父/解码子进程 CPU 时间、首帧时间、模型调用与识别输出。此处吞吐包含解码和 RGB 传输，不含网络；不等同实时摄像头延迟。

OpenCV 沿用其库内解码器；另设 FFmpeg CPU 控制组隔离不同库及进程管道的差异。CPU 控制组与 VAAPI 使用相同 PPM/RGB 管道。CPU 时间含父进程和已回收的解码子进程。解码计时中只做稀疏像素摘要，避免全帧哈希掩盖性能变化。

## 复现

按 [联合编译说明](cocompile.md)准备 `experiments/artifacts/co-person-first`，使用 Python 3.9 与同一依赖版本。

```bash
PYTHONPATH=. python -m experiments.motion_video --phase motion \
  --models experiments/artifacts/co-person-first --frames 60 --repeats 3 \
  --work /tmp/hd-motion-benchmark --output motion.json
PYTHONPATH=. python -m experiments.motion_video --phase decode \
  --models experiments/artifacts/co-person-first --frames 150 --repeats 3 \
  --work /tmp/hd-motion-benchmark --output decode.json
PYTHONPATH=. python -m experiments.motion_video --phase integrated \
  --models experiments/artifacts/co-person-first --frames 150 --repeats 3 \
  --work /tmp/hd-motion-benchmark --output integrated.json
```

视频仅为公开素材生成的临时测试文件，收集 JSON 与日志后删除，不保留用户摄像头图像。所有测试串行使用同一块 TPU，不部署常驻服务、不修改 main。

## 服务配置

```bash
HD_MODELS_DIR=experiments/artifacts/co-person-first \
HD_TEMPORAL_ENABLED=true HD_MOTION_ENABLED=true HD_EMPTY_SCENE_ENABLED=true \
HD_CAPTURE_BACKEND=vaapi human-detect serve
```

需要系统 FFmpeg、iHD 驱动和 `/dev/dri/renderD128` 访问权限。pn41 测试用临时 `setpriv` 进程赋予 render 主组，保留 apex 等附属组；不永久修改组成员，不以 root 部署服务。使用 `HD_CAPTURE_BACKEND=ffmpeg` 可运行同一管道的 CPU 解码控制组。显式 vaapi 路径失败时重连并返回状态，不切换软件解码；TPU 故障仍按原策略报错。

PPM 帧头携带实际尺寸，先校验像素上限再分配 RGB 内存。采集进程使用独立进程组，超时、删除流和关闭服务时同时回收 FFmpeg 子进程。静态图片 API 不使用运动复用。

容器提供可选 `compose.vaapi.yaml`：通过构建参数加入 FFmpeg/iHD，并映射 render 设备及数字 GID。先设置 `HD_APEX_GID`、`HD_RENDER_GID`，再使用 `docker compose -f compose.yaml -f compose.vaapi.yaml ...`。此次测量使用 Ubuntu 主机原有 FFmpeg/驱动，不代表 Debian 容器内驱动组合已经通过相同验证。该 overlay 沿用基础模型目录；如需联合模型，按组合分支说明挂载相应目录。

## 初测后的格式对照

初始协议使用 NV12→RGB 路径。初测发现该路径慢，另外记录 3 轮分段对照：GPU 表面、NV12 下载、NV12→RGB、NV12→I420→RGB，输出至 `/dev/null`，不计 Python 管道；命令和结果单独保存。后者的 I420 平面重排发生在 CPU，不能称作核显色彩转换。

增加显式 `vaapi-i420` 后端，并用相同视频、3 轮重复补测完整 RGB 回传和整体 pipeline；不覆盖初始数据，不重测已失败的 VPP 缩放路径：

```bash
PYTHONPATH=. python -m experiments.motion_video --phase decode \
  --backends vaapi-i420 --skip-scale --reuse-clips \
  --models experiments/artifacts/co-person-first --frames 150 --repeats 3 \
  --work /tmp/hd-motion-benchmark --output decode-i420.json
PYTHONPATH=. python -m experiments.motion_video --phase integrated \
  --backends vaapi-i420 --reuse-clips \
  --models experiments/artifacts/co-person-first --frames 150 --repeats 3 \
  --work /tmp/hd-motion-benchmark --output integrated-i420.json
```

`experiments.summarize_motion_video --extra` 可合并同阶段补测记录，断言相同素材和模型；保留两次源码版本。OpenCV 与外部 FFmpeg 的版本、线程策略、内存路径不同，须分别报告库内基线与相同管道的 CPU 对照。
