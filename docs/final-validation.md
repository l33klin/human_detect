# main 合并与最终验证

2026-10-21，在 `experiment/motion-vaapi` 的 `ce4e4b4` 基础上整理，合并目标为原 main `2775693`。验证使用 pn41 的 N6005 与单块 PCIe Coral；未部署常驻服务。

## 最终保留的方案

- 默认启用视频跟踪和有期限的身份特征复用：每 5 个分析帧或最多 0.5 秒刷新检测，身份特征最多复用 1 秒。图片接口始终完整推理。
- 默认启用已确认空场景的复用，始终附带运动唤醒；最长 2 秒兜底检测。不会因为人员静止就将其删除。
- 视频采集统一使用 FFmpeg，在 RGB 转换前按目标分析 fps 筛帧。默认 CPU 解码；可显式选择强制 VAAPI 核显解码，保留已验证的 I420 布局转换路径。I420 转换发生在 CPU。
- 稳定少人场景可使用 `scripts/prepare_joint_models.py` 准备人体、人脸联合模型，或通过 `compose.stable.yaml` 挂载；身份模型不变。多人场景保留原单独编译模型。
- 三个神经网络仍只在 TPU 上执行。身份库仅保存特征与身份信息；生产服务不保存图像。

删除了无运动唤醒的空场景降频、未前置筛帧的独立解码路径、直接 NV12→RGB 的慢路径、未跑通的 VPP 缩放、原 OpenCV 视频采集，以及反向联合编译等实验驱动代码。历史报告与数据保留在 `docs/experiments`，完整历史复现代码保留在实验分支。main 只保留模型准备和有效方案验证脚本。

增加按 Interpreter→Delegate 顺序释放模型，并在启动部分失败时释放已经加载的模型，避免原生资源在退出时以不确定顺序清理。

## 验证结果

本地与 pn41 均通过 **51 项测试**，ruff 检查和格式检查通过。Compose 的基础、VAAPI、联合模型 overlay 合并校验通过，确认包含 apex/render 两个设备与 GID，以及联合模型的只读挂载。验证服务器的 36 个源码与配置文件 SHA-256 与本地完全一致。

真实硬件验证包含：

- 三个联合配置模型的真实 TPU 预热和计算图审计通过。指定不存在的 `pci:99` 时退出码为 1，拒绝 CPU 神经网络回退。
- HTTP H.264 1080p 视频 → 真实采集子进程 → TPU → 已录入身份 JSON：CPU 解码、VAAPI 解码均通过。
- 流运行中删除已录入身份，下一次新结果不再沿用该身份；正常删除流与强制结束采集父进程后，采集线程和 FFmpeg 子进程均回收。
- 指定不存在的 render 设备，核显解码以非零状态失败，不回退软件解码。
- 20 次静态图片分析识别到已录入参考身份；删除后返回 unknown。默认 CPU HTTP MJPEG 流运行 8.01 秒处理 35 帧，含启动时间约 4.37 FPS，未出现队列拒绝。该速率受目标 5 FPS 限制。

所有身份样本均为公开参考图。临时视频仅用于验证，未使用用户摄像头图片。render 权限只通过临时 setpriv 进程授予，没有修改永久组成员或系统驱动。

## 最终性能复测

1080p、5 FPS 源时间戳，每场景每组 30 帧、3 轮，预热 5 帧，顺序随机；五场景、三配置共 **1,350 帧**。只测识别 pipeline，不含解码、网络或实时限速。基线为原预编译模型逐帧完整推理。

| 场景 | 原逐帧流程 ms | 优化＋单独模型 ms | 优化＋联合模型 ms |
| --- | --- | --- | --- |
| 稳定空画面 | 8.775 | 1.929 | 2.646 |
| 静止单人 | 61.832 | 31.076 | 27.156 |
| 进入／离开 | 26.527 | 11.024 | 11.206 |
| 移动单人 | 60.541 | 30.893 | 28.210 |
| 多人拼接 | 142.622 | 66.717 | 71.674 |

静止单人联合方案较原逐帧流程平均耗时下降约 56.1%；移动单人下降约 53.4%。空场景三轮人体调用由 90 次降至 9 次。多人采用单独模型优于联合模型约 6.9%（以联合方案耗时为分母）。因此联合模型保持场景可选，不强制覆盖所有视频。

所有配对帧相对基线的额外正身份、缺失正身份、有人／无人差异均为 0。进入场景三组、三轮均在帧索引 10（从 0 开始）首次发现人员，没有新增相对基线的发现延迟。

这些结果反映合成回放的平均处理开销和基线一致性，不能证明真实摄像头准确率。完整刷新仍有耗时峰值；跟踪可能出现框漂移、短时沿用旧身份，小目标或弱运动可能到周期检测才发现。P95、阶段调用和逐帧输出见原始数据。

## 镜像与部署

最终镜像使用 Python 3.9 / Debian Bookworm。补充构建发现原 Bullseye 软件源出现 404，因此更新基础镜像；保留 Coral 所需的 Python 3.9 与 `tflite-runtime==2.5.0.post1`。FFmpeg 成为基础依赖，iHD 仍通过 `WITH_VAAPI=true` 选择。Coral APT 入口实际返回 403，因此镜像改为下载官方固定版本运行时并验证 SHA-256；安装的标准版 x86_64 库与 pn41 已验证库的 SHA-256 完全一致。二进制许可证随库保留在镜像中。

最终镜像已在 pn41 原生 x86_64 上构建成功，使用可选 Debian/PyPI 镜像站参数完成下载。镜像 ID 为 `sha256:05e1535301312c9bc64b40dd916b5cf80315292db0414674327050cd01bddc15`。

容器内以 UID 10001、只读根文件系统、tmpfs 临时目录、apex/render 设备和对应附属组运行，未使用 privileged 或 root 应用用户。容器中的真实 TPU 预热、CPU/VAAPI 两种 HTTP H.264 视频、身份删除缓存失效、采集父进程异常清理，以及错误核显设备无软件回退均通过。容器 FFmpeg 为 5.1.9、iHD 为 23.1.1、TFLite Runtime 为 2.5.0.post1。额外的 HTTP 测试依赖只安装在临时容器 tmpfs 中，没有修改发布镜像。验证容器运行结束后自动删除。

本次同时验证了 Ubuntu 原生和 Bookworm 容器路径。性能表来自 Ubuntu 原生回放，不能直接当成容器吞吐结果。

## 数据与重现

- [版本、模型与服务器源码校验和](validation/data/final-environment.json)
- [本地源码校验和与合并来源](validation/data/source-metadata.json)
- [性能汇总](validation/data/final-replay-summary.json) · [逐帧原始数据](validation/data/final-replay.json.gz)
- [H.264 API 与进程清理](validation/data/final-video-api.json)
- [静态图片与 MJPEG](validation/data/final-image-mjpeg.json)
- [错误 TPU 设备](validation/data/final-invalid-tpu.json)
- [只读普通用户容器的真实硬件验证](validation/data/final-container.json) · [镜像 ID 与构建参数](validation/data/final-container-image.json)

准备依赖和模型后，可使用以下命令重现性能对照：

```bash
PYTHONPATH=. .venv/bin/python scripts/validate_optimizations.py \
  --models models --joint-models models/joint --output /tmp/hd-final-replay.json
```

视频 API 验证命令见项目 README。模型准备与所有检查均不自动部署服务。
