# 实验一：跟踪与身份特征降频

分支 `experiment/temporal-tracking`，详细对照方法见 [共同协议](protocol.md)，实测见 [结果](results.md)。

## 策略

- 视频每 5 个已分析帧，或距离上次检测超过 0.5 秒时，执行人体/人脸完整检测。
- 中间帧在不超过 640×480 的灰度图上做稀疏光流和前后向一致性检查，估计每个人体的仿射位移。它是 CPU 图像处理，神经网络仍仅在 TPU 执行。
- 身份特征最多复用 1 秒；人脸框重叠且小尺寸外观相似时才允许命中缓存。每次完整检测都会重新查询当前身份库。
- 场景突变、光流失败、人脸外观变化、时间戳倒退/中断、分辨率或身份库变化时刷新。没有人体时每帧检测，避免空画面无限复用。
- 每路视频独立会话；静态图片接口始终完整推理；TPU 已被标记故障时，跟踪路径也拒绝继续输出。

返回结果增加 `temporal`：

```json
{
  "mode": "tracked",
  "detection_age_ms": 120,
  "identity_age_ms": 320,
  "identity_source": "cached_embedding"
}
```

`mode=tracked` 的框和置信度由历史检测传播，不能当成当前帧重新检测的置信度。`identity_age_ms` 是会话缓存中最老特征的年龄上界，可能比当前输出人脸的实际特征年龄大；它不是每张脸单独的年龄。`identity_source` 在一个结果中有任意复用时标记缓存来源。当前是保守的实验性元数据。

## 运行

默认关闭，先按 README 配置 TPU 和模型。启用视频实验：

```bash
HD_TEMPORAL_ENABLED=true human-detect serve
```

可配置 `HD_TEMPORAL_DETECT_EVERY=5`、`HD_TEMPORAL_DETECTION_SECONDS=0.5`、`HD_TEMPORAL_IDENTITY_SECONDS=1`。需要输出更高帧率时，创建流的 `fps` 也需相应提高；降低单帧计算不自动改变 API 请求的速率上限。

复现 25 FPS 源时间戳的不限速处理测试：

```bash
PYTHONPATH=. python -m experiments.benchmark --variant baseline=models --temporal \
  --frames 40 --repeats 3 --input-fps 25 --output temporal-results.json
PYTHONPATH=. python -m experiments.compare temporal-results.json \
  --output temporal-summary.json
```

实际默认流分析速率为 5 FPS；此时 0.5 秒刷新约束会更早触发，不能直接套用 25 FPS 测试中的 TPU 调用降幅。

## 适用限制

合成回放主要检验机制、速度和相对基线的输出变化。它没有验证真实人物交叉、非刚性动作、相似外观换人、长期漂移等困难场景。新人物可能要等到下一次检测才能被发现，不能宣称无延迟发现所有新人。场景和外观检查降低错误复用的概率，但不构成身份正确性的保证。

不同步变化的人脸与身体会使光流传播的脸框偏移。真实摄像头验证和阈值校准完成前，保持此开关关闭，不将该实验分支自动合并到 main。
