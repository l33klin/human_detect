# 组合实验：跟踪降频 × 联合编译

日期：2026-10-21。分支 `experiment/combined-optimizations` 合并两个实验分支，保持 main 不变，默认关闭时序优化。

假设：身份特征降频减少独立身份模型打断检测共享缓存的次数，因此联合编译可能对时序路径更有价值；但缓存分配缩小与片外参数读取仍可能抵消收益。平均耗时、P95 和结果变化分别判断。

## 四组对照

| 组 | 检测编译方式 | 视频策略 |
| --- | --- | --- |
| baseline | 同版本编译器单独编译 | 每帧完整推理 |
| baseline+temporal | 同版本编译器单独编译 | 跟踪＋身份特征降频 |
| joint | 两检测模型联合编译，人体优先 | 每帧完整推理 |
| joint+temporal | 同一组联合模型 | 跟踪＋身份特征降频 |

使用相同原始量化权重、编译器 16.0.384591198、同一份 MobileFaceNet、阈值和身份库。baseline 表示原版逐帧流程的同版本重编译控制组，避免把编译器版本变化计为组合收益。上一实验中原预编译模型和重编译控制组的样例输出张量一致。

联合编译仍然只覆盖人体与人脸检测模型；没有同一权重的原始身份量化模型，不能称作三模型联合编译。

## 主要实验

- `pn41` 同一块 PCIe TPU，串行运行所有变体，不同时运行其他 TPU 实验。
- 公开素材、场景生成方法与前两组实验相同，记录源图与每组帧序列 SHA-256。
- 尺寸：512×600、1280×720、1920×1080；静止、平移、遮挡、切换/空白、两图多人拼接。
- 每组合预热 5 帧、测量 40 帧、重复 3 轮；四组顺序随机，固定种子；共 7,200 帧。
- 25 FPS 源时间戳、不 sleep、不限制处理 FPS。OpenCV 单线程。
- 记录平均/P95、每阶段 TPU 调用次数及耗时、逐帧输出，并比较组合组相对仅时序组的增量收益。
- 只测 pipeline，不含解码和网络；素材为静态图的几何合成，不代表真实相机的识别准确率和自然运动表现。

额外测试默认 5 FPS 源时间戳：512×600 和 1920×1080 的静止、平移、多人拼接，各 20 帧、3 轮，共 1,440 帧。它有独立基线，帧数不同使运动步长和片段持续时间不同，不与主实验直接配对。

## 复现

先按 [联合编译说明](cocompile.md)准备相同编译器与模型：

```bash
PYTHONPATH=. python -m experiments.compile_models \
  --compiler-bundle /path/to/compiler/usr/bin/edgetpu_compiler_bin

PYTHONPATH=. python -m experiments.benchmark \
  --variant baseline=experiments/artifacts/individual \
  --variant joint=experiments/artifacts/co-person-first --temporal \
  --frames 40 --repeats 3 --input-fps 25 --output combined-25fps.json

PYTHONPATH=. python -m experiments.benchmark \
  --variant baseline=experiments/artifacts/individual \
  --variant joint=experiments/artifacts/co-person-first --temporal \
  --frames 20 --repeats 3 --input-fps 5 --sizes 512x600,1920x1080 \
  --scenarios static,translation,two_people --output combined-5fps.json

PYTHONPATH=. python -m experiments.combined_summary \
  combined-25fps.json combined-5fps.json --output-dir combined-results
```

在服务中同时开启两个方案：

```bash
HD_MODELS_DIR=experiments/artifacts/co-person-first \
HD_TEMPORAL_ENABLED=true human-detect serve
```

创建流的 `fps` 仍决定输出速率上限。该分支不自动修改默认配置，不部署常驻服务。

历史独立实验：[时序结果](results.md)、[联合编译结果](cocompile-results.md)。
