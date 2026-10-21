# 实验二：检测模型联合编译

分支 `experiment/cocompile`。只改变编译和缓存分配，不启用跟踪、跳帧或身份特征复用。

## 对照组

1. `baseline`：main 的三个预编译模型。
2. `individual`：用同一编译器分别编译两个检测模型，保留原身份模型。用于隔离编译器版本变化的影响。
3. `co-person-first`：同一次命令编译人体、人脸两个模型，优先为人体模型分配缓存。
4. `co-face-first`：相反顺序编译，优先给人脸模型分配缓存。

后面三组使用相同的原始量化权重；所有组的 MobileFaceNet 文件完全相同。源码只发布已编译身份模型，因此本实验无法验证三模型共同缓存。完整链路仍会切换到独立身份模型，必须实测这种切换是否抵消收益。

## 重现

Google APT 仓库在本次实验中返回 403。采用 Ultralytics 官方导出代码引用的原版编译器便携包，完全在临时目录运行，未安装系统包：

- 地址：`https://github.com/ultralytics/assets/releases/download/v0.0.0/edgetpu-compiler_16.0_amd64.tar.gz`
- SHA-256：`f9a97adbf2ff621d24a91721191264f2abc3c1db31e63d9a266d988f907269ce`
- 版本：`16.0.384591198`
- 来源：[Ultralytics 官方导出实现](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/export/tensorflow.py)

准备项目依赖、原模型与编译器，解包后运行：

```bash
PYTHONPATH=. python -m experiments.compile_models \
  --compiler-bundle /path/to/compiler/usr/bin/edgetpu_compiler_bin

PYTHONPATH=. python -m experiments.switch_cost \
  --variant baseline=models \
  --variant individual=experiments/artifacts/individual \
  --variant co-person-first=experiments/artifacts/co-person-first \
  --variant co-face-first=experiments/artifacts/co-face-first \
  --runs 100 --repeats 3 --output switch-results.json

PYTHONPATH=. python -m experiments.benchmark \
  --variant baseline=models \
  --variant individual=experiments/artifacts/individual \
  --variant co-person-first=experiments/artifacts/co-person-first \
  --variant co-face-first=experiments/artifacts/co-face-first \
  --frames 20 --repeats 3 --output cocompile-results.json

PYTHONPATH=. python -m experiments.compare cocompile-results.json \
  --output cocompile-summary.json
```

固定调用微基准每组合执行 100 次、重复 3 次，分别调用两个检测模型和三个模型，不根据检测结果减少调用。这是用于解释缓存机制的控制实验，不是实际身份识别吞吐。

完整 pipeline 每组合预热 5 帧、测量 20 帧、重复 3 次，覆盖共同协议的全部场景和分辨率。它与本实验自己的基线比较，不与跟踪实验的不同帧序列直接配对。输出差异在逐帧结果中保留；编译器参数缓存日志保留供审计。

选定变体用于服务时，设置 `HD_MODELS_DIR=experiments/artifacts/<变体>`，先运行 `human-detect doctor --invoke`。该分支默认仍使用原版模型，不自动替换生产模型。
