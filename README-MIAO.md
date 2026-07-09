```Plain Text
# 启动环境
conda activate recondreamerNew-rl

# 到代码目录
cd ReconSparse_base

# 设置cuda环境
export PYTHONPATH=$(pwd):${PYTHONPATH:-}
export CUDA_HOME=/usr/local/cuda
export CPATH=/usr/local/cuda/include:${CPATH:-}
export LIBRARY_PATH=/usr/local/cuda/lib64:${LIBRARY_PATH:-}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}

# 加载NAVSIM配置脚本
source navsim_env.sh

# 运行DiffsuionDriveV2评估
bash tools/evaluate_fast.sh
bash tools/evaluate_rl.sh
bash tools/evaluate_sel.sh

# 闭环 RL 训练

```
