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

# 闭环 RL 训练（后台）
screen -S train_rl
LOG_DIR=./logs CONFIG=script/configs/ppo_closed_loop.yaml bash tools/train_actor_learner.sh
退出：Ctrl + A 然后按 D
查看当前有哪些 screen：：screen -ls
重新进入这个训练窗口：screen -r train_rl

# 评估 （开源代码无相关部分）
insight：main分支中的想法，在ReconDreamer上训练，在HUGSIM上进行评估

