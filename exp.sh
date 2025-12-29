#!/bin/bash

# ================= 1. 实验参数配置 (在这里修改) =================
# 你可以在这里随意修改环境名，比如改为 Ant-v3, Humanoid-v4 等
ENV_NAME="dog-run"
SEED=22345

# ================= 2. 路径配置 (通常不需要动) =================
PROJECT_DIR="/home/yuzhouyang/code/doublegum"
CONDA_ENV_DIR="/home/yuzhouyang/miniconda3/envs/doublegum"
MUJOCO_ROOT="$PROJECT_DIR/.mujoco/mujoco210"

# ================= 3. 环境变量自动注入 =================

# 设置 MuJoCo 路径
export MUJOCO_PATH="$MUJOCO_ROOT"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$MUJOCO_ROOT/bin"

# 设置 NVIDIA/JAX 库路径 (修复 libcublasLt 报错)
NVIDIA_LIB="$CONDA_ENV_DIR/lib/python3.9/site-packages/nvidia"

export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$NVIDIA_LIB/cublas/lib"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$NVIDIA_LIB/cudnn/lib"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$NVIDIA_LIB/cufft/lib"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$NVIDIA_LIB/cusolver/lib"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$NVIDIA_LIB/cusparse/lib"

# ================= 4. 执行命令 =================

echo "#########################################################"
echo "   Configuration Loaded"
echo "   Target Env : ${ENV_NAME}"
echo "   Target Seed: ${SEED}"
echo "   Status     : Launching..."
echo "#########################################################"

# 使用变量 $ENV_NAME 和 $SEED 动态填充参数
"$CONDA_ENV_DIR/bin/python" "$PROJECT_DIR/main_cont.py" \
    --env "$ENV_NAME" \
    --seed "$SEED"