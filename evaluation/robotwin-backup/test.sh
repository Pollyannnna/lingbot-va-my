export NVIDIA_DRIVER_CAPABILITIES=all

#精准拦截图形库，解决 SAPIEN 渲染崩溃
export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libEGL.so.1 /usr/lib/x86_64-linux-gnu/libGL.so.1"

apt-get update -qq
apt-get install -y libgl1-mesa-glx libglib2.0-0 librdmacm-dev libibverbs-dev rdma-core infiniband-diags
apt-get install -y libvulkan1 mesa-vulkan-drivers vulkan-utils ffmpeg

# 3. 渲染与显存环境配置
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

export DISPLAY=""
export PYOPENGL_PLATFORM=egl
export MUJOCO_GL=egl
export EGL_PLATFORM=surfaceless

# 4. 修复 Python 包版本冲突 (pkg_resources 和 pillow)
python -m pip install "setuptools<81.0.0" "pillow<12.0.0"

cd /data/250010187/yeziyang1/lingbot-va

# 5. 动态修改子脚本显存占用
sed -i 's/XLA_PYTHON_CLIENT_MEM_FRACTION=0.9/XLA_PYTHON_CLIENT_MEM_FRACTION=0.5/g' evaluation/robotwin/launch_client_with_logging.sh

# 6. 正式启动评估
task_name="rotate_qrcode";
save_root="results/rotate_qrcode-0303-withvideo&log";
bash evaluation/robotwin/launch_client_with_logging.sh ${save_root} ${task_name}
