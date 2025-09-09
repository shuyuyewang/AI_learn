import torch

# 检查是否支持 CUDA
print("Is CUDA available:", torch.cuda.is_available())

# 检查 CUDA 版本
print("PyTorch CUDA version:", torch.version.cuda)

# 检查 cuDNN 可用性和版本
print("Is cuDNN available:", torch.backends.cudnn.is_available())
print("cuDNN version:", torch.backends.cudnn.version())

# 如果 CUDA 可用，输出 GPU 相关信息
if torch.cuda.is_available():
   # GPU 数量
    print("GPU device count:", torch.cuda.device_count())
    # 当前 GPU 设备
    print("Current GPU device:", torch.cuda.current_device())
    # 使用的 GPU 设备名称
    print("Current GPU device name:", torch.cuda.get_device_name(torch.cuda.current_device()))