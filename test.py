import torch
import librosa
import scikit-learn

if torch.cuda.is_available():
    print("CUDA is available. GPU support is enabled.")
    print("CUDA Version:", torch.version.cuda)
    print(torch.__version__)
    # Example to demonstrate using the GPU
    device = torch.device("cuda")  # Use CUDA if available
    # 获取并打印当前CUDA设备的名称
    cuda_device_name = torch.cuda.get_device_name(torch.cuda.current_device())
    print("CUDA Device Name:", cuda_device_name)
    x = torch.rand(5, 5).to(device)
    print("A random tensor:", x)
else:
    print("CUDA is not available. Check your installation.")
