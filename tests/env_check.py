import torch, triton
print("torch ", torch.__version__, " hip ", torch.version.hip)
print("triton", triton.__version__)
print("GPUs  ", torch.cuda.device_count(), torch.cuda.get_device_name(0))