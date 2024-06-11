# try.py

import torch

print("torch version",torch.__version__)
print(torch.cuda.is_available(), torch.distributed.is_nccl_available())
print("nccl version:",torch.cuda.nccl.version())
print("cuda version:", torch.version.cuda)       
        
cudnn_version = torch.backends.cudnn.version()
print("cuDNN version:", cudnn_version)
print(torch.cuda.device_count(), torch.cuda.get_device_name(0))


