#!/bin/bash

singularity shell --nv $1 <<EOF
cat /etc/issue
python3 <<XEOF
import torch
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
m = torch.Tensor([[1, -3, 3], [3, -5, 3], [6, -6, 4]])
v1 = torch.Tensor([1, 1, 0]).view(3, 1)
m, v1 = m.to(device), v1.to(device)
print(all(m.mm(v1) == -2 * v1))
XEOF
EOF
