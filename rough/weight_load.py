import torch
import sys
from pathlib import Path
import time

sys.path.insert(0, '..')
from models.mobilenetv3 import mobilenetv3_small

model = mobilenetv3_small()
# weights = torch.load(Path('../weights/mobilenetv3-small-55df8e1f.pth'))

# print(weights)
# sys.exit(0)

# model.load_state_dict(weights, strict=False)
inp = torch.randn(1, 1, 1000, 1000)
out = model(inp)
print(out.shape)

# for child1 in model.children():
#     for child2 in child1.children():
#         print(child2)
#         print('-'*22)