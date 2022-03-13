import torch
import sys
from pathlib import Path

sys.path.insert(0, '..')
from models.mobilenetv3 import mobilenetv3_small

model = mobilenetv3_small()
weights = torch.load(Path('../weights/mobilenetv3-small-55df8e1f.pth'))

print(weights)
sys.exit(0)

model.load_state_dict(weights, strict=False)