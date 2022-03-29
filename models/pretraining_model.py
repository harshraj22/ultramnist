import torch
import torch.nn as nn

# import sys
# sys.path.append('..')
from .mobilenetv3 import mobilenetv3_small, h_swish


class PreTrainingModel(nn.Module):
    def __init__(self):
        super(PreTrainingModel, self).__init__()

        self.model = mobilenetv3_small(num_classes=576)
        # self.model.classifier = nn.Sequential()
        self.flip_label_head = nn.Sequential(
            nn.Linear(576, 200),
            h_swish(),
            nn.Dropout(0.2),
            nn.Linear(200, 2),
        )

        self.rotate_label_head = nn.Sequential(
            nn.Linear(576, 200),
            h_swish(),
            nn.Dropout(0.2),
            nn.Linear(200, 3),
        )

    def forward(self, x):
        features = self.model(x)
        filp_pred = self.flip_label_head(features)
        rotate_pred = self.rotate_label_head(features)
        return filp_pred, rotate_pred


if __name__ == '__main__':
    model = PreTrainingModel()
    x = torch.randn(4, 1, 500, 500)
    out = model(x)
    print(out[0].shape, out[1].shape)