import torch.nn as nn


class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        """Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]"""
        N, C, H, W = x.size()
        g = self.groups
        assert C % g == 0, "Incompatible group size {} for input channel {}".format(
            g, C
        )
        return (
            x.view(N, g, int(C / g), H, W)
                .permute(0, 2, 1, 3, 4)
                .contiguous()
                .view(N, C, H, W)
        )
