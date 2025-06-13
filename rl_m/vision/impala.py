import torch
import torch.nn as nn

class ResnetBlock(nn.Module):
    def __init__(self, num_ch):
        super().__init__()
        self.num_ch = num_ch

        self.conv1 = nn.Conv2d(self.num_ch, self.num_ch, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(self.num_ch, self.num_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = nn.functional.relu(x)
        out = nn.functional.relu(self.conv1(x))
        out = self.conv2(out)

        return x + out

class ResnetStack(nn.Module):
    def __init__(self, num_in, num_ch, num_blocks, use_max_pooling=True):
        super().__init__()
        self.num_in = num_in
        self.num_ch = num_ch
        self.num_blocks = num_blocks
        self.use_max_pooling = use_max_pooling

        self.conv_in = nn.Conv2d(self.num_in, self.num_ch, kernel_size=3, stride=1, padding=1)

        if self.use_max_pooling:
            self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.blocks = nn.ModuleList([
            ResnetBlock(num_ch=self.num_ch)
            for _ in range(self.num_blocks)
        ])
    
    def forward(self, x):
        x = self.conv_in(x)
        if self.use_max_pooling:
            x = self.max_pool(x)

        for block in self.blocks:
            x = block(x)

        return x


class ImpalaEncoder(nn.Module):
    def __init__(self, width=1, in_channels=3, use_multiplicative_cond=False, stack_sizes=[8, 16, 16], num_blocks=2, dropout_rate=None):
        super().__init__()
        self.width = width
        self.use_multiplicative_cond = use_multiplicative_cond
        self.stack_sizes = stack_sizes
        self.num_blocks = num_blocks
        self.dropout_rate = dropout_rate

        self.stack_blocks = nn.ModuleList([
            ResnetStack(
                num_in=in_channels if idx == 0 else self.stack_sizes[idx - 1] * self.width,
                num_ch= self.stack_sizes[idx] * self.width,
                num_blocks=self.num_blocks,
                use_max_pooling=True
            )
            for idx in range(len(stack_sizes))
        ])
        self.flatten = nn.Flatten()
        self.out = nn.Linear(1024, 256)
        if self.dropout_rate is not None:
            self.dropout = nn.Dropout(p=self.dropout_rate)

    def forward(self, x, train=True, cond_var=None):
        x = x.float() / 255.0

        conv_out = x

        for idx in range(len(self.stack_blocks)):
            conv_out = self.stack_blocks[idx](conv_out)
            if self.dropout_rate is not None:
                conv_out = self.dropout(conv_out)
            if self.use_multiplicative_cond:
                # assert cond_var is not None, "Cond var shouldn't be done when using it"
                temp_out = nn.Linear(conv_out.shape[-1], conv_out.shape[-1])(cond_var)
                x_mult = temp_out.unsqueeze(2).unsqueeze(3)
                # print ('x_mult shape in IMPALA:', x_mult.shape, conv_out.shape)
                conv_out = conv_out * x_mult

        conv_out = nn.functional.relu(conv_out)

        out = self.flatten(conv_out)
        out = self.out(out)
        # return conv_out.reshape(x.shape[0], -1)
        return out
    
import functools as ft
impala_configs = {
    'impala': ImpalaEncoder,
    'impala_small': ft.partial(ImpalaEncoder, num_blocks=1),
    'impala_large': ft.partial(ImpalaEncoder, stack_sizes=(16, 32, 32, 32)),
    'impala_larger': ft.partial(ImpalaEncoder, stack_sizes=(16, 32, 32, 32, 32)),
    'impala_largest': ft.partial(ImpalaEncoder, stack_sizes=(16, 32, 32, 32, 32, 32)),
    'impala_wider': ft.partial(ImpalaEncoder, width=2),
    'impala_widest': ft.partial(ImpalaEncoder, width=4),
    'impala_deeper': ft.partial(ImpalaEncoder, num_blocks=4),
    'impala_deepest': ft.partial(ImpalaEncoder, num_blocks=8),
}