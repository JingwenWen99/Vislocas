import torch
import torch.nn as nn
import torch.nn.functional as F

# from utils.config_defaults import get_cfg


# cfg = get_cfg()
# # Set random seed from configs.
# torch.manual_seed(cfg.RNG_SEED)
# torch.cuda.manual_seed(cfg.RNG_SEED)
# torch.cuda.manual_seed_all(cfg.RNG_SEED)


class Tokenizer(nn.Module):
    def __init__(self,
                 kernel_size, stride, padding,
                 pooling_kernel_size=3, pooling_stride=2, pooling_padding=1,
                 n_conv_layers=1,
                 n_input_channels=3,
                 n_output_channels=64,
                 in_planes=[64, 64],
                 norm=False,
                 activation=None,
                 max_pool=[True],
                 downsample=1,
                 conv_bias=False):
        super(Tokenizer, self).__init__()

        n_filter_list = [n_input_channels] + \
                        [in_planes[i] for i in range(n_conv_layers - 1)] + \
                        [n_output_channels]

        self.conv_layers = nn.Sequential(
            nn.MaxPool2d(kernel_size=downsample, stride=downsample, padding=0) if downsample!=1 else nn.Identity(),
            *[nn.Sequential(
                nn.Conv2d(n_filter_list[i], n_filter_list[i + 1],
                          kernel_size=(kernel_size[i], kernel_size[i]),
                          stride=(stride[i], stride[i]),
                          padding=(padding[i], padding[i]), bias=conv_bias),
                nn.BatchNorm2d(n_filter_list[i + 1]) if norm else nn.Identity(),
                # nn.LayerNorm(n_filter_list[i + 1]) if norm else nn.Identity(),
                nn.Identity() if activation is None else activation(),
                nn.MaxPool2d(kernel_size=pooling_kernel_size,
                             stride=pooling_stride,
                             padding=pooling_padding) if max_pool[i] else nn.Identity()
            )
                for i in range(n_conv_layers)
            ],
            # nn.BatchNorm2d(n_output_channels) if norm else nn.Identity(),
            )


        # self.conv_layers = nn.Sequential(
        #     nn.MaxPool2d(kernel_size=downsample, stride=downsample, padding=0) if downsample!=1 else nn.Identity(),
        #     *[nn.Sequential(
        #         nn.Conv2d(n_filter_list[i], n_filter_list[i + 1],
        #                   kernel_size=1, stride=1, padding=0, bias=conv_bias),
        #         nn.Conv2d(n_filter_list[i + 1], n_filter_list[i + 1],
        #                   kernel_size=(1, kernel_size[i]),
        #                   stride=(1, stride[i]),
        #                   padding=(0, padding[i]), bias=conv_bias),
        #         nn.Conv2d(n_filter_list[i + 1], n_filter_list[i + 1],
        #                   kernel_size=(kernel_size[i], 1),
        #                   stride=(stride[i], 1),
        #                   padding=(padding[i], 0), bias=conv_bias),
        #         nn.BatchNorm2d(n_filter_list[i + 1]) if norm else nn.Identity(),
        #         # nn.LayerNorm(n_filter_list[i + 1]) if norm else nn.Identity(),
        #         nn.Identity() if activation is None else activation(),
        #         nn.MaxPool2d(kernel_size=pooling_kernel_size,
        #                      stride=pooling_stride,
        #                      padding=pooling_padding) if max_pool[i] else nn.Identity()
        #     )
        #         for i in range(n_conv_layers)
        #     ],
        #     # nn.BatchNorm2d(n_output_channels) if norm else nn.Identity(),
        #     )

        # self.conv_layers = nn.Sequential(
        #     nn.MaxPool2d(kernel_size=downsample, stride=downsample, padding=0) if downsample!=1 else nn.Identity(),
        #     *[nn.Sequential(
        #         nn.Conv2d(n_filter_list[i], n_filter_list[i + 1],
        #                   kernel_size=(kernel_size[i], kernel_size[i]),
        #                   stride=(stride[i], stride[i]),
        #                   padding=(padding[i], padding[i]), bias=conv_bias),
        #         nn.BatchNorm2d(n_filter_list[i + 1]) if norm else nn.Identity(),
        #         # nn.LayerNorm(n_filter_list[i + 1]) if norm else nn.Identity(),
        #         nn.Identity() if activation is None else activation(),
        #         # nn.MaxPool2d(kernel_size=pooling_kernel_size,
        #         #              stride=pooling_stride,
        #         #              padding=pooling_padding) if max_pool else nn.Identity()
        #     )
        #         for i in range(n_conv_layers)
        #     ],
        #     nn.MaxPool2d(kernel_size=downsample, stride=downsample, padding=0) if downsample!=1 else nn.Identity(),
        #     # nn.BatchNorm2d(n_output_channels) if norm else nn.Identity(),
        #     )

        self.flattener = nn.Flatten(2, 3)
        self.apply(self.init_weight)

    def sequence_length(self, n_channels=3, height=224, width=224):
        return self.forward(torch.zeros((1, n_channels, height, width))).shape[1]

    def forward(self, x):
        x = self.conv_layers(x)
        return self.flattener(x).transpose(-2, -1)

    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class TextTokenizer(nn.Module):
    def __init__(self,
                 kernel_size, stride, padding,
                 pooling_kernel_size=3, pooling_stride=2, pooling_padding=1,
                 embedding_dim=300,
                 n_output_channels=128,
                 activation=None,
                 max_pool=True,
                 *args, **kwargs):
        super(TextTokenizer, self).__init__()

        self.max_pool = max_pool
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, n_output_channels,
                      kernel_size=(kernel_size, embedding_dim),
                      stride=(stride, 1),
                      padding=(padding, 0), bias=False),
            nn.Identity() if activation is None else activation(),
            nn.MaxPool2d(
                kernel_size=(pooling_kernel_size, 1),
                stride=(pooling_stride, 1),
                padding=(pooling_padding, 0)
            ) if max_pool else nn.Identity()
        )

        self.apply(self.init_weight)

    def seq_len(self, seq_len=32, embed_dim=300):
        return self.forward(torch.zeros((1, seq_len, embed_dim)))[0].shape[1]

    def forward_mask(self, mask):
        new_mask = mask.unsqueeze(1).float()
        cnn_weight = torch.ones(
            (1, 1, self.conv_layers[0].kernel_size[0]),
            device=mask.device,
            dtype=torch.float)
        new_mask = F.conv1d(
            new_mask, cnn_weight, None,
            self.conv_layers[0].stride[0], self.conv_layers[0].padding[0], 1, 1)
        if self.max_pool:
            new_mask = F.max_pool1d(
                new_mask, self.conv_layers[2].kernel_size[0],
                self.conv_layers[2].stride[0], self.conv_layers[2].padding[0], 1, False, False)
        new_mask = new_mask.squeeze(1)
        new_mask = (new_mask > 0)
        return new_mask

    def forward(self, x, mask=None):
        x = x.unsqueeze(1)
        x = self.conv_layers(x)
        x = x.transpose(1, 3).squeeze(1)
        if mask is not None:
            mask = self.forward_mask(mask).unsqueeze(-1).float()
            x = x * mask
        return x, mask

    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)
