from torch.hub import load_state_dict_from_url
import torch
import torch.nn as nn
from .cvt_utils.transformers import TransformerClassifier
from .cvt_utils.tokenizer import Tokenizer
from .cvt_utils.helpers import pe_check, fc_check

from timm.models.registry import register_model

from utils.config_defaults import get_cfg


cfg = get_cfg()
# Set random seed from configs.
torch.manual_seed(cfg.RNG_SEED)
torch.cuda.manual_seed(cfg.RNG_SEED)
torch.cuda.manual_seed_all(cfg.RNG_SEED)


model_urls = {
    'cct_7_3x1_32':
        'https://shi-labs.com/projects/cct/checkpoints/pretrained/cct_7_3x1_32_cifar10_300epochs.pth',
    'cct_7_3x1_32_sine':
        'https://shi-labs.com/projects/cct/checkpoints/pretrained/cct_7_3x1_32_sine_cifar10_5000epochs.pth',
    'cct_7_3x1_32_c100':
        'https://shi-labs.com/projects/cct/checkpoints/pretrained/cct_7_3x1_32_cifar100_300epochs.pth',
    'cct_7_3x1_32_sine_c100':
        'https://shi-labs.com/projects/cct/checkpoints/pretrained/cct_7_3x1_32_sine_cifar100_5000epochs.pth',
    'cct_7_7x2_224_sine':
        'https://shi-labs.com/projects/cct/checkpoints/pretrained/cct_7_7x2_224_flowers102.pth',
    'cct_14_7x2_224':
        'https://shi-labs.com/projects/cct/checkpoints/pretrained/cct_14_7x2_224_imagenet.pth',
    'cct_14_7x2_384':
        'https://shi-labs.com/projects/cct/checkpoints/finetuned/cct_14_7x2_384_imagenet.pth',
    'cct_14_7x2_384_fl':
        'https://shi-labs.com/projects/cct/checkpoints/finetuned/cct_14_7x2_384_flowers102.pth',
}


class CCT(nn.Module):
    def __init__(self,
                 img_size=224,
                 embedding_dim=768,
                 n_input_channels=3,
                 n_conv_layers=1,
                 token_in_planes=[64, 64],
                 downsample=1,
                 kernel_size=[7, 7],
                 stride=[2, 2],
                 padding=[3, 3],
                 pooling_kernel_size=3,
                 pooling_stride=2,
                 pooling_padding=1,
                 norm=False,
                 max_pool=[True],
                 activation=nn.ReLU,
                 dropout=0.,
                 attention_dropout=0.1,
                 stochastic_depth=0.1,
                 num_layers=14,
                 num_heads=6,
                 mlp_ratio=4.0,
                 head_ratio=8.0,
                 num_classes=1000,
                 positional_embedding='learnable',
                 *args, **kwargs):
        super(CCT, self).__init__()

        self.tokenizer = Tokenizer(n_input_channels=n_input_channels,
                                   n_output_channels=embedding_dim,
                                   in_planes=token_in_planes,
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   padding=padding,
                                   pooling_kernel_size=pooling_kernel_size,
                                   pooling_stride=pooling_stride,
                                   pooling_padding=pooling_padding,
                                   norm=norm,
                                   max_pool=max_pool,
                                   activation=activation,
                                   n_conv_layers=n_conv_layers,
                                   downsample=downsample,
                                   conv_bias=False)

        self.classifier = TransformerClassifier(
            sequence_length=self.tokenizer.sequence_length(n_channels=n_input_channels,
                                                           height=img_size,
                                                           width=img_size),
            embedding_dim=embedding_dim,
            seq_pool=True,
            dropout=dropout,
            attention_dropout=attention_dropout,
            stochastic_depth=stochastic_depth,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            head_ratio=head_ratio,
            num_classes=num_classes,
            positional_embedding=positional_embedding
        )


    def forward(self, x):
        x = self.tokenizer(x)
        return self.classifier(x)


def _cct(arch, pretrained, progress,
         num_layers, num_heads, mlp_ratio, embedding_dim,
         kernel_size=[3], stride=None, padding=None,
         positional_embedding='learnable',
         *args, **kwargs):
    stride = stride if stride is not None else [max(1, (k // 2) - 1) for k in kernel_size]
    padding = padding if padding is not None else [max(1, (k // 2)) for k in kernel_size]
    model = CCT(num_layers=num_layers,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                embedding_dim=embedding_dim,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                *args, **kwargs)

    if pretrained:
        if arch in model_urls:
            state_dict = load_state_dict_from_url(model_urls[arch],
                                                  progress=progress)
            if positional_embedding == 'learnable':
                state_dict = pe_check(model, state_dict)
            elif positional_embedding == 'sine':
                state_dict['classifier.positional_emb'] = model.state_dict()['classifier.positional_emb']
            state_dict = fc_check(model, state_dict)
            model.load_state_dict(state_dict)
        else:
            raise RuntimeError(f'Variant {arch} does not yet have pretrained weights.')
    return model


def cct_2(arch, pretrained, progress, *args, **kwargs):
    return _cct(arch, pretrained, progress, num_layers=2, num_heads=2, mlp_ratio=1, embedding_dim=128,
                *args, **kwargs)


def cct_4(arch, pretrained, progress, *args, **kwargs):
    return _cct(arch, pretrained, progress, num_layers=4, num_heads=2, mlp_ratio=1, embedding_dim=128,
                *args, **kwargs)


def cct_6(arch, pretrained, progress, *args, **kwargs):
    return _cct(arch, pretrained, progress, num_layers=6, num_heads=4, mlp_ratio=2, embedding_dim=256,
                *args, **kwargs)


def cct_7(arch, pretrained, progress, *args, **kwargs):
    return _cct(arch, pretrained, progress, num_layers=7, num_heads=4, mlp_ratio=2, embedding_dim=256,
                *args, **kwargs)


def cct_14(arch, pretrained, progress, *args, **kwargs):
    return _cct(arch, pretrained, progress, num_layers=14, num_heads=6, mlp_ratio=3, embedding_dim=384,
                *args, **kwargs)


@register_model
def cct_2_3x2_32(pretrained=False, progress=False,
                 img_size=32, positional_embedding='learnable', num_classes=10,
                 *args, **kwargs):
    return cct_2('cct_2_3x2_32', pretrained, progress,
                 kernel_size=[3, 3], n_conv_layers=2,
                 img_size=img_size, positional_embedding=positional_embedding,
                 num_classes=num_classes,
                 *args, **kwargs)


@register_model
def cct_2_3x2_32_sine(pretrained=False, progress=False,
                      img_size=32, positional_embedding='sine', num_classes=10,
                      *args, **kwargs):
    return cct_2('cct_2_3x2_32_sine', pretrained, progress,
                 kernel_size=[3, 3], n_conv_layers=2,
                 img_size=img_size, positional_embedding=positional_embedding,
                 num_classes=num_classes,
                 *args, **kwargs)


@register_model
def cct_4_3x2_32(pretrained=False, progress=False,
                 img_size=32, positional_embedding='learnable', num_classes=10,
                 *args, **kwargs):
    return cct_4('cct_4_3x2_32', pretrained, progress,
                 kernel_size=[3, 3], n_conv_layers=2,
                 img_size=img_size, positional_embedding=positional_embedding,
                 num_classes=num_classes,
                 *args, **kwargs)


@register_model
def cct_4_3x2_32_sine(pretrained=False, progress=False,
                      img_size=32, positional_embedding='sine', num_classes=10,
                      *args, **kwargs):
    return cct_4('cct_4_3x2_32_sine', pretrained, progress,
                 kernel_size=[3, 3], n_conv_layers=2,
                 img_size=img_size, positional_embedding=positional_embedding,
                 num_classes=num_classes,
                 *args, **kwargs)


@register_model
def cct_6_3x1_32(pretrained=False, progress=False,
                 img_size=32, positional_embedding='learnable', num_classes=10,
                 *args, **kwargs):
    return cct_6('cct_6_3x1_32', pretrained, progress,
                 kernel_size=[3], n_conv_layers=1,
                 img_size=img_size, positional_embedding=positional_embedding,
                 num_classes=num_classes,
                 *args, **kwargs)


@register_model
def cct_6_3x1_32_sine(pretrained=False, progress=False,
                      img_size=32, positional_embedding='sine', num_classes=10,
                      *args, **kwargs):
    return cct_6('cct_6_3x1_32_sine', pretrained, progress,
                 kernel_size=[3], n_conv_layers=1,
                 img_size=img_size, positional_embedding=positional_embedding,
                 num_classes=num_classes,
                 *args, **kwargs)


@register_model
def cct_6_3x2_32(pretrained=False, progress=False,
                 img_size=32, positional_embedding='learnable', num_classes=10,
                 *args, **kwargs):
    return cct_6('cct_6_3x2_32', pretrained, progress,
                 kernel_size=[3, 3], n_conv_layers=2,
                 img_size=img_size, positional_embedding=positional_embedding,
                 num_classes=num_classes,
                 *args, **kwargs)


@register_model
def cct_6_3x2_32_sine(pretrained=False, progress=False,
                      img_size=32, positional_embedding='sine', num_classes=10,
                      *args, **kwargs):
    return cct_6('cct_6_3x2_32_sine', pretrained, progress,
                 kernel_size=[3, 3], n_conv_layers=2,
                 img_size=img_size, positional_embedding=positional_embedding,
                 num_classes=num_classes,
                 *args, **kwargs)


@register_model
def cct_7_3x1_32(pretrained=False, progress=False,
                 img_size=32, positional_embedding='learnable', num_classes=10,
                 *args, **kwargs):
    return cct_7('cct_7_3x1_32', pretrained, progress,
                 kernel_size=[3], n_conv_layers=1,
                 img_size=img_size, positional_embedding=positional_embedding,
                 num_classes=num_classes,
                 *args, **kwargs)


@register_model
def cct_7_3x1_32_sine(pretrained=False, progress=False,
                      img_size=32, positional_embedding='sine', num_classes=10,
                      *args, **kwargs):
    return cct_7('cct_7_3x1_32_sine', pretrained, progress,
                 kernel_size=[3], n_conv_layers=1,
                 img_size=img_size, positional_embedding=positional_embedding,
                 num_classes=num_classes,
                 *args, **kwargs)


@register_model
def cct_7_3x1_32_c100(pretrained=False, progress=False,
                      img_size=32, positional_embedding='learnable', num_classes=100,
                      *args, **kwargs):
    return cct_7('cct_7_3x1_32_c100', pretrained, progress,
                 kernel_size=[3], n_conv_layers=1,
                 img_size=img_size, positional_embedding=positional_embedding,
                 num_classes=num_classes,
                 *args, **kwargs)


@register_model
def cct_7_3x1_32_sine_c100(pretrained=False, progress=False,
                           img_size=32, positional_embedding='sine', num_classes=100,
                           *args, **kwargs):
    return cct_7('cct_7_3x1_32_sine_c100', pretrained, progress,
                 kernel_size=[3], n_conv_layers=1,
                 img_size=img_size, positional_embedding=positional_embedding,
                 num_classes=num_classes,
                 *args, **kwargs)


@register_model
def cct_7_3x2_32(pretrained=False, progress=False,
                 img_size=32, positional_embedding='learnable', num_classes=10,
                 *args, **kwargs):
    return cct_7('cct_7_3x2_32', pretrained, progress,
                 kernel_size=[3, 3], n_conv_layers=2,
                 img_size=img_size, positional_embedding=positional_embedding,
                 num_classes=num_classes,
                 *args, **kwargs)


@register_model
def cct_7_3x2_32_sine(pretrained=False, progress=False,
                      img_size=32, positional_embedding='sine', num_classes=10,
                      *args, **kwargs):
    return cct_7('cct_7_3x2_32_sine', pretrained, progress,
                 kernel_size=[3, 3], n_conv_layers=2,
                 img_size=img_size, positional_embedding=positional_embedding,
                 num_classes=num_classes,
                 *args, **kwargs)


@register_model
def cct_7_7x2_224(pretrained=False, progress=False,
                  img_size=224, positional_embedding='learnable', num_classes=102,
                  *args, **kwargs):
    return cct_7('cct_7_7x2_224', pretrained, progress,
                 kernel_size=[7, 7], n_conv_layers=2,
                 img_size=img_size, positional_embedding=positional_embedding,
                 num_classes=num_classes,
                 *args, **kwargs)


@register_model
def cct_7_7x2_224_sine(pretrained=False, progress=False,
                       img_size=224, positional_embedding='sine', num_classes=102,
                       *args, **kwargs):
    return cct_7('cct_7_7x2_224_sine', pretrained, progress,
                 kernel_size=[7, 7], n_conv_layers=2,
                 img_size=img_size, positional_embedding=positional_embedding,
                 num_classes=num_classes,
                 *args, **kwargs)


@register_model
def cct_14_7x2_224(pretrained=False, progress=False,
                   img_size=224, positional_embedding='learnable', num_classes=1000,
                   *args, **kwargs):
    return cct_14('cct_14_7x2_224', pretrained, progress,
                  kernel_size=[7, 7], n_conv_layers=2,
                  img_size=img_size, positional_embedding=positional_embedding,
                  num_classes=num_classes,
                  *args, **kwargs)


@register_model
def cct_14_7x2_384(pretrained=False, progress=False,
                   img_size=384, positional_embedding='learnable', num_classes=1000,
                   *args, **kwargs):
    return cct_14('cct_14_7x2_384', pretrained, progress,
                  kernel_size=[7, 7], n_conv_layers=2,
                  img_size=img_size, positional_embedding=positional_embedding,
                  num_classes=num_classes,
                  *args, **kwargs)


@register_model
def cct_14_7x2_384_fl(pretrained=False, progress=False,
                      img_size=384, positional_embedding='learnable', num_classes=102,
                      *args, **kwargs):
    return cct_14('cct_14_7x2_384_fl', pretrained, progress,
                  kernel_size=[7, 7], n_conv_layers=2,
                  img_size=img_size, positional_embedding=positional_embedding,
                  num_classes=num_classes,
                  *args, **kwargs)


@register_model
def cct_modified(**kwargs):
    cct_model = _cct(None, False, None, num_layers=14, num_heads=8, mlp_ratio=3, embedding_dim=512,
                kernel_size=[5, 5], stride=[2, 2], padding=[2, 2], n_conv_layers=2, positional_embedding='learnable',
                **kwargs)

    return cct_model


@register_model
def cct_modified_(**kwargs):
    cct_model = _cct(None, False, None, num_layers=8, num_heads=16, mlp_ratio=3, embedding_dim=512,
                kernel_size=[5, 5], stride=[2, 2], padding=[2, 2], n_conv_layers=2, positional_embedding='learnable',
                **kwargs)

    return cct_model


@register_model
def cct_modified2(**kwargs):
    cct_model = _cct(None, False, None, num_layers=2, num_heads=2, mlp_ratio=1, n_conv_layers=3, embedding_dim=512, token_in_planes=[192, 256, 384], downsample=2,
                kernel_size=[7, 7, 7], stride=[3, 2, 2], padding=[2, 3, 3], positional_embedding='learnable',
                **kwargs)

    return cct_model


@register_model
def cct_modified3(**kwargs):
    cct_model = _cct(None, False, None, num_layers=6, num_heads=4, mlp_ratio=2, n_conv_layers=3, embedding_dim=512, token_in_planes=[192, 256, 384], downsample=2,
                kernel_size=[7, 7, 7], stride=[3, 2, 2], padding=[2, 3, 3], positional_embedding='learnable',
                **kwargs)

    return cct_model


@register_model
def cct_modified4(**kwargs):
    cct_model = _cct(None, False, None, num_layers=14, num_heads=8, mlp_ratio=3, n_conv_layers=3, embedding_dim=512, token_in_planes=[192, 256, 384], downsample=2,
                kernel_size=[7, 7, 7], stride=[3, 2, 2], padding=[2, 3, 3], positional_embedding='learnable',
                **kwargs)

    return cct_model


@register_model
def cct_modified5(**kwargs):
    cct_model = _cct(None, False, None, num_layers=17, num_heads=16, mlp_ratio=4, n_conv_layers=3, embedding_dim=512, token_in_planes=[192, 256, 384], downsample=2,
                kernel_size=[7, 7, 7], stride=[3, 2, 2], padding=[2, 3, 3], positional_embedding='learnable',
                **kwargs)

    return cct_model

@register_model
def cct_modified6(**kwargs):
    cct_model = _cct(None, False, None, num_layers=14, num_heads=8, mlp_ratio=3, n_conv_layers=3, embedding_dim=512, token_in_planes=[96, 192, 384], downsample=2,
                kernel_size=[7, 7, 7], stride=[2, 2, 2], padding=[3, 3, 3], positional_embedding='learnable',
                **kwargs)

    return cct_model


@register_model
def cct_modified7(**kwargs):
    cct_model = _cct(None, False, None, num_layers=14, num_heads=8, mlp_ratio=3, n_conv_layers=2, embedding_dim=384, token_in_planes=[96, 256], downsample=2,
                kernel_size=[7, 7], stride=[4, 3], padding=[2, 2], positional_embedding='learnable',
                **kwargs)

    return cct_model


@register_model
def cct_modified8(**kwargs):
    cct_model = _cct(None, False, None, num_layers=8, num_heads=32, mlp_ratio=4, n_conv_layers=3, embedding_dim=512, token_in_planes=[96, 192, 384], downsample=2,
                kernel_size=[7, 7, 7], stride=[2, 2, 2], padding=[3, 3, 3], positional_embedding='learnable',
                **kwargs)

    return cct_model

@register_model
def cct_modified9(**kwargs):
    cct_model = _cct(None, False, None, num_layers=6, num_heads=16, mlp_ratio=3, n_conv_layers=3, embedding_dim=512, token_in_planes=[96, 192, 384], downsample=2,
                kernel_size=[7, 7, 7], stride=[2, 2, 2], padding=[3, 3, 3], positional_embedding='learnable',
                **kwargs)

    return cct_model


@register_model
def cct_modified10(**kwargs):
    cct_model = _cct(None, False, None, num_layers=3, num_heads=16, mlp_ratio=3, n_conv_layers=3, embedding_dim=512, token_in_planes=[96, 192, 384], downsample=2,
                kernel_size=[7, 7, 7], stride=[2, 2, 2], padding=[3, 3, 3], positional_embedding='learnable',
                **kwargs)

    return cct_model


@register_model
def cct_modified11(**kwargs):
    cct_model = _cct(None, False, None, num_layers=8, num_heads=32, mlp_ratio=4, n_conv_layers=3, embedding_dim=256, token_in_planes=[96, 128, 192], downsample=2,
                kernel_size=[7, 7, 7], stride=[2, 2, 2], padding=[3, 3, 3], positional_embedding='learnable',
                **kwargs)

    return cct_model

@register_model
def cct_modified12(**kwargs):
    cct_model = _cct(None, False, None, num_layers=14, num_heads=16, mlp_ratio=4, n_conv_layers=3, embedding_dim=512, token_in_planes=[128, 192, 384], downsample=2,
                kernel_size=[7, 7, 7], stride=[2, 2, 2], padding=[3, 3, 3], positional_embedding='learnable',
                **kwargs)

    return cct_model


@register_model
def cct_modified13(**kwargs):
    cct_model = _cct(None, False, None, num_layers=12, num_heads=16, mlp_ratio=4, n_conv_layers=3, embedding_dim=512, token_in_planes=[96, 192, 384], downsample=2,
                kernel_size=[7, 7, 7], stride=[2, 2, 2], padding=[3, 3, 3], positional_embedding='learnable',
                **kwargs)

    return cct_model


@register_model
def cct_modified14(**kwargs):
    cct_model = _cct(None, False, None, num_layers=17, num_heads=16, mlp_ratio=4, n_conv_layers=3, embedding_dim=512, token_in_planes=[96, 192, 384], downsample=2,
                kernel_size=[7, 7, 7], stride=[2, 2, 2], padding=[3, 3, 3], positional_embedding='learnable',
                **kwargs)

    return cct_model


@register_model
def cct_modified15(**kwargs):
    cct_model = _cct(None, False, None, num_layers=17, num_heads=16, mlp_ratio=4, n_conv_layers=3, embedding_dim=384, token_in_planes=[192, 256, 384], downsample=2,
                kernel_size=[7, 7, 7], stride=[3, 2, 2], padding=[2, 3, 3], positional_embedding='learnable',
                **kwargs)

    return cct_model


@register_model
def cct_modified16(**kwargs):
    cct_model = _cct(None, False, None, num_layers=6, num_heads=16, mlp_ratio=4, n_conv_layers=3, embedding_dim=384, token_in_planes=[192, 256, 384], downsample=2,
                kernel_size=[7, 7, 7], stride=[3, 2, 2], padding=[2, 3, 3], positional_embedding='learnable',
                **kwargs)

    return cct_model


@register_model
def cct_modified17(**kwargs):
    cct_model = _cct(None, False, None, num_layers=14, num_heads=16, mlp_ratio=4, n_conv_layers=3, embedding_dim=480, token_in_planes=[160, 256, 384],
    # cct_model = _cct(None, False, None, num_layers=14, num_heads=16, mlp_ratio=4, n_conv_layers=3, embedding_dim=512, token_in_planes=[192, 256, 320],
                kernel_size=[7, 7, 7], stride=[3, 2, 2], padding=[2, 3, 3], positional_embedding='learnable',
                **kwargs)

    return cct_model


@register_model
def cct_modified17_(**kwargs):
    cct_model = _cct(None, False, None, num_layers=14, num_heads=16, mlp_ratio=4, n_conv_layers=3, embedding_dim=256, token_in_planes=[160, 256, 256],
    # cct_model = _cct(None, False, None, num_layers=14, num_heads=16, mlp_ratio=4, n_conv_layers=3, embedding_dim=512, token_in_planes=[192, 256, 320],
                kernel_size=[7, 7, 7], stride=[3, 3, 2], padding=[2, 2, 3], positional_embedding='learnable',
                **kwargs)

    return cct_model


@register_model
def cct_modified18(**kwargs):
    cct_model = _cct(None, False, None, num_layers=14, num_heads=16, mlp_ratio=4, n_conv_layers=3, embedding_dim=480, token_in_planes=[160, 256, 320],
                kernel_size=[7, 7, 7], stride=[3, 3, 2], padding=[2, 2, 3], positional_embedding='learnable', norm=True,
                **kwargs)

    return cct_model


@register_model
def cct_modified19(**kwargs):
    cct_model = _cct(None, False, None, num_layers=6, num_heads=16, mlp_ratio=4, n_conv_layers=3, embedding_dim=448, token_in_planes=[160, 224],
                kernel_size=[7, 7, 7], stride=[3, 3, 2], padding=[2, 2, 3], positional_embedding='learnable', norm=True,
                **kwargs)

    return cct_model


@register_model
def cct_modified20(**kwargs):
    cct_model = _cct(None, False, None, num_layers=7, num_heads=16, mlp_ratio=2, head_ratio=8, n_conv_layers=3, embedding_dim=480, token_in_planes=[192, 288],
                kernel_size=[7, 7, 7], stride=[3, 3, 2], padding=[2, 2, 3], positional_embedding='learnable', norm=True,
                **kwargs)

    return cct_model


@register_model
def cct_modified21(**kwargs):
    cct_model = _cct(None, False, None, num_layers=7, num_heads=16, mlp_ratio=1, n_conv_layers=3, embedding_dim=480, token_in_planes=[196, 288],
                kernel_size=[7, 7, 7], stride=[3, 3, 2], padding=[2, 2, 3], positional_embedding='learnable', norm=True,
                **kwargs)

    return cct_model


@register_model
def cct_modified22(**kwargs):
    cct_model = _cct(None, False, None, num_layers=7, num_heads=16, mlp_ratio=3, n_conv_layers=3, embedding_dim=480, token_in_planes=[180, 256],
                kernel_size=[7, 7, 7], stride=[3, 3, 2], padding=[2, 2, 3], positional_embedding='learnable', norm=True,
                **kwargs)

    return cct_model


@register_model
def cct_modified23(**kwargs):
    cct_model = _cct(None, False, None, num_layers=6, num_heads=12, mlp_ratio=3, n_conv_layers=3, embedding_dim=384, token_in_planes=[160, 224],
                kernel_size=[7, 5, 5], stride=[3, 2, 2], padding=[0, 0, 0], pooling_padding=0, positional_embedding='learnable', norm=True,
                **kwargs)

    return cct_model


@register_model
def cct_modified24(**kwargs):
    cct_model = _cct(None, False, None, num_layers=6, num_heads=12, mlp_ratio=3, n_conv_layers=3, embedding_dim=384, token_in_planes=[96, 192],
                kernel_size=[7, 5, 5], stride=[3, 2, 2], padding=[0, 0, 0], pooling_padding=0, positional_embedding='learnable', norm=True,
                **kwargs)

    return cct_model



@register_model
def cct_modified25(**kwargs):
    cct_model = _cct(None, False, None, num_layers=6, num_heads=8, mlp_ratio=3, n_conv_layers=3, embedding_dim=256, token_in_planes=[160, 224],
                kernel_size=[7, 5, 5], stride=[3, 2, 2], padding=[0, 0, 0], pooling_padding=0, positional_embedding='learnable', norm=True,
                **kwargs)

    return cct_model


@register_model
def cct_modified26(**kwargs):
    cct_model = _cct(None, False, None, num_layers=6, num_heads=8, mlp_ratio=2, n_conv_layers=3, embedding_dim=384, token_in_planes=[128, 128],
                kernel_size=[5, 3, 3], stride=[3, 2, 2], padding=[0, 0, 0], pooling_padding=0, positional_embedding='learnable', norm=True,
                **kwargs)

    return cct_model

@register_model
def cct_modified27(**kwargs):
    cct_model = _cct(None, False, None, num_layers=6, num_heads=8, mlp_ratio=2, n_conv_layers=3, embedding_dim=480, token_in_planes=[180, 224],
                kernel_size=[5, 3, 3], stride=[3, 2, 2], padding=[0, 0, 0], pooling_padding=1, positional_embedding='learnable', norm=True,
                **kwargs)

    return cct_model


@register_model
def cct_modified28(**kwargs):
    cct_model = _cct(None, False, None, num_layers=7, num_heads=16, mlp_ratio=3, n_conv_layers=3, embedding_dim=480, token_in_planes=[160, 288],
                kernel_size=[7, 3, 3], stride=[3, 2, 2], padding=[1, 1, 1], pooling_padding=1, positional_embedding='learnable', norm=True,
                **kwargs)

    return cct_model

@register_model
def cct_modified29(**kwargs):
    cct_model = _cct(None, False, None, num_layers=6, num_heads=12, mlp_ratio=3, n_conv_layers=3, embedding_dim=384, token_in_planes=[128, 224],
                kernel_size=[7, 5, 3], stride=[3, 2, 2], padding=[1, 1, 1], pooling_padding=1, positional_embedding='learnable', norm=False,
                **kwargs)

    return cct_model


@register_model
def cct_modified30(**kwargs):
    cct_model = _cct(None, False, None, num_layers=4, num_heads=8, mlp_ratio=3, n_conv_layers=3, embedding_dim=256, token_in_planes=[128, 128],
                kernel_size=[7, 5, 3], stride=[3, 2, 2], padding=[1, 1, 1], pooling_padding=1, positional_embedding='learnable', norm=False,
                **kwargs)

    return cct_model

@register_model
def cct_modified31(**kwargs):
    cct_model = _cct(None, False, None, num_layers=7, num_heads=8, mlp_ratio=3, n_conv_layers=3, embedding_dim=384, token_in_planes=[128, 192],
                kernel_size=[7, 5, 3], stride=[3, 2, 2], padding=[1, 1, 1], pooling_padding=1, positional_embedding='learnable', norm=True,
                **kwargs)

    return cct_model


@register_model
def cct_modified32(**kwargs):
    cct_model = _cct(None, False, None, num_layers=3, num_heads=6, mlp_ratio=2, n_conv_layers=3, embedding_dim=192, token_in_planes=[96, 96],
                kernel_size=[7, 5, 3], stride=[3, 2, 2], padding=[1, 1, 1], pooling_padding=1, positional_embedding='learnable', norm=False,
                **kwargs)

    return cct_model

@register_model
def cct_modified33(**kwargs):
    cct_model = _cct(None, False, None, num_layers=4, num_heads=8, mlp_ratio=3, n_conv_layers=3, embedding_dim=128, token_in_planes=[64, 64],
                kernel_size=[7, 5, 3], stride=[3, 2, 2], padding=[1, 1, 1], pooling_padding=1, positional_embedding='learnable', norm=True,
                **kwargs)

    return cct_model


@register_model
def cct_modified34(**kwargs):
    cct_model = _cct(None, False, None, num_layers=3, num_heads=5, mlp_ratio=2, n_conv_layers=3, embedding_dim=160, token_in_planes=[64, 96],
                kernel_size=[7, 3, 3], stride=[3, 2, 2], padding=[1, 1, 1], pooling_padding=1, positional_embedding='learnable', norm=False,
                **kwargs)

    return cct_model

@register_model
def cct_modified35(**kwargs):
    cct_model = _cct(None, False, None, num_layers=2, num_heads=5, mlp_ratio=2, n_conv_layers=3, embedding_dim=160, token_in_planes=[96, 96],
                kernel_size=[7, 5, 3], stride=[3, 2, 2], padding=[1, 1, 1], pooling_padding=1, positional_embedding='learnable', norm=False,
                **kwargs)

    return cct_model


@register_model
def cct_modified36(**kwargs):
    cct_model = _cct(None, False, None, num_layers=4, num_heads=5, mlp_ratio=2, n_conv_layers=3, embedding_dim=180, token_in_planes=[64, 96],
                kernel_size=[7, 3, 3], stride=[3, 2, 2], padding=[1, 1, 1], pooling_padding=1, positional_embedding='learnable', norm=True,
                **kwargs)

    return cct_model

@register_model
def cct_modified37(**kwargs):
    cct_model = _cct(None, False, None, num_layers=4, num_heads=7, mlp_ratio=2, n_conv_layers=3, embedding_dim=224, token_in_planes=[64, 128],
                kernel_size=[7, 3, 3], stride=[3, 2, 2], padding=[1, 1, 1], pooling_padding=1, positional_embedding='learnable', norm=True,
                **kwargs)

    return cct_model

@register_model
def cct_modified38(**kwargs):
    cct_model = _cct(None, False, None, num_layers=4, num_heads=7, mlp_ratio=2, n_conv_layers=3, embedding_dim=224, token_in_planes=[96, 96],
                kernel_size=[7, 3, 3], stride=[3, 2, 2], padding=[1, 1, 1], pooling_padding=1, positional_embedding='learnable', norm=True,
                **kwargs)

    return cct_model

@register_model
def cct_modified39(**kwargs):
    cct_model = _cct(None, False, None, num_layers=6, num_heads=8, mlp_ratio=2, n_conv_layers=3, embedding_dim=192, token_in_planes=[64, 96],
                kernel_size=[7, 3, 3], stride=[3, 2, 2], padding=[1, 1, 1], pooling_padding=1, positional_embedding='learnable', norm=True,
                **kwargs)

    return cct_model


@register_model
def cct_modified40(**kwargs):
    cct_model = _cct(None, False, None, num_layers=7, num_heads=8, mlp_ratio=3, n_conv_layers=3, embedding_dim=256, token_in_planes=[128, 192],
                kernel_size=[7, 3, 3], stride=[3, 2, 2], padding=[1, 1, 1], pooling_padding=1, positional_embedding='learnable', norm=True,
                **kwargs)

    return cct_model


@register_model
def cct_modified41(**kwargs):
    cct_model = _cct(None, False, None, num_layers=7, num_heads=8, mlp_ratio=3, n_conv_layers=3, embedding_dim=128, token_in_planes=[64, 128],
                kernel_size=[7, 5, 3], stride=[3, 2, 2], padding=[1, 1, 1], pooling_padding=1, positional_embedding='learnable', norm=True,
                **kwargs)

    return cct_model


@register_model
def cct_modified42(**kwargs):
    cct_model = _cct(None, False, None, num_layers=7, num_heads=8, mlp_ratio=3, n_conv_layers=3, embedding_dim=384, token_in_planes=[128, 192],
                kernel_size=[7, 5, 3], stride=[3, 2, 2], padding=[1, 1, 1], pooling_kernel_size=2, pooling_stride=2, pooling_padding=0, positional_embedding='learnable', norm=True,
                **kwargs)

    return cct_model


@register_model
def cct_modified43(**kwargs):
    cct_model = _cct(None, False, None, num_layers=3, num_heads=8, mlp_ratio=3, n_conv_layers=3, embedding_dim=384, token_in_planes=[128, 192],
                kernel_size=[7, 5, 3], stride=[3, 2, 2], padding=[1, 1, 1], pooling_padding=1, positional_embedding='learnable', norm=True,
                **kwargs)

    return cct_model


@register_model
def cct_modified44(**kwargs):
    cct_model = _cct(None, False, None, num_layers=10, num_heads=12, mlp_ratio=3, n_conv_layers=3, embedding_dim=384, token_in_planes=[128, 192],
                kernel_size=[7, 5, 3], stride=[3, 2, 2], padding=[1, 1, 1], pooling_padding=1, positional_embedding='learnable', norm=True,
                **kwargs)

    return cct_model


@register_model
def cct_modified45(**kwargs):
    cct_model = _cct(None, False, None, num_layers=7, num_heads=16, mlp_ratio=3, n_conv_layers=3, embedding_dim=480, token_in_planes=[160, 288],
                kernel_size=[7, 3, 3], stride=[3, 2, 2], padding=[1, 1, 1], pooling_padding=1, positional_embedding='learnable', norm=True,
                **kwargs)

    return cct_model

@register_model
def cct_modified46(**kwargs):
    cct_model = _cct(None, False, None, num_layers=7, num_heads=8, mlp_ratio=3, n_conv_layers=3, embedding_dim=128, token_in_planes=[96, 128],
                kernel_size=[7, 3, 3], stride=[3, 2, 2], padding=[1, 1, 1], pooling_padding=1, positional_embedding='learnable', norm=True,
                **kwargs)

    return cct_model


@register_model
def cct_modified47(**kwargs):
    cct_model = _cct(None, False, None, num_layers=7, num_heads=8, mlp_ratio=3, n_conv_layers=3, embedding_dim=256, token_in_planes=[128, 192],
                kernel_size=[7, 7, 7], stride=[3, 2, 2], padding=[2, 3, 3], pooling_padding=1, positional_embedding='learnable', norm=False,
                **kwargs)

    return cct_model


@register_model
def cct_modified48(**kwargs):
    cct_model = _cct(None, False, None, num_layers=7, num_heads=8, mlp_ratio=3, n_conv_layers=3, embedding_dim=192, token_in_planes=[128, 144],
                # kernel_size=[7, 3, 3], stride=[3, 2, 2], padding=[1, 1, 1], pooling_padding=1, positional_embedding='learnable', norm=False,
                kernel_size=[7, 3, 3], stride=[3, 2, 2], padding=[1, 1, 1], pooling_padding=1, positional_embedding='learnable',
                **kwargs)

    return cct_model


@register_model
def cct_modified49(**kwargs):
    cct_model = _cct(None, False, None, num_layers=6, num_heads=8, mlp_ratio=3, n_conv_layers=3, embedding_dim=192, token_in_planes=[128, 144],
                kernel_size=[7, 3, 3], stride=[3, 2, 2], padding=[1, 1, 1], pooling_padding=1, positional_embedding='learnable',
                **kwargs)

    return cct_model


@register_model
def cct_modified50(**kwargs):
    cct_model = _cct(None, False, None, num_layers=10, num_heads=8, mlp_ratio=3, n_conv_layers=3, embedding_dim=192, token_in_planes=[128, 144],
                kernel_size=[7, 3, 3], stride=[3, 2, 2], padding=[1, 1, 1], pooling_padding=1, positional_embedding='learnable',
                **kwargs)

    return cct_model

@register_model
def cct_modified51(**kwargs):
    cct_model = _cct(None, False, None, num_layers=10, num_heads=8, mlp_ratio=2, n_conv_layers=3, embedding_dim=192, token_in_planes=[128, 144],
                kernel_size=[7, 3, 3], stride=[3, 2, 2], padding=[1, 1, 1], pooling_padding=1, positional_embedding='learnable',
                **kwargs)

    return cct_model

@register_model
def cct_modified52(**kwargs):
    cct_model = _cct(None, False, None, num_layers=6, num_heads=8, mlp_ratio=2, n_conv_layers=3, embedding_dim=192, token_in_planes=[128, 144],
                kernel_size=[7, 3, 3], stride=[3, 2, 2], padding=[1, 1, 1], pooling_padding=1, positional_embedding='learnable',
                **kwargs)

    return cct_model


@register_model
def cct_modified53(**kwargs):
    cct_model = _cct(None, False, None, num_layers=7, num_heads=8, mlp_ratio=2, n_conv_layers=3, embedding_dim=192, token_in_planes=[128, 144],
                kernel_size=[5, 3, 3], stride=[3, 2, 2], padding=[1, 1, 1], pooling_padding=1, positional_embedding='learnable',
                **kwargs)

    return cct_model



@register_model
def cct_modified54(**kwargs):
    cct_model = _cct(None, False, None, num_layers=4, num_heads=8, mlp_ratio=3, n_conv_layers=3, embedding_dim=192, token_in_planes=[96, 96],
                kernel_size=[7, 3, 3], stride=[3, 2, 2], padding=[1, 1, 1], pooling_padding=1, positional_embedding='learnable',
                **kwargs)

    return cct_model


@register_model
def cct_modified55(**kwargs):
    cct_model = _cct(None, False, None, num_layers=3, num_heads=4, mlp_ratio=2, n_conv_layers=3, embedding_dim=160, token_in_planes=[96, 96],
                kernel_size=[7, 5, 3], stride=[3, 2, 1], padding=[1, 1, 1], pooling_kernel_size=2, pooling_stride=2, pooling_padding=0, positional_embedding='learnable', norm=True,
                **kwargs)

    return cct_model


@register_model
def cct_modified56(**kwargs):
    cct_model = _cct(None, False, None, num_layers=17, num_heads=6, mlp_ratio=2, n_conv_layers=3, embedding_dim=144, token_in_planes=[96, 128],
                kernel_size=[7, 3, 3], stride=[3, 2, 2], padding=[1, 1, 1], pooling_kernel_size=3, pooling_stride=2, pooling_padding=1, positional_embedding='learnable', norm=True,
                max_pool=[True, True, True],
                **kwargs)

    return cct_model


@register_model
def cct_modified57(**kwargs):
    cct_model = _cct(None, False, None, num_layers=17, num_heads=8, mlp_ratio=3, n_conv_layers=3, embedding_dim=256, token_in_planes=[128, 160],
                kernel_size=[7, 3, 3], stride=[3, 2, 2], padding=[1, 1, 1], pooling_kernel_size=3, pooling_stride=2, pooling_padding=1, positional_embedding='learnable', norm=True,
                **kwargs)

    return cct_model


@register_model
def cct_modified58(**kwargs):
    cct_model = _cct(None, False, None, num_layers=12, num_heads=8, mlp_ratio=2, n_conv_layers=3, embedding_dim=160, token_in_planes=[96, 128, 144],
                kernel_size=[7, 3, 3, 3], stride=[4, 2, 2, 2], padding=[1, 1, 1, 1], pooling_kernel_size=3, pooling_stride=2, pooling_padding=1, positional_embedding='learnable', norm=True, downsample=2,
                **kwargs)

    return cct_model


@register_model
def cct_modified59(**kwargs):
    cct_model = _cct(None, False, None, num_layers=7, num_heads=6, mlp_ratio=2, n_conv_layers=5, embedding_dim=192, token_in_planes=[96, 144, 160, 160],
                kernel_size=[7, 5, 5, 3, 3], stride=[3, 2, 2, 1, 1], padding=[1, 1, 1, 1, 1], pooling_kernel_size=3, pooling_stride=2, pooling_padding=1, positional_embedding='learnable', norm=True,
                downsample=2, max_pool=[False, False, True, False, True],
                **kwargs)

    return cct_model


@register_model
def cct_modified60(**kwargs):
    cct_model = _cct(None, False, None, num_layers=12, num_heads=8, mlp_ratio=2, n_conv_layers=7, embedding_dim=160, token_in_planes=[32, 96, 128, 128, 144, 144],
                kernel_size=[5, 5, 5, 3, 3, 3, 3], stride=[2, 2, 1, 1, 1, 1, 1], padding=[1, 1, 1, 1, 1, 1, 1], pooling_kernel_size=3, pooling_stride=2, pooling_padding=1, positional_embedding='learnable', norm=True, downsample=2,
                max_pool=[False, True, True, False, True, False, True],
                **kwargs)

    return cct_model


@register_model
def cct_modified61(**kwargs):
    cct_model = _cct(None, False, None, num_layers=17, num_heads=8, mlp_ratio=2, n_conv_layers=3, embedding_dim=160, token_in_planes=[108, 128],
                kernel_size=[7, 3, 3], stride=[3, 2, 2], padding=[1, 1, 1], pooling_kernel_size=3, pooling_stride=2, pooling_padding=1, positional_embedding='learnable', norm=True,
                max_pool=[True, True, True],
                **kwargs)

    return cct_model


@register_model
def cct_modified62(**kwargs):
    cct_model = _cct(None, False, None, num_layers=6, num_heads=4, mlp_ratio=3, n_conv_layers=3, embedding_dim=144, token_in_planes=[96, 128],
                kernel_size=[7, 3, 3], stride=[3, 2, 2], padding=[1, 1, 1], pooling_kernel_size=3, pooling_stride=2, pooling_padding=1, positional_embedding='learnable', norm=True,
                max_pool=[True, True, True],
                **kwargs)

    return cct_model


@register_model
def cct_modified63(**kwargs):
    cct_model = _cct(None, False, None, num_layers=17, num_heads=6, mlp_ratio=2, n_conv_layers=3, embedding_dim=192, token_in_planes=[96, 144],
                kernel_size=[7, 3, 3], stride=[3, 2, 2], padding=[1, 1, 1], pooling_kernel_size=3, pooling_stride=2, pooling_padding=1, positional_embedding='learnable', norm=True,
                max_pool=[True, True, True],
                **kwargs)

    return cct_model


@register_model
def cct_modified64(**kwargs):
    cct_model = _cct(None, False, None, num_layers=12, num_heads=4, mlp_ratio=2, n_conv_layers=3, embedding_dim=128, token_in_planes=[96, 128],
                kernel_size=[7, 3, 3], stride=[3, 2, 2], padding=[1, 1, 1], pooling_kernel_size=3, pooling_stride=2, pooling_padding=1, positional_embedding='learnable', norm=True,
                max_pool=[True, True, True],
                **kwargs)

    return cct_model


@register_model
def cct_modified65(**kwargs):
    cct_model = _cct(None, False, None, num_layers=14, num_heads=4, mlp_ratio=2, n_conv_layers=3, embedding_dim=128, token_in_planes=[64, 96],
                kernel_size=[7, 3, 3], stride=[3, 2, 2], padding=[1, 1, 1], pooling_kernel_size=3, pooling_stride=2, pooling_padding=1, positional_embedding='learnable', norm=True,
                max_pool=[True, True, True],
                **kwargs)

    return cct_model

@register_model
def cct_modified66(**kwargs):
    cct_model = _cct(None, False, None, num_layers=12, num_heads=4, mlp_ratio=1, n_conv_layers=3, embedding_dim=96, token_in_planes=[64, 72],
                kernel_size=[7, 3, 3], stride=[3, 2, 2], padding=[1, 1, 1], pooling_kernel_size=3, pooling_stride=2, pooling_padding=1, positional_embedding='learnable', norm=True,
                max_pool=[True, True, True],
                **kwargs)

    return cct_model

@register_model
def cct_modified67(**kwargs):
    cct_model = _cct(None, False, None, num_layers=7, num_heads=3, mlp_ratio=1, n_conv_layers=3, embedding_dim=72, token_in_planes=[48, 64],
                kernel_size=[7, 3, 3], stride=[3, 2, 2], padding=[1, 1, 1], pooling_kernel_size=3, pooling_stride=2, pooling_padding=1, positional_embedding='learnable', norm=True,
                max_pool=[True, True, True],
                **kwargs)

    return cct_model




@register_model
def cct_modified68(**kwargs):
    cct_model = _cct(None, False, None, num_layers=21, num_heads=6, mlp_ratio=2, n_conv_layers=3, embedding_dim=144, token_in_planes=[96, 128],
                kernel_size=[7, 3, 3], stride=[3, 2, 2], padding=[1, 1, 1], pooling_kernel_size=3, pooling_stride=2, pooling_padding=1, positional_embedding='learnable', norm=True,
                max_pool=[True, True, True],
                **kwargs)

    return cct_model


@register_model
def cct_modified69(**kwargs):
    cct_model = _cct(None, False, None, num_layers=19, num_heads=6, mlp_ratio=2, n_conv_layers=3, embedding_dim=144, token_in_planes=[96, 128],
                kernel_size=[7, 3, 3], stride=[3, 2, 2], padding=[1, 1, 1], pooling_kernel_size=3, pooling_stride=2, pooling_padding=1, positional_embedding='learnable', norm=True,
                max_pool=[True, True, True],
                **kwargs)

    return cct_model


@register_model
def cct_modified70(**kwargs):
    cct_model = _cct(None, False, None, num_layers=15, num_heads=6, mlp_ratio=2, n_conv_layers=3, embedding_dim=144, token_in_planes=[96, 128],
                kernel_size=[7, 3, 3], stride=[3, 2, 2], padding=[1, 1, 1], pooling_kernel_size=3, pooling_stride=2, pooling_padding=1, positional_embedding='learnable', norm=True,
                max_pool=[True, True, True],
                **kwargs)

    return cct_model

@register_model
def cct_modified71(**kwargs):
    cct_model = _cct(None, False, None, num_layers=13, num_heads=6, mlp_ratio=2, n_conv_layers=3, embedding_dim=144, token_in_planes=[96, 128],
                kernel_size=[7, 3, 3], stride=[3, 2, 2], padding=[1, 1, 1], pooling_kernel_size=3, pooling_stride=2, pooling_padding=1, positional_embedding='learnable', norm=True,
                max_pool=[True, True, True],
                **kwargs)

    return cct_model

@register_model
def cct_modified72(**kwargs):
    cct_model = _cct(None, False, None, num_layers=3, num_heads=6, mlp_ratio=2, n_conv_layers=3, embedding_dim=144, token_in_planes=[96, 128],
                kernel_size=[7, 3, 3], stride=[3, 2, 2], padding=[1, 1, 1], pooling_kernel_size=3, pooling_stride=2, pooling_padding=1, positional_embedding='learnable', norm=True,
                max_pool=[True, True, True],
                **kwargs)

    return cct_model

@register_model
def cct_modified73(**kwargs):
    cct_model = _cct(None, False, None, num_layers=2, num_heads=6, mlp_ratio=2, n_conv_layers=3, embedding_dim=144, token_in_planes=[96, 128],
                kernel_size=[7, 3, 3], stride=[3, 2, 2], padding=[1, 1, 1], pooling_kernel_size=3, pooling_stride=2, pooling_padding=1, positional_embedding='learnable', norm=True,
                max_pool=[True, True, True],
                **kwargs)

    return cct_model

@register_model
def cct_modified74(**kwargs):
    cct_model = _cct(None, False, None, num_layers=1, num_heads=6, mlp_ratio=2, n_conv_layers=3, embedding_dim=144, token_in_planes=[96, 128],
                kernel_size=[7, 3, 3], stride=[3, 2, 2], padding=[1, 1, 1], pooling_kernel_size=3, pooling_stride=2, pooling_padding=1, positional_embedding='learnable', norm=True,
                max_pool=[True, True, True],
                **kwargs)

    return cct_model

@register_model
def cct_modified75(**kwargs):
    cct_model = _cct(None, False, None, num_layers=5, num_heads=6, mlp_ratio=2, n_conv_layers=3, embedding_dim=144, token_in_planes=[96, 128],
                kernel_size=[7, 3, 3], stride=[3, 2, 2], padding=[1, 1, 1], pooling_kernel_size=3, pooling_stride=2, pooling_padding=1, positional_embedding='learnable', norm=True,
                max_pool=[True, True, True],
                **kwargs)

    return cct_model

@register_model
def cct_modified76(**kwargs):
    cct_model = _cct(None, False, None, num_layers=7, num_heads=6, mlp_ratio=2, n_conv_layers=3, embedding_dim=144, token_in_planes=[96, 128],
                kernel_size=[7, 3, 3], stride=[3, 2, 2], padding=[1, 1, 1], pooling_kernel_size=3, pooling_stride=2, pooling_padding=1, positional_embedding='learnable', norm=True,
                max_pool=[True, True, True],
                **kwargs)

    return cct_model

@register_model
def cct_modified77(**kwargs):
    cct_model = _cct(None, False, None, num_layers=9, num_heads=6, mlp_ratio=2, n_conv_layers=3, embedding_dim=144, token_in_planes=[96, 128],
                kernel_size=[7, 3, 3], stride=[3, 2, 2], padding=[1, 1, 1], pooling_kernel_size=3, pooling_stride=2, pooling_padding=1, positional_embedding='learnable', norm=True,
                max_pool=[True, True, True],
                **kwargs)

    return cct_model

@register_model
def cct_modified78(**kwargs):
    cct_model = _cct(None, False, None, num_layers=11, num_heads=6, mlp_ratio=2, n_conv_layers=3, embedding_dim=144, token_in_planes=[96, 128],
                kernel_size=[7, 3, 3], stride=[3, 2, 2], padding=[1, 1, 1], pooling_kernel_size=3, pooling_stride=2, pooling_padding=1, positional_embedding='learnable', norm=True,
                max_pool=[True, True, True],
                **kwargs)

    return cct_model
