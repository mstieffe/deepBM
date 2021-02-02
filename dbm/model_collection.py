import torch
import torch.nn as nn

from dbm.util import compute_same_padding


def _facify(n, fac):
    return int(n // fac)


def _sn_to_specnorm(sn: int):
    if sn > 0:

        def specnorm(module):
            return nn.utils.spectral_norm(module, n_power_iterations=sn)

    else:

        def specnorm(module, **kw):
            return module

    return specnorm


class Residual3DConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        n_filters,
        kernel_size,
        stride,
        trans=False,
        sn: int = 0,
        device=None,
    ):
        super(Residual3DConvBlock, self).__init__()
        specnorm = _sn_to_specnorm(sn)
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.trans = trans

        if self.stride != 1:
            self.downsample = nn.Sequential(
                nn.MaxPool3d(kernel_size=2, stride=2),
                specnorm(
                    nn.Conv3d(
                        in_channels,
                        out_channels=self.n_filters,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                    )
                ),
            )
        elif self.trans:
            self.downsample = nn.Sequential(
                specnorm(
                    nn.Conv3d(
                        in_channels,
                        out_channels=self.n_filters,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                    )
                )
            )
        else:
            self.downsample = nn.Identity()
        self.downsample = self.downsample.to(device=device)

        same_padding = compute_same_padding(self.kernel_size, self.stride, dilation=1)
        block_elements = [
            specnorm(
                nn.Conv3d(
                    in_channels=in_channels,
                    out_channels=self.n_filters,
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    padding=same_padding,
                )
            ),
            nn.GroupNorm(1, num_channels=self.n_filters),
            nn.LeakyReLU(),
            specnorm(
                nn.Conv3d(
                    in_channels=self.n_filters,
                    out_channels=self.n_filters,
                    kernel_size=self.kernel_size,
                    stride=1,
                    padding=same_padding,
                )
            ),
            nn.GroupNorm(1, num_channels=self.n_filters),
        ]
        self.block = nn.Sequential(*tuple(block_elements)).to(device=device)
        self.nonlin = nn.LeakyReLU()

    def forward(self, inputs):
        out = self.block(inputs)
        downsampled = self.downsample(inputs)
        result = 0.5 * (out + downsampled)
        result = self.nonlin(result)
        return result



class Residual3DConvBlock_drop(nn.Module):
    def __init__(
        self,
        in_channels,
        n_filters,
        kernel_size,
        stride,
        trans=False,
        sn: int = 0,
        device=None,
    ):
        super(Residual3DConvBlock_drop, self).__init__()
        specnorm = _sn_to_specnorm(sn)
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.trans = trans

        if self.stride != 1:
            self.downsample = nn.Sequential(
                nn.MaxPool3d(kernel_size=2, stride=2),
                specnorm(
                    nn.Conv3d(
                        in_channels,
                        out_channels=self.n_filters,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                    )
                ),
            )
        elif self.trans:
            self.downsample = nn.Sequential(
                specnorm(
                    nn.Conv3d(
                        in_channels,
                        out_channels=self.n_filters,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                    )
                )
            )
        else:
            self.downsample = nn.Identity()
        self.downsample = self.downsample.to(device=device)

        same_padding = compute_same_padding(self.kernel_size, self.stride, dilation=1)
        block_elements = [
            specnorm(
                nn.Conv3d(
                    in_channels=in_channels,
                    out_channels=self.n_filters,
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    padding=same_padding,
                )
            ),
            nn.Dropout3d(),
            nn.GroupNorm(1, num_channels=self.n_filters),
            nn.LeakyReLU(),
            specnorm(
                nn.Conv3d(
                    in_channels=self.n_filters,
                    out_channels=self.n_filters,
                    kernel_size=self.kernel_size,
                    stride=1,
                    padding=same_padding,
                )
            ),
            nn.Dropout3d(),
            nn.GroupNorm(1, num_channels=self.n_filters),
        ]
        self.block = nn.Sequential(*tuple(block_elements)).to(device=device)
        self.nonlin = nn.LeakyReLU()

    def forward(self, inputs):
        out = self.block(inputs)
        downsampled = self.downsample(inputs)
        result = 0.5 * (out + downsampled)
        result = self.nonlin(result)
        return result


class Residual3DDeconvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters, kernel_size, sn=0):
        super(Residual3DDeconvBlock, self).__init__()

        specnorm = _sn_to_specnorm(sn)

        self.n_filters_in = n_filters_in
        self.n_filters = n_filters
        self.kernel_size = kernel_size

        self.conv = specnorm(
            nn.Conv3d(n_filters_in, n_filters, kernel_size=1, stride=1)
        )
        same_padding = compute_same_padding(self.kernel_size, 1, dilation=1)
        block_blocks = [
            specnorm(
                nn.Conv3d(
                    n_filters,
                    n_filters,
                    kernel_size=self.kernel_size,
                    stride=1,
                    padding=same_padding,
                )
            ),
            nn.GroupNorm(1, num_channels=n_filters),
            nn.LeakyReLU(),
            specnorm(
                nn.Conv3d(
                    n_filters,
                    n_filters,
                    kernel_size=self.kernel_size,
                    stride=1,
                    padding=same_padding,
                )
            ),
            nn.GroupNorm(1, num_channels=n_filters),
        ]
        self.block = nn.Sequential(*tuple(block_blocks))
        self.nonlin = nn.LeakyReLU()

    def forward(self, inputs: torch.Tensor):
        # print("SHAPE", inputs.shape)
        inputs = inputs.repeat(1, 1, 2, 2, 2)  # B, C, dims repeat factor 2
        inputs = self.conv(inputs)
        out = self.block(inputs)
        out = 0.5 * (out + inputs)
        out = self.nonlin(out)
        return out


class Critic(nn.Module):
    def __init__(
        self, width, height, depth, in_channels, fac=1, sn: int = 0, device=None
    ):
        super().__init__()
        specnorm = _sn_to_specnorm(sn)
        self.step1 = specnorm(
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=_facify(128, fac),
                kernel_size=5,
                stride=1,
                padding=compute_same_padding(5, 1, 1),
            )
        )
        self.step2 = nn.LeakyReLU()

        self.step3 = Residual3DConvBlock(
            in_channels=_facify(128, fac),
            n_filters=_facify(128, fac),
            kernel_size=3,
            stride=1,
            sn=sn,
            device=device,
        )
        self.step4 = Residual3DConvBlock(
            in_channels=_facify(128, fac),
            n_filters=_facify(256, fac),
            kernel_size=3,
            stride=2,
            sn=sn,
            device=device,
        )
        self.step5 = Residual3DConvBlock(
            in_channels=_facify(256, fac),
            n_filters=_facify(256, fac),
            kernel_size=3,
            stride=1,
            sn=sn,
            device=device,
        )
        self.step6 = Residual3DConvBlock(
            in_channels=_facify(256, fac),
            n_filters=_facify(512, fac),
            kernel_size=3,
            stride=2,
            sn=sn,
            device=device,
        )
        self.step7 = specnorm(
            nn.Conv3d(
                in_channels=_facify(512, fac),
                out_channels=_facify(512, fac),
                kernel_size=3,
                stride=1,
                padding=compute_same_padding(3, 1, 1),
            )
        )
        self.step8 = nn.GroupNorm(1, _facify(512, fac))
        self.step9 = nn.LeakyReLU()
        self.to_critic_value = specnorm(
            nn.Linear(
                in_features=(width // 4)
                * (height // 4)
                * (depth // 4)
                * _facify(512, fac),
                out_features=1,
            )
        )

    def forward(self, inputs):
        out = self.step1(inputs)
        out = self.step2(out)
        out = self.step3(out)
        out = self.step4(out)
        out = self.step5(out)
        out = self.step6(out)
        out = self.step7(out)
        out = self.step8(out)
        out = self.step9(out)
        out = torch.flatten(out, start_dim=1, end_dim=-1)
        out = self.to_critic_value(out)
        return out


class EmbedNoise(nn.Module):
    def __init__(self, z_dim, channels, sn=0):
        super().__init__()
        specnorm = _sn_to_specnorm(sn)
        self.pad = nn.Linear(z_dim, channels * 4 * 4 * 4)
        self.pad = specnorm(self.pad)
        # self.pad = nn.ConstantPad3d(padding=(3, 3, 3, 3, 3, 3), value=0.)  # -> (B, z_dim, 7, 7, 7)
        # self.conv = nn.Conv3d(z_dim, channels, kernel_size=4, stride=1, padding=0)  # -> (B, channels, 4, 4, 4)
        self.nonlin = nn.LeakyReLU()
        self.z_dim = z_dim
        self.channels = channels

    def forward(self, z):
        # batch_size = z.shape[0]
        out = self.pad(z)
        # out = self.conv(out.view((-1, self.z_dim, 7, 7, 7)))
        out = self.nonlin(out)
        out = out.view((-1, self.channels, 4, 4, 4))
        return out


class GeneratorCombined1Block(nn.Module):
    def __init__(self, in_channels, out_channels, sn=0):
        super().__init__()
        specnorm = _sn_to_specnorm(sn)
        self.conv = specnorm(
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            )
        )
        self.group_norm = nn.GroupNorm(1, num_channels=out_channels)
        self.nonlin = nn.LeakyReLU()
        self.res1 = Residual3DConvBlock(out_channels, out_channels, 3, 1, sn=sn)
        self.res2 = Residual3DConvBlock(out_channels, out_channels, 3, 1, sn=sn)
        self.res_deconv = Residual3DDeconvBlock(
            n_filters_in=out_channels, n_filters=out_channels, kernel_size=3, sn=sn
        )

    def forward(self, embedded_z, down2):
        out = torch.cat((embedded_z, down2), dim=1)
        out = self.conv(out)
        out = self.group_norm(out)
        out = self.nonlin(out)
        out = self.res1(out)
        out = self.res2(out)
        out = self.res_deconv(out)
        return out


class GeneratorCombined2Block(nn.Module):
    def __init__(self, channels, sn=0):
        super().__init__()
        self.conv = Residual3DConvBlock(
            channels, n_filters=channels, kernel_size=3, stride=1, sn=sn
        )
        self.deconv = Residual3DDeconvBlock(
            n_filters_in=channels, n_filters=channels, kernel_size=3, sn=sn
        )

    def forward(self, inputs):
        out = self.conv(inputs)
        out = self.deconv(out)
        return out

class GeneratorCombined3Block(nn.Module):
    def __init__(self, in_channels, out_channels, sn=0):
        super().__init__()
        specnorm = _sn_to_specnorm(sn)
        self.conv = specnorm(
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            )
        )
        self.group_norm = nn.GroupNorm(1, num_channels=out_channels)
        self.nonlin = nn.LeakyReLU()
        self.res_deconv = Residual3DDeconvBlock(
            n_filters_in=out_channels, n_filters=out_channels, kernel_size=3, sn=sn
        )

    def forward(self, embedded_z, down2):
        out = torch.cat((embedded_z, down2), dim=1)
        out = self.conv(out)
        out = self.group_norm(out)
        out = self.nonlin(out)
        out = self.res_deconv(out)
        return out


class Generator(nn.Module):
    def __init__(
        self,
        z_dim,
        condition_n_channels,
        fac=1,
        testmode=False,
        sn: int = 0,
        device=None,
    ):
        super().__init__()
        specnorm = _sn_to_specnorm(sn)
        embed_condition_blocks = [
            specnorm(
                nn.Conv3d(
                    in_channels=condition_n_channels,
                    out_channels=_facify(128, fac),
                    kernel_size=5,
                    stride=1,
                    padding=2,
                )
            ),
            nn.GroupNorm(1, num_channels=_facify(128, fac)),
            nn.LeakyReLU(),
            Residual3DConvBlock(
                _facify(128, fac), _facify(128, fac), kernel_size=3, stride=1, sn=sn
            ),
            Residual3DConvBlock(
                _facify(128, fac), _facify(128, fac), kernel_size=3, stride=1, sn=sn
            ),
        ]
        self.embed_condition = nn.Sequential(*tuple(embed_condition_blocks)).to(
            device=device
        )
        self.downsample_cond1 = Residual3DConvBlock(
            _facify(128, fac),
            n_filters=_facify(128, fac),
            kernel_size=3,
            stride=2,
            sn=sn,
        )
        self.downsample_cond2 = Residual3DConvBlock(
            _facify(128, fac),
            n_filters=_facify(128, fac),
            kernel_size=3,
            stride=2,
            sn=sn,
        )

        self.embed_noise = EmbedNoise(z_dim, _facify(128, fac), sn=sn)
        self.combined1 = GeneratorCombined1Block(
            _facify(256, fac), _facify(128, fac), sn=sn
        )
        self.combined2 = GeneratorCombined2Block(_facify(256, fac), sn=sn)

        to_image_blocks = [
            Residual3DConvBlock(
                _facify(384, fac), _facify(256, fac), 3, 1, trans=True, sn=sn
            ),
            Residual3DConvBlock(_facify(256, fac), _facify(256, fac), 3, 1, sn=sn),
            specnorm(nn.Conv3d(_facify(256, fac), 53, kernel_size=1, stride=1)),
            nn.Sigmoid() if not testmode else nn.Tanh(),
        ]
        self.to_image = nn.Sequential(*tuple(to_image_blocks)).to(device=device)
        self.testmode = testmode

    def forward(self, z, c):
        embedded_c = self.embed_condition(c)
        down1 = self.downsample_cond1(embedded_c)
        down2 = self.downsample_cond2(down1)

        embedded_z = self.embed_noise(z)

        out = self.combined1(embedded_z, down2)

        c_down1 = torch.cat((out, down1), dim=1)
        out = self.combined2(c_down1)

        out = torch.cat((out, embedded_c), dim=1)
        out = (
            c[:, :53, ...] + self.to_image(out) if self.testmode else self.to_image(out)
        )

        return out


class Critic2(nn.Module):
    def __init__(
        self, in_channels, fac=1, sn: int = 0, device=None
    ):
        super(Critic2, self).__init__()
        specnorm = _sn_to_specnorm(sn)

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [
                specnorm(nn.Conv3d(in_filters, out_filters, 3, 2, 1)),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout3d(0.25),
            ]
            if bn:
                block.append(nn.BatchNorm3d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(in_channels, _facify(256, fac), bn=False),
            *discriminator_block(_facify(256, fac), _facify(256, fac)),
            *discriminator_block(_facify(256, fac), _facify(256, fac)),
            *discriminator_block(_facify(256, fac), _facify(256, fac)),
            *discriminator_block(_facify(256, fac), _facify(256, fac)),
            *discriminator_block(_facify(256, fac), _facify(256, fac)),
        )
        self.to_critic_value = specnorm(
            nn.Linear(in_features=_facify(256, fac), out_features=1,)
        )
        self.to(device=device)

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        out = self.to_critic_value(out)
        return out


class Generator2(nn.Module):
    def __init__(self, z_dim, condition_n_channels, fac=1, sn: int = 0, device=None):
        super(Generator2, self).__init__()
        specnorm = _sn_to_specnorm(sn)
        embed_condition_blocks = [
            specnorm(
                nn.Conv3d(
                    in_channels=condition_n_channels,
                    out_channels=_facify(128, fac),
                    kernel_size=5,
                    stride=1,
                    padding=2,
                )
            ),
            nn.GroupNorm(1, num_channels=_facify(128, fac)),
            nn.LeakyReLU(inplace=True),
            Residual3DConvBlock(
                _facify(128, fac), _facify(128, fac), kernel_size=3, stride=1, sn=sn
            ),
            Residual3DConvBlock(
                _facify(128, fac), _facify(128, fac), kernel_size=3, stride=1, sn=sn
            ),
            nn.MaxPool3d(2),
            Residual3DConvBlock(
                _facify(128, fac), _facify(128, fac), kernel_size=3, stride=1, sn=sn
            ),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool3d(2),
        ]
        self.embed_condition = nn.Sequential(*tuple(embed_condition_blocks)).to(
            device=device
        )
        self.embed_noise = EmbedNoise(z_dim, _facify(128, fac), sn=sn)

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm3d(2 * _facify(128, fac)),
            nn.Upsample(scale_factor=2),
            specnorm(
                nn.Conv3d(
                    2 * _facify(128, fac), _facify(128, fac), 3, stride=1, padding=1
                )
            ),
            nn.BatchNorm3d(_facify(128, fac), 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            specnorm(
                nn.Conv3d(_facify(128, fac), _facify(64, fac), 3, stride=1, padding=1)
            ),
            nn.BatchNorm3d(_facify(64, fac), 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            specnorm(nn.Conv3d(_facify(64, fac), 1, 3, stride=1, padding=1)),
            nn.Sigmoid(),
        )
        self.to(device=device)

    def forward(self, z, c):
        embedded_c = self.embed_condition(c)
        embedded_z = self.embed_noise(z)
        out = torch.cat((embedded_z, embedded_c), dim=1)
        out = self.conv_blocks(out)
        return out





class AtomCritic(nn.Module):
    def __init__(
        self, in_channels, fac=1, sn: int = 0, device=None
    ):
        super(AtomCritic, self).__init__()
        specnorm = _sn_to_specnorm(sn)

        def discriminator_block(in_filters, out_filters):
            block = [
                specnorm(nn.Conv3d(in_filters, out_filters, 3, 2, 1)),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout3d(0.25),
                nn.GroupNorm(1, num_channels=out_filters)
            ]
            return block

        self.model = nn.Sequential(
            *discriminator_block(in_channels, _facify(256, fac)),
            *discriminator_block(_facify(256, fac), _facify(256, fac)),
            *discriminator_block(_facify(256, fac), _facify(256, fac)),
            *discriminator_block(_facify(256, fac), _facify(256, fac)),
            *discriminator_block(_facify(256, fac), _facify(256, fac)),
            *discriminator_block(_facify(256, fac), _facify(256, fac)),
        )
        self.to_critic_value = specnorm(
            nn.Linear(in_features=_facify(256, fac), out_features=1,)
        )
        self.to(device=device)

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        out = self.to_critic_value(out)
        return out



class AtomCrit2(nn.Module):
    def __init__(
        self, in_channels, fac=1, sn: int = 0, device=None
    ):
        super().__init__()
        specnorm = _sn_to_specnorm(sn)
        self.step1 = specnorm(
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=_facify(128, fac),
                kernel_size=5,
                stride=1,
                padding=compute_same_padding(5, 1, 1),
            )
        )
        self.step2 = nn.LeakyReLU()

        self.step3 = Residual3DConvBlock(
            in_channels=_facify(128, fac),
            n_filters=_facify(128, fac),
            kernel_size=3,
            stride=1,
            sn=sn,
            device=device,
        )
        self.step4 = Residual3DConvBlock(
            in_channels=_facify(128, fac),
            n_filters=_facify(256, fac),
            kernel_size=3,
            stride=2,
            sn=sn,
            device=device,
        )
        self.step5 = Residual3DConvBlock(
            in_channels=_facify(256, fac),
            n_filters=_facify(256, fac),
            kernel_size=3,
            stride=1,
            sn=sn,
            device=device,
        )

        self.step6 = specnorm(
            nn.Conv3d(
                in_channels=_facify(256, fac),
                out_channels=_facify(256, fac),
                kernel_size=3,
                stride=1,
                padding=compute_same_padding(3, 1, 1),
            )
        )
        self.step7 = nn.GroupNorm(1, _facify(256, fac))
        self.step8 = nn.LeakyReLU()

        self.step9 = specnorm(
            nn.Conv3d(
                in_channels=_facify(256, fac),
                out_channels=_facify(128, fac),
                kernel_size=4,
                stride=1,
                padding=0,
            )
        )
        self.step10 = nn.GroupNorm(1, _facify(128, fac))
        self.step11 = nn.LeakyReLU()

        self.to_critic_value = specnorm(
            nn.Linear(
                in_features=_facify(128, fac), out_features=1
            )
        )

    def forward(self, inputs):
        out = self.step1(inputs)
        out = self.step2(out)
        out = self.step3(out)
        out = self.step4(out)
        out = self.step5(out)
        out = self.step6(out)
        out = self.step7(out)
        out = self.step8(out)
        out = self.step9(out)
        out = self.step10(out)
        out = self.step11(out)
        out = torch.flatten(out, start_dim=1, end_dim=-1)
        out = self.to_critic_value(out)
        return out


class AtomGen(nn.Module):
    def __init__(self, z_l_dim, condition_n_channels, fac=1, sn: int = 0, device=None):
        super(AtomGen, self).__init__()
        specnorm = _sn_to_specnorm(sn)
        embed_condition_blocks = [
            specnorm(
                nn.Conv3d(
                    in_channels=condition_n_channels,
                    out_channels=_facify(128, fac),
                    kernel_size=5,
                    stride=1,
                    padding=2,
                )
            ),
            nn.GroupNorm(1, num_channels=_facify(128, fac)),
            nn.LeakyReLU(inplace=True),
            Residual3DConvBlock(
                _facify(128, fac), _facify(128, fac), kernel_size=3, stride=1, sn=sn
            ),
            Residual3DConvBlock(
                _facify(128, fac), _facify(128, fac), kernel_size=3, stride=1, sn=sn
            ),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool3d(2),
        ]
        self.embed_condition = nn.Sequential(*tuple(embed_condition_blocks)).to(
            device=device
        )
        self.embed_noise_and_label = EmbedNoise(z_l_dim, _facify(128, fac), sn=sn)

        self.conv_blocks = nn.Sequential(
            specnorm(
                nn.Conv3d(
                    2 * _facify(128, fac), _facify(128, fac), 3, stride=1, padding=1
                )
            ),
            nn.GroupNorm(1, num_channels=_facify(128, fac)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            specnorm(
                nn.Conv3d(_facify(128, fac), _facify(64, fac), 3, stride=1, padding=1)
            ),
            nn.GroupNorm(1, num_channels=_facify(64, fac)),
            nn.LeakyReLU(0.2, inplace=True),
            specnorm(nn.Conv3d(_facify(64, fac), 1, 3, stride=1, padding=1)),
            nn.Sigmoid(),
        )
        self.to(device=device)

    def forward(self, z, l, c):
        z_l = torch.cat((z, l), dim=1)
        embedded_z_l = self.embed_noise_and_label(z_l)
        embedded_c = self.embed_condition(c)
        out = torch.cat((embedded_z_l, embedded_c), dim=1)
        out = self.conv_blocks(out)
        return out



class AtomGen2(nn.Module):
    def __init__(
        self,
        z_dim,
        condition_n_channels,
        fac=1,
        sn: int = 0,
        device=None,
    ):
        super().__init__()
        specnorm = _sn_to_specnorm(sn)
        embed_condition_blocks = [
            specnorm(
                nn.Conv3d(
                    in_channels=condition_n_channels,
                    out_channels=_facify(128, fac),
                    kernel_size=5,
                    stride=1,
                    padding=2,
                )
            ),
            nn.GroupNorm(1, num_channels=_facify(128, fac)),
            nn.LeakyReLU(),
            Residual3DConvBlock(
                _facify(128, fac), _facify(128, fac), kernel_size=3, stride=1, sn=sn
            ),
            Residual3DConvBlock(
                _facify(128, fac), _facify(128, fac), kernel_size=3, stride=1, sn=sn
            ),
        ]
        self.embed_condition = nn.Sequential(*tuple(embed_condition_blocks)).to(
            device=device
        )
        self.downsample_cond = Residual3DConvBlock(
            _facify(128, fac),
            n_filters=_facify(128, fac),
            kernel_size=3,
            stride=2,
            sn=sn,
        )

        self.embed_noise_label = EmbedNoise(z_dim, _facify(128, fac), sn=sn)
        self.combined = GeneratorCombined1Block(
            _facify(256, fac), _facify(128, fac), sn=sn
        )

        to_image_blocks = [
            Residual3DConvBlock(
                _facify(256, fac), _facify(128, fac), 3, 1, trans=True, sn=sn
            ),
            Residual3DConvBlock(_facify(128, fac), _facify(64, fac), 3, 1, trans=True, sn=sn),
            specnorm(nn.Conv3d(_facify(64, fac), 1, kernel_size=1, stride=1)),
            nn.Sigmoid(),
        ]
        self.to_image = nn.Sequential(*tuple(to_image_blocks)).to(device=device)

    def forward(self, z, l, c):
        z_l = torch.cat((z, l), dim=1)

        embedded_c = self.embed_condition(c)
        down = self.downsample_cond(embedded_c)

        embedded_z_l = self.embed_noise_label(z_l)

        out = self.combined(embedded_z_l, down)

        out = torch.cat((out, embedded_c), dim=1)
        out = self.to_image(out)

        return out


class AtomGen_small(nn.Module):
    def __init__(
        self,
        z_dim,
        condition_n_channels,
        fac=1,
        sn: int = 0,
        device=None,
    ):
        super().__init__()
        specnorm = _sn_to_specnorm(sn)
        embed_condition_blocks = [
            specnorm(
                nn.Conv3d(
                    in_channels=condition_n_channels,
                    out_channels=_facify(128, fac),
                    kernel_size=5,
                    stride=1,
                    padding=2,
                )
            ),
            nn.GroupNorm(1, num_channels=_facify(128, fac)),
            nn.LeakyReLU(),
            Residual3DConvBlock(
                _facify(128, fac), _facify(128, fac), kernel_size=3, stride=1, sn=sn
            ),
        ]
        self.embed_condition = nn.Sequential(*tuple(embed_condition_blocks)).to(
            device=device
        )
        self.downsample_cond = Residual3DConvBlock(
            _facify(128, fac),
            n_filters=_facify(128, fac),
            kernel_size=3,
            stride=2,
            sn=sn,
        )

        self.embed_noise_label = EmbedNoise(z_dim, _facify(128, fac), sn=sn)
        self.combined = GeneratorCombined3Block(
            _facify(256, fac), _facify(128, fac), sn=sn
        )

        to_image_blocks = [
            Residual3DConvBlock(
                _facify(256, fac), _facify(64, fac), 3, 1, trans=True, sn=sn
            ),
            specnorm(nn.Conv3d(_facify(64, fac), 1, kernel_size=1, stride=1)),
            nn.Sigmoid(),
        ]
        self.to_image = nn.Sequential(*tuple(to_image_blocks)).to(device=device)

    def forward(self, z, l, c):
        z_l = torch.cat((z, l), dim=1)

        embedded_c = self.embed_condition(c)
        down = self.downsample_cond(embedded_c)

        embedded_z_l = self.embed_noise_label(z_l)

        out = self.combined(embedded_z_l, down)

        out = torch.cat((out, embedded_c), dim=1)
        out = self.to_image(out)

        return out

class AtomCrit_small(nn.Module):
    def __init__(
        self, in_channels, fac=1, sn: int = 0, device=None
    ):
        super().__init__()
        specnorm = _sn_to_specnorm(sn)
        self.step1 = specnorm(
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=_facify(128, fac),
                kernel_size=5,
                stride=1,
                padding=compute_same_padding(5, 1, 1),
            )
        )
        self.step2 = nn.LeakyReLU()

        self.step4 = Residual3DConvBlock(
            in_channels=_facify(128, fac),
            n_filters=_facify(256, fac),
            kernel_size=3,
            stride=2,
            sn=sn,
            device=device,
        )

        self.step6 = specnorm(
            nn.Conv3d(
                in_channels=_facify(256, fac),
                out_channels=_facify(256, fac),
                kernel_size=3,
                stride=1,
                padding=compute_same_padding(3, 1, 1),
            )
        )
        self.step7 = nn.GroupNorm(1, _facify(256, fac))
        self.step8 = nn.LeakyReLU()

        self.step9 = specnorm(
            nn.Conv3d(
                in_channels=_facify(256, fac),
                out_channels=_facify(128, fac),
                kernel_size=4,
                stride=1,
                padding=0,
            )
        )
        self.step10 = nn.GroupNorm(1, _facify(128, fac))
        self.step11 = nn.LeakyReLU()

        self.to_critic_value = specnorm(
            nn.Linear(
                in_features=_facify(128, fac), out_features=1
            )
        )

    def forward(self, inputs):
        out = self.step1(inputs)
        out = self.step2(out)
        #out = self.step3(out)
        out = self.step4(out)
        #out = self.step5(out)
        out = self.step6(out)
        out = self.step7(out)
        out = self.step8(out)
        out = self.step9(out)
        out = self.step10(out)
        out = self.step11(out)
        out = torch.flatten(out, start_dim=1, end_dim=-1)
        out = self.to_critic_value(out)
        return out



class AtomGen_small16(nn.Module):
    def __init__(
        self,
        z_dim,
        condition_n_channels,
        fac=1,
        sn: int = 0,
        device=None,
    ):
        super().__init__()
        specnorm = _sn_to_specnorm(sn)
        embed_condition_blocks = [
            specnorm(
                nn.Conv3d(
                    in_channels=condition_n_channels,
                    out_channels=_facify(64, fac),
                    kernel_size=5,
                    stride=1,
                    padding=2,
                )
            ),
            nn.GroupNorm(1, num_channels=_facify(64, fac)),
            nn.LeakyReLU(),
            Residual3DConvBlock(
                _facify(64, fac), _facify(64, fac), kernel_size=3, stride=1, sn=sn
            ),
        ]
        self.embed_condition = nn.Sequential(*tuple(embed_condition_blocks)).to(
            device=device
        )
        self.downsample_cond = Residual3DConvBlock(
            _facify(64, fac),
            n_filters=_facify(64, fac),
            kernel_size=3,
            stride=2,
            sn=sn,
        )

        self.embed_noise_label = EmbedNoise(z_dim, _facify(64, fac), sn=sn)
        self.combined1 = GeneratorCombined3Block(
            _facify(128, fac), _facify(64, fac), sn=sn
        )
        self.combined2 = GeneratorCombined3Block(
            _facify(128, fac), _facify(64, fac), sn=sn
        )

        to_image_blocks = [
            Residual3DConvBlock(
                _facify(128, fac), _facify(64, fac), 3, 1, trans=True, sn=sn
            ),
            specnorm(nn.Conv3d(_facify(64, fac), 1, kernel_size=1, stride=1)),
            nn.Sigmoid(),
        ]
        self.to_image = nn.Sequential(*tuple(to_image_blocks)).to(device=device)

    def forward(self, z, l, c):
        z_l = torch.cat((z, l), dim=1)

        embedded_c = self.embed_condition(c)
        down1 = self.downsample_cond(embedded_c)
        down2 = self.downsample_cond(down1)


        embedded_z_l = self.embed_noise_label(z_l)

        out = self.combined1(embedded_z_l, down2)
        out = self.combined2(out, down1)

        out = torch.cat((out, embedded_c), dim=1)
        out = self.to_image(out)

        return out

class AtomCrit_small16(nn.Module):
    def __init__(
        self, in_channels, fac=1, sn: int = 0, device=None
    ):
        super().__init__()
        specnorm = _sn_to_specnorm(sn)
        self.step1 = specnorm(
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=_facify(128, fac),
                kernel_size=5,
                stride=1,
                padding=compute_same_padding(5, 1, 1),
            )
        )
        self.step2 = nn.LeakyReLU()

        self.step4 = Residual3DConvBlock(
            in_channels=_facify(128, fac),
            n_filters=_facify(256, fac),
            kernel_size=3,
            stride=2,
            sn=sn,
            device=device,
        )

        self.step5 = Residual3DConvBlock(
            in_channels=_facify(256, fac),
            n_filters=_facify(256, fac),
            kernel_size=3,
            stride=2,
            sn=sn,
            device=device,
        )


        self.step6 = specnorm(
            nn.Conv3d(
                in_channels=_facify(256, fac),
                out_channels=_facify(256, fac),
                kernel_size=3,
                stride=1,
                padding=compute_same_padding(3, 1, 1),
            )
        )
        self.step7 = nn.GroupNorm(1, _facify(256, fac))
        self.step8 = nn.LeakyReLU()

        self.step9 = specnorm(
            nn.Conv3d(
                in_channels=_facify(256, fac),
                out_channels=_facify(128, fac),
                kernel_size=4,
                stride=1,
                padding=0,
            )
        )
        self.step10 = nn.GroupNorm(1, _facify(128, fac))
        self.step11 = nn.LeakyReLU()

        self.to_critic_value = specnorm(
            nn.Linear(
                in_features=_facify(128, fac), out_features=1
            )
        )

    def forward(self, inputs):
        out = self.step1(inputs)
        out = self.step2(out)
        #out = self.step3(out)
        out = self.step4(out)
        out = self.step5(out)
        #out = self.step5(out)
        out = self.step6(out)
        out = self.step7(out)
        out = self.step8(out)
        out = self.step9(out)
        out = self.step10(out)
        out = self.step11(out)
        out = torch.flatten(out, start_dim=1, end_dim=-1)
        out = self.to_critic_value(out)
        return out



class AtomCrit_dstr_small(nn.Module):
    def __init__(        self, in_channels_vox, in_channels_dstr, fac=1, sn: int = 0, device=None):
        super().__init__()
        specnorm = _sn_to_specnorm(sn)
        self.step1 = specnorm(
            nn.Conv3d(
                in_channels=in_channels_vox,
                out_channels=_facify(128, fac),
                kernel_size=5,
                stride=1,
                padding=compute_same_padding(5, 1, 1),
            )
        )
        self.step2 = nn.LeakyReLU()
        self.step4 = Residual3DConvBlock(
            in_channels=_facify(128, fac),
            n_filters=_facify(256, fac),
            kernel_size=3,
            stride=2,
            sn=sn,
            device=device,
        )
        self.step6 = specnorm(
            nn.Conv3d(
                in_channels=_facify(256, fac),
                out_channels=_facify(256, fac),
                kernel_size=3,
                stride=1,
                padding=compute_same_padding(3, 1, 1),
            )
        )
        self.step7 = nn.GroupNorm(1, _facify(256, fac))
        self.step8 = nn.LeakyReLU()
        self.step9 = specnorm(
            nn.Conv3d(
                in_channels=_facify(256, fac),
                out_channels=_facify(128, fac),
                kernel_size=4,
                stride=1,
                padding=0,
            )
        )
        self.step10 = nn.GroupNorm(1, _facify(128, fac))
        self.step11 = nn.LeakyReLU()
        self.step12 = specnorm(
            nn.Linear(
                in_features=in_channels_dstr, out_features=_facify(128, fac)
            )
        )
        self.step13 = nn.LeakyReLU()

        self.step14 = specnorm(
            nn.Linear(
                in_features=_facify(128, fac), out_features=_facify(128, fac)
            )
        )
        self.step15 = nn.LeakyReLU()

        self.step16 = specnorm(
            nn.Linear(
                in_features=_facify(256, fac), out_features=_facify(128, fac)
            )
        )
        self.step17 = nn.LeakyReLU()


        self.to_critic_value = specnorm(
            nn.Linear(
                in_features=_facify(128, fac), out_features=1
            )
        )
    def forward(self, input):
        voxel_input, dstr_input = input
        vox_out = self.step1(voxel_input)
        vox_out = self.step2(vox_out)
        vox_out = self.step4(vox_out)
        vox_out = self.step6(vox_out)
        vox_out = self.step7(vox_out)
        vox_out = self.step8(vox_out)
        vox_out = self.step9(vox_out)
        vox_out = self.step10(vox_out)
        vox_out = self.step11(vox_out)
        vox_out = torch.flatten(vox_out, start_dim=1, end_dim=-1)
        dstr_out = torch.flatten(dstr_input, start_dim=1, end_dim=-1)
        dstr_out = self.step12(dstr_out)
        dstr_out = self.step13(dstr_out)
        dstr_out = self.step14(dstr_out)
        dstr_out = self.step15(dstr_out)
        out = torch.cat([vox_out, dstr_out], 1)
        out = self.step16(out)
        out = self.step17(out)
        out = self.to_critic_value(out)
        return out



class AtomGen_small_drop(nn.Module):
    def __init__(
        self,
        z_dim,
        condition_n_channels,
        fac=1,
        sn: int = 0,
        device=None,
    ):
        super().__init__()
        specnorm = _sn_to_specnorm(sn)
        embed_condition_blocks = [
            specnorm(
                nn.Conv3d(
                    in_channels=condition_n_channels,
                    out_channels=_facify(128, fac),
                    kernel_size=5,
                    stride=1,
                    padding=2,
                )
            ),
            nn.Dropout3d(),
            nn.GroupNorm(1, num_channels=_facify(128, fac)),
            nn.LeakyReLU(),
            Residual3DConvBlock_drop(
                _facify(128, fac), _facify(128, fac), kernel_size=3, stride=1, sn=sn
            ),
        ]
        self.embed_condition = nn.Sequential(*tuple(embed_condition_blocks)).to(
            device=device
        )
        self.downsample_cond = Residual3DConvBlock_drop(
            _facify(128, fac),
            n_filters=_facify(128, fac),
            kernel_size=3,
            stride=2,
            sn=sn,
        )

        self.embed_noise_label = EmbedNoise(z_dim, _facify(128, fac), sn=sn)
        self.combined = GeneratorCombined3Block(
            _facify(256, fac), _facify(128, fac), sn=sn
        )

        to_image_blocks = [
            Residual3DConvBlock(
                _facify(256, fac), _facify(64, fac), 3, 1, trans=True, sn=sn
            ),
            specnorm(nn.Conv3d(_facify(64, fac), 1, kernel_size=1, stride=1)),
            nn.Sigmoid(),
        ]
        self.to_image = nn.Sequential(*tuple(to_image_blocks)).to(device=device)

        self.dropout = nn.Dropout3d()

    def forward(self, z, l, c):
        z_l = torch.cat((z, l), dim=1)

        embedded_c = self.embed_condition(c)
        down = self.downsample_cond(embedded_c)

        embedded_z_l = self.embed_noise_label(z_l)

        out = self.combined(embedded_z_l, down)

        out = torch.cat((out, embedded_c), dim=1)
        out = self.to_image(out)

        return out

class AtomGen_small16_drop(nn.Module):
    def __init__(
        self,
        z_dim,
        condition_n_channels,
        fac=1,
        sn: int = 0,
        device=None,
    ):
        super().__init__()
        specnorm = _sn_to_specnorm(sn)
        embed_condition_blocks = [
            specnorm(
                nn.Conv3d(
                    in_channels=condition_n_channels,
                    out_channels=_facify(64, fac),
                    kernel_size=5,
                    stride=1,
                    padding=2,
                )
            ),
            nn.Dropout3d(),
            nn.GroupNorm(1, num_channels=_facify(64, fac)),
            nn.LeakyReLU(),
            Residual3DConvBlock_drop(
                _facify(64, fac), _facify(64, fac), kernel_size=3, stride=1, sn=sn
            ),
        ]
        self.embed_condition = nn.Sequential(*tuple(embed_condition_blocks)).to(
            device=device
        )
        self.downsample_cond = Residual3DConvBlock_drop(
            _facify(64, fac),
            n_filters=_facify(64, fac),
            kernel_size=3,
            stride=2,
            sn=sn,
        )

        self.embed_noise_label = EmbedNoise(z_dim, _facify(64, fac), sn=sn)
        self.combined1 = GeneratorCombined3Block(
            _facify(128, fac), _facify(64, fac), sn=sn
        )
        self.combined2 = GeneratorCombined3Block(
            _facify(128, fac), _facify(64, fac), sn=sn
        )

        to_image_blocks = [
            Residual3DConvBlock(
                _facify(128, fac), _facify(64, fac), 3, 1, trans=True, sn=sn
            ),
            specnorm(nn.Conv3d(_facify(64, fac), 1, kernel_size=1, stride=1)),
            nn.Sigmoid(),
        ]
        self.to_image = nn.Sequential(*tuple(to_image_blocks)).to(device=device)

    def forward(self, z, l, c):
        z_l = torch.cat((z, l), dim=1)

        embedded_c = self.embed_condition(c)
        down1 = self.downsample_cond(embedded_c)
        down2 = self.downsample_cond(down1)


        embedded_z_l = self.embed_noise_label(z_l)

        out = self.combined1(embedded_z_l, down2)
        out = self.combined2(out, down1)

        out = torch.cat((out, embedded_c), dim=1)
        out = self.to_image(out)

        return out