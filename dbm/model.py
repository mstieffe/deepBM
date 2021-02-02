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


class AtomGen_tiny(nn.Module):
    def __init__(
        self,
        z_dim,
        in_channels,
        start_channels,
        fac=1,
        sn: int = 0,
        device=None,
    ):
        super().__init__()
        specnorm = _sn_to_specnorm(sn)
        embed_condition_blocks = [
            specnorm(
                nn.Conv3d(
                    in_channels=in_channels,
                    out_channels=_facify(start_channels, fac),
                    kernel_size=5,
                    stride=1,
                    padding=2,
                )
            ),
            nn.GroupNorm(1, num_channels=_facify(start_channels, fac)),
            nn.LeakyReLU(),
            specnorm(
                nn.Conv3d(
                    in_channels=start_channels,
                    out_channels=_facify(start_channels, fac),
                    kernel_size=3,
                    stride=1,
                    padding=compute_same_padding(3, 1, 1)
                )
            ),
            nn.GroupNorm(1, num_channels=_facify(start_channels, fac)),
            nn.LeakyReLU(),

        ]
        self.embed_condition = nn.Sequential(*tuple(embed_condition_blocks)).to(device=device)

        downsample_cond_block = [
            specnorm(
                nn.Conv3d(
                    in_channels=start_channels,
                    out_channels=_facify(start_channels*2, fac),
                    kernel_size=3,
                    stride=2,
                    padding=compute_same_padding(3, 2, 1)
                )
            ),
            nn.GroupNorm(1, num_channels=_facify(start_channels*2, fac)),
            nn.LeakyReLU(),

        ]
        self.downsample_cond = nn.Sequential(*tuple(downsample_cond_block)).to(device=device)

        self.embed_noise_label = EmbedNoise(z_dim, _facify(start_channels*2, fac), sn=sn)

        combined_block = [
            specnorm(
                nn.Conv3d(
                    in_channels=start_channels*4,
                    out_channels=_facify(start_channels*2, fac),
                    kernel_size=3,
                    stride=1,
                    padding=compute_same_padding(3, 1, 1),
                )
            ),
            nn.GroupNorm(1, num_channels=_facify(start_channels*2, fac)),
            nn.LeakyReLU(),

        ]
        self.combined = nn.Sequential(*tuple(combined_block)).to(device=device)

        deconv_block = [
            specnorm(
                nn.Conv3d(
                    in_channels=start_channels*2,
                    out_channels=_facify(start_channels, fac),
                    kernel_size=3,
                    stride=1,
                    padding=compute_same_padding(3, 1, 1),
                )
            ),
            nn.GroupNorm(1, num_channels=_facify(start_channels, fac)),
            nn.LeakyReLU(),

        ]
        self.deconv = nn.Sequential(*tuple(deconv_block)).to(device=device)

        to_image_blocks = [
            specnorm(
                nn.Conv3d(
                    in_channels=start_channels*2,
                    out_channels=_facify(start_channels, fac),
                    kernel_size=3,
                    stride=1,
                    padding=compute_same_padding(3, 1, 1),
                )
            ),
            nn.GroupNorm(1, num_channels=_facify(start_channels, fac)),
            nn.LeakyReLU(),
            specnorm(
                nn.Conv3d(
                    in_channels=start_channels,
                    out_channels=_facify(start_channels / 2, fac),
                    kernel_size=3,
                    stride=1,
                    padding=compute_same_padding(3, 1, 1),
                )
            ),
            nn.GroupNorm(1, num_channels=_facify(start_channels/2, fac)),
            nn.LeakyReLU(),
            specnorm(nn.Conv3d(_facify(start_channels/2, fac), 1, kernel_size=1, stride=1)),
            nn.Sigmoid(),
        ]
        self.to_image = nn.Sequential(*tuple(to_image_blocks)).to(device=device)

    def forward(self, z, l, c):
        z_l = torch.cat((z, l), dim=1)
        embedded_c = self.embed_condition(c)
        down = self.downsample_cond(embedded_c)
        embedded_z_l = self.embed_noise_label(z_l)
        out = torch.cat((embedded_z_l, down), dim=1)
        out = self.combined(out)
        out = out.repeat(1, 1, 2, 2, 2)
        out = self.deconv(out)
        out = torch.cat((out, embedded_c), dim=1)
        out = self.to_image(out)

        return out


class AtomGen_tiny16(nn.Module):
    def __init__(
            self,
            z_dim,
            in_channels,
            start_channels,
            fac=1,
            sn: int = 0,
            device=None,
    ):
        super().__init__()
        specnorm = _sn_to_specnorm(sn)
        embed_condition_blocks = [
            specnorm(
                nn.Conv3d(
                    in_channels=in_channels,
                    out_channels=_facify(start_channels, fac),
                    kernel_size=5,
                    stride=1,
                    padding=2,
                )
            ),
            nn.GroupNorm(1, num_channels=_facify(start_channels, fac)),
            nn.LeakyReLU(),
            specnorm(
                nn.Conv3d(
                    in_channels=start_channels,
                    out_channels=_facify(start_channels, fac),
                    kernel_size=3,
                    stride=1,
                    padding=compute_same_padding(3, 1, 1),
                )
            ),
            nn.GroupNorm(1, num_channels=_facify(start_channels, fac)),
            nn.LeakyReLU(),

        ]
        self.embed_condition = nn.Sequential(*tuple(embed_condition_blocks)).to(device=device)

        downsample_cond_block = [
            specnorm(
                nn.Conv3d(
                    in_channels=start_channels,
                    out_channels=_facify(start_channels * 2, fac),
                    kernel_size=3,
                    stride=2,
                    padding=compute_same_padding(3, 2, 1),
                )
            ),
            nn.GroupNorm(1, num_channels=_facify(start_channels * 2, fac)),
            nn.LeakyReLU(),

        ]
        self.downsample_cond = nn.Sequential(*tuple(downsample_cond_block)).to(device=device)

        downsample_cond_block2 = [
            specnorm(
                nn.Conv3d(
                    in_channels=start_channels * 2,
                    out_channels=_facify(start_channels * 2, fac),
                    kernel_size=3,
                    stride=2,
                    padding=compute_same_padding(3, 2, 1),
                )
            ),
            nn.GroupNorm(1, num_channels=_facify(start_channels * 2, fac)),
            nn.LeakyReLU(),

        ]
        self.downsample_cond2 = nn.Sequential(*tuple(downsample_cond_block2)).to(device=device)

        self.embed_noise_label = EmbedNoise(z_dim, _facify(start_channels * 2, fac), sn=sn)

        combined_block = [
            specnorm(
                nn.Conv3d(
                    in_channels=start_channels * 4,
                    out_channels=_facify(start_channels * 2, fac),
                    kernel_size=3,
                    stride=1,
                    padding=compute_same_padding(3, 1, 1),
                )
            ),
            nn.GroupNorm(1, num_channels=_facify(start_channels * 2, fac)),
            nn.LeakyReLU(),

        ]
        self.combined = nn.Sequential(*tuple(combined_block)).to(device=device)

        deconv_block = [
            specnorm(
                nn.Conv3d(
                    in_channels=start_channels * 2,
                    out_channels=_facify(start_channels * 2, fac),
                    kernel_size=3,
                    stride=1,
                    padding=compute_same_padding(3, 1, 1),
                )
            ),
            nn.GroupNorm(1, num_channels=_facify(start_channels * 2, fac)),
            nn.LeakyReLU(),
            specnorm(
                nn.Conv3d(
                    in_channels=start_channels * 2,
                    out_channels=_facify(start_channels, fac),
                    kernel_size=3,
                    stride=1,
                    padding=compute_same_padding(3, 1, 1),
                )
            ),
            nn.GroupNorm(1, num_channels=_facify(start_channels , fac)),
            nn.LeakyReLU(),

        ]
        self.deconv = nn.Sequential(*tuple(deconv_block)).to(device=device)

        deconv_block2 = [

            specnorm(
                nn.Conv3d(
                    in_channels=start_channels ,
                    out_channels=_facify(start_channels, fac),
                    kernel_size=3,
                    stride=1,
                    padding=compute_same_padding(3, 1, 1),
                )
            ),
            nn.GroupNorm(1, num_channels=_facify(start_channels, fac)),
            nn.LeakyReLU(),

        ]
        self.deconv2 = nn.Sequential(*tuple(deconv_block2)).to(device=device)

        to_image_blocks = [
            specnorm(
                nn.Conv3d(
                    in_channels=start_channels*2 ,
                    out_channels=_facify(start_channels / 2, fac),
                    kernel_size=3,
                    stride=1,
                    padding=compute_same_padding(3, 1, 1),
                )
            ),
            nn.GroupNorm(1, num_channels=_facify(start_channels / 2, fac)),
            nn.LeakyReLU(),
            specnorm(
                nn.Conv3d(
                    in_channels=_facify(start_channels / 2, fac),
                    out_channels=_facify(start_channels / 4, fac),
                    kernel_size=3,
                    stride=1,
                    padding=compute_same_padding(3, 1, 1),
                )
            ),
            nn.GroupNorm(1, num_channels=_facify(start_channels / 4, fac)),
            nn.LeakyReLU(),
            specnorm(nn.Conv3d(_facify(start_channels / 4, fac), 1, kernel_size=1, stride=1)),
            nn.Sigmoid(),
        ]
        self.to_image = nn.Sequential(*tuple(to_image_blocks)).to(device=device)

    def forward(self, z, l, c):
        z_l = torch.cat((z, l), dim=1)
        embedded_c = self.embed_condition(c)
        down = self.downsample_cond(embedded_c)
        down2 = self.downsample_cond2(down)
        embedded_z_l = self.embed_noise_label(z_l)
        out = torch.cat((embedded_z_l, down2), dim=1)
        out = self.combined(out)
        out = out.repeat(1, 1, 2, 2, 2)
        out = self.deconv(out)
        out = out.repeat(1, 1, 2, 2, 2)
        out = self.deconv2(out)
        out = torch.cat((out, embedded_c), dim=1)

        out = self.to_image(out)

        return out



class AtomGen(nn.Module):
    def __init__(
        self,
        z_dim,
        in_channels,
        start_channels,
        fac=1,
        sn: int = 0,
        device=None,
    ):
        super().__init__()
        specnorm = _sn_to_specnorm(sn)
        embed_condition_blocks = [
            specnorm(
                nn.Conv3d(
                    in_channels=in_channels,
                    out_channels=_facify(start_channels, fac),
                    kernel_size=5,
                    stride=1,
                    padding=2,
                )
            ),
            nn.GroupNorm(1, num_channels=_facify(start_channels, fac)),
            nn.LeakyReLU(),
            Residual3DConvBlock(
                _facify(start_channels, fac), _facify(start_channels, fac), kernel_size=3, stride=1, sn=sn
            ),
        ]
        self.embed_condition = nn.Sequential(*tuple(embed_condition_blocks)).to(
            device=device
        )
        self.downsample_cond = Residual3DConvBlock(
            _facify(start_channels, fac),
            n_filters=_facify(start_channels, fac),
            kernel_size=3,
            stride=2,
            sn=sn,
        )

        self.embed_noise_label = EmbedNoise(z_dim, _facify(start_channels, fac), sn=sn)
        self.combined = GeneratorCombined3Block(
            _facify(start_channels*2, fac), _facify(start_channels, fac), sn=sn
        )

        to_image_blocks = [
            Residual3DConvBlock(
                _facify(start_channels*2, fac), _facify(start_channels/2, fac), 3, 1, trans=True, sn=sn
            ),
            specnorm(nn.Conv3d(_facify(start_channels/2, fac), 1, kernel_size=1, stride=1)),
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


class AtomGen16(nn.Module):
    def __init__(
        self,
        z_dim,
        in_channels,
        start_channels,
        fac=1,
        sn: int = 0,
        device=None,
    ):
        super().__init__()
        specnorm = _sn_to_specnorm(sn)
        embed_condition_blocks = [
            specnorm(
                nn.Conv3d(
                    in_channels=in_channels,
                    out_channels=_facify(start_channels, fac),
                    kernel_size=5,
                    stride=1,
                    padding=2,
                )
            ),
            nn.GroupNorm(1, num_channels=_facify(start_channels, fac)),
            nn.LeakyReLU(),
            Residual3DConvBlock(
                _facify(start_channels, fac), _facify(start_channels, fac), kernel_size=3, stride=1, sn=sn
            ),
        ]
        self.embed_condition = nn.Sequential(*tuple(embed_condition_blocks)).to(
            device=device
        )
        self.downsample_cond = Residual3DConvBlock(
            _facify(start_channels, fac),
            n_filters=_facify(start_channels, fac),
            kernel_size=3,
            stride=2,
            sn=sn,
        )

        self.embed_noise_label = EmbedNoise(z_dim, _facify(start_channels, fac), sn=sn)
        self.combined1 = GeneratorCombined3Block(
            _facify(start_channels*2, fac), _facify(start_channels, fac), sn=sn
        )
        self.combined2 = GeneratorCombined3Block(
            _facify(start_channels*2, fac), _facify(start_channels, fac), sn=sn
        )

        to_image_blocks = [
            Residual3DConvBlock(
                _facify(start_channels*2, fac), _facify(start_channels, fac), 3, 1, trans=True, sn=sn
            ),
            specnorm(nn.Conv3d(_facify(start_channels, fac), 1, kernel_size=1, stride=1)),
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


class AtomGen_big(nn.Module):
    def __init__(
        self,
        z_dim,
        in_channels,
        start_channels,
        fac=1,
        sn: int = 0,
        device=None,
    ):
        super().__init__()
        specnorm = _sn_to_specnorm(sn)
        embed_condition_blocks = [
            specnorm(
                nn.Conv3d(
                    in_channels=in_channels,
                    out_channels=_facify(start_channels, fac),
                    kernel_size=5,
                    stride=1,
                    padding=2,
                )
            ),
            nn.GroupNorm(1, num_channels=_facify(start_channels, fac)),
            nn.LeakyReLU(),
            Residual3DConvBlock(
                _facify(start_channels, fac), _facify(start_channels, fac), kernel_size=3, stride=1, sn=sn
            ),
            Residual3DConvBlock(
                _facify(start_channels, fac), _facify(start_channels, fac), kernel_size=3, stride=1, sn=sn
            ),
        ]
        self.embed_condition = nn.Sequential(*tuple(embed_condition_blocks)).to(
            device=device
        )
        self.downsample_cond = Residual3DConvBlock(
            _facify(start_channels, fac),
            n_filters=_facify(start_channels*2, fac),
            kernel_size=3,
            stride=2,
            trans=True,
            sn=sn,
        )

        self.embed_noise_label = EmbedNoise(z_dim, _facify(start_channels*2, fac), sn=sn)
        self.combined = GeneratorCombined1Block(
            _facify(start_channels*4, fac), _facify(start_channels*2, fac), sn=sn
        )

        to_image_blocks = [
            Residual3DConvBlock(
                _facify(start_channels*2, fac), _facify(start_channels, fac), 3, 1, trans=True, sn=sn
            ),
            Residual3DConvBlock(_facify(start_channels, fac), _facify(start_channels/2, fac), 3, 1, trans=True, sn=sn),
            specnorm(nn.Conv3d(_facify(start_channels/2, fac), 1, kernel_size=1, stride=1)),
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



class AtomCrit_tiny(nn.Module):
    def __init__(
        self, in_channels, start_channels, fac=1, sn: int = 0, device=None
    ):
        super().__init__()
        specnorm = _sn_to_specnorm(sn)
        self.step1 = specnorm(
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=_facify(start_channels, fac),
                kernel_size=5,
                stride=1,
                padding=compute_same_padding(5, 1, 1),
            )
        )
        self.step2 = nn.LeakyReLU()

        self.step3 = specnorm(
            nn.Conv3d(
                in_channels=_facify(start_channels, fac),
                out_channels=_facify(start_channels*2, fac),
                kernel_size=3,
                stride=2,
                padding=compute_same_padding(3, 1, 1),
            )
        )
        self.step4 = nn.GroupNorm(1, _facify(start_channels*2, fac))
        self.step5 = nn.LeakyReLU()

        self.step6 = specnorm(
            nn.Conv3d(
                in_channels=_facify(start_channels*2, fac),
                out_channels=_facify(start_channels*2, fac),
                kernel_size=3,
                stride=1,
                padding=compute_same_padding(3, 1, 1),
            )
        )
        self.step7 = nn.GroupNorm(1, _facify(start_channels*2, fac))
        self.step8 = nn.LeakyReLU()

        self.step9 = specnorm(
            nn.Conv3d(
                in_channels=_facify(start_channels*2, fac),
                out_channels=_facify(start_channels, fac),
                kernel_size=4,
                stride=1,
                padding=0,
            )
        )
        self.step10 = nn.GroupNorm(1, _facify(start_channels, fac))
        self.step11 = nn.LeakyReLU()

        self.to_critic_value = specnorm(
            nn.Linear(
                in_features=_facify(start_channels, fac), out_features=1
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

class AtomCrit(nn.Module):
    def __init__(
        self, in_channels, start_channels, fac=1, sn: int = 0, device=None
    ):
        super().__init__()
        specnorm = _sn_to_specnorm(sn)
        self.step1 = specnorm(
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=_facify(start_channels, fac),
                kernel_size=5,
                stride=1,
                padding=compute_same_padding(5, 1, 1),
            )
        )
        self.step2 = nn.LeakyReLU()

        self.step4 = Residual3DConvBlock(
            in_channels=_facify(start_channels, fac),
            n_filters=_facify(start_channels*2, fac),
            kernel_size=3,
            stride=2,
            sn=sn,
            device=device,
        )

        self.step6 = specnorm(
            nn.Conv3d(
                in_channels=_facify(start_channels*2, fac),
                out_channels=_facify(start_channels*2, fac),
                kernel_size=3,
                stride=1,
                padding=compute_same_padding(3, 1, 1),
            )
        )
        self.step7 = nn.GroupNorm(1, _facify(start_channels*2, fac))
        self.step8 = nn.LeakyReLU()

        self.step9 = specnorm(
            nn.Conv3d(
                in_channels=_facify(start_channels*2, fac),
                out_channels=_facify(start_channels, fac),
                kernel_size=4,
                stride=1,
                padding=0,
            )
        )
        self.step10 = nn.GroupNorm(1, _facify(start_channels, fac))
        self.step11 = nn.LeakyReLU()

        self.to_critic_value = specnorm(
            nn.Linear(
                in_features=_facify(start_channels, fac), out_features=1
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


class AtomCrit_big(nn.Module):
    def __init__(
        self, in_channels, start_channels, fac=1, sn: int = 0, device=None
    ):
        super().__init__()
        specnorm = _sn_to_specnorm(sn)
        self.step1 = specnorm(
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=_facify(start_channels, fac),
                kernel_size=5,
                stride=1,
                padding=compute_same_padding(5, 1, 1),
            )
        )
        self.step2 = nn.LeakyReLU()

        self.step3 = Residual3DConvBlock(
            in_channels=_facify(start_channels, fac),
            n_filters=_facify(start_channels, fac),
            kernel_size=3,
            stride=1,
            sn=sn,
            device=device,
        )
        self.step4 = Residual3DConvBlock(
            in_channels=_facify(start_channels, fac),
            n_filters=_facify(start_channels*2, fac),
            kernel_size=3,
            stride=2,
            sn=sn,
            device=device,
        )
        self.step5 = Residual3DConvBlock(
            in_channels=_facify(start_channels*2, fac),
            n_filters=_facify(start_channels*2, fac),
            kernel_size=3,
            stride=1,
            sn=sn,
            device=device,
        )

        self.step6 = specnorm(
            nn.Conv3d(
                in_channels=_facify(start_channels*2, fac),
                out_channels=_facify(start_channels*2, fac),
                kernel_size=3,
                stride=1,
                padding=compute_same_padding(3, 1, 1),
            )
        )
        self.step7 = nn.GroupNorm(1, _facify(start_channels*2, fac))
        self.step8 = nn.LeakyReLU()

        self.step9 = specnorm(
            nn.Conv3d(
                in_channels=_facify(start_channels*2, fac),
                out_channels=_facify(start_channels, fac),
                kernel_size=4,
                stride=1,
                padding=0,
            )
        )
        self.step10 = nn.GroupNorm(1, _facify(start_channels, fac))
        self.step11 = nn.LeakyReLU()

        self.to_critic_value = specnorm(
            nn.Linear(
                in_features=_facify(start_channels, fac), out_features=1
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


class AtomCrit_tiny16(nn.Module):
    def __init__(
        self, in_channels, start_channels, fac=1, sn: int = 0, device=None
    ):
        super().__init__()
        specnorm = _sn_to_specnorm(sn)
        self.step1 = specnorm(
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=_facify(start_channels, fac),
                kernel_size=5,
                stride=1,
                padding=compute_same_padding(5, 1, 1),
            )
        )
        self.step2 = nn.LeakyReLU()

        self.step3 = specnorm(
            nn.Conv3d(
                in_channels=_facify(start_channels, fac),
                out_channels=_facify(start_channels*2, fac),
                kernel_size=3,
                stride=2,
                padding=compute_same_padding(3, 1, 1),
            )
        )
        self.step4 = nn.GroupNorm(1, _facify(start_channels*2, fac))
        self.step5 = nn.LeakyReLU()

        self.step6 = specnorm(
            nn.Conv3d(
                in_channels=_facify(start_channels*2, fac),
                out_channels=_facify(start_channels*2, fac),
                kernel_size=3,
                stride=1,
                padding=compute_same_padding(3, 1, 1),
            )
        )
        self.step7 = nn.GroupNorm(1, _facify(start_channels*2, fac))
        self.step8 = nn.LeakyReLU()

        self.step9 = specnorm(
            nn.Conv3d(
                in_channels=_facify(start_channels*2, fac),
                out_channels=_facify(start_channels*4, fac),
                kernel_size=3,
                stride=2,
                padding=compute_same_padding(3, 1, 1),
            )
        )
        self.step10 = nn.GroupNorm(1, _facify(start_channels*4, fac))
        self.step11 = nn.LeakyReLU()

        self.step12 = specnorm(
            nn.Conv3d(
                in_channels=_facify(start_channels*4, fac),
                out_channels=_facify(start_channels, fac),
                kernel_size=4,
                stride=1,
                padding=0,
            )
        )
        self.step13 = nn.GroupNorm(1, _facify(start_channels, fac))
        self.step14 = nn.LeakyReLU()

        self.to_critic_value = specnorm(
            nn.Linear(
                in_features=_facify(start_channels, fac), out_features=1
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
        out = self.step12(out)
        out = self.step13(out)
        out = self.step14(out)
        out = torch.flatten(out, start_dim=1, end_dim=-1)
        out = self.to_critic_value(out)
        return out


class AtomCrit16(nn.Module):
    def __init__(
        self, in_channels, start_channels, fac=1, sn: int = 0, device=None
    ):
        super().__init__()
        specnorm = _sn_to_specnorm(sn)
        self.step1 = specnorm(
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=_facify(start_channels, fac),
                kernel_size=5,
                stride=1,
                padding=compute_same_padding(5, 1, 1),
            )
        )
        self.step2 = nn.LeakyReLU()

        self.step4 = Residual3DConvBlock(
            in_channels=_facify(start_channels, fac),
            n_filters=_facify(start_channels*2, fac),
            kernel_size=3,
            stride=2,
            sn=sn,
            device=device,
        )

        self.step5 = Residual3DConvBlock(
            in_channels=_facify(start_channels*2, fac),
            n_filters=_facify(start_channels*4, fac),
            kernel_size=3,
            stride=2,
            sn=sn,
            device=device,
        )


        self.step6 = specnorm(
            nn.Conv3d(
                in_channels=_facify(start_channels*4, fac),
                out_channels=_facify(start_channels*4, fac),
                kernel_size=3,
                stride=1,
                padding=compute_same_padding(3, 1, 1),
            )
        )
        self.step7 = nn.GroupNorm(1, _facify(start_channels*4, fac))
        self.step8 = nn.LeakyReLU()

        self.step9 = specnorm(
            nn.Conv3d(
                in_channels=_facify(start_channels*4, fac),
                out_channels=_facify(start_channels, fac),
                kernel_size=4,
                stride=1,
                padding=0,
            )
        )
        self.step10 = nn.GroupNorm(1, _facify(start_channels, fac))
        self.step11 = nn.LeakyReLU()

        self.to_critic_value = specnorm(
            nn.Linear(
                in_features=_facify(start_channels, fac), out_features=1
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
