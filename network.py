import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.backends.opt_einsum import strategy
from torch.nn import init
from torchvision.models.resnet import resnet18, resnet101, resnet34
from einops.layers.torch import Rearrange

# ablation
class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result = self.maxpool(x)
        avg_result = self.avgpool(x)
        max_out = self.se(max_result)
        avg_out = self.se(avg_result)
        output = self.sigmoid(max_out + avg_out)
        return output

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], 1)
        output = self.conv(result)
        output = self.sigmoid(output)
        return output

class cbam_block(nn.Module):
    def __init__(self, channel, ratio=8, kernel_size=7):
        super(cbam_block, self).__init__()
        self.channelattention = ChannelAttention(channel, 8)
        self.spatialattention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        x = x * self.channelattention(x)
        x = x * self.spatialattention(x)
        return x


class SEAttention11(nn.Module):
    def __init__(self,in_channel,reduction=16):
        super(SEAttention11, self).__init__()
        self.pool=nn.AdaptiveAvgPool2d(output_size=1)
        self.fc=nn.Sequential(
            nn.Linear(in_features=in_channel,out_features=in_channel//reduction,bias=False),
            nn.ReLU(),
            nn.Linear(in_features=in_channel//reduction,out_features=in_channel,bias=False),
            nn.Sigmoid()
        )
        self.soft_max = nn.Softmax(dim=0)

    def forward(self,x):
        out=self.pool(x)
        out=self.fc(out.view(out.size(0),-1))
        out=out.view(x.size(0),x.size(1),1,1)

        return out


class SEAttention(nn.Module):
    def __init__(self, in_channel, ratio):
        super(SEAttention, self).__init__()
        hide_channel = in_channel // ratio
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(in_channel, hide_channel, kernel_size=1, bias=False)
        self.softmax = nn.Softmax(2)
        self.A0 = torch.eye(hide_channel).to('cuda')
        self.A2 = nn.Parameter(torch.FloatTensor(torch.zeros((hide_channel, hide_channel))), requires_grad=True)
        init.constant_(self.A2, 1e-6)
        self.conv2 = nn.Conv1d(1, 1, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(1, 1, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(hide_channel, in_channel, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv1(y)
        B, C, _, _ = y.size()
        y = y.flatten(2).transpose(1, 2)
        A1 = self.softmax(self.conv2(y))
        A1 = A1.expand(B, C, C)
        A = (self.A0 * A1)
        y = torch.matmul(y, A)
        y = self.relu(self.conv3(y))
        y = y.transpose(1, 2).view(-1, C, 1, 1)
        y = self.sigmoid(self.conv4(y))

        return x * y

class HAGM(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(HAGM, self).__init__()

        self.fc_h = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.bn_h = nn.BatchNorm2d(in_planes // ratio)
        self.relu_h = nn.ReLU()
        self.conv_h_sptial = nn.Conv2d(2 * (in_planes // ratio), in_planes // ratio, 7, padding=3, bias=False)

        self.fc_w = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.bn_w = nn.BatchNorm2d(in_planes // ratio)
        self.relu_w = nn.ReLU()
        self.conv_w_sptial = nn.Conv2d(2 * (in_planes // ratio), in_planes // ratio, 7, padding=3, bias=False)

        self.fc_general = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

        self.se = AGAB(in_planes)
        self.ca = ChannelAttention(in_planes, ratio)
        self.pa = PixelAttention(in_planes)
        self.sigmoid = nn.Sigmoid()

        self.conv_3 = nn.Conv2d(2 * (in_planes // ratio), in_planes // ratio,kernel_size=3,stride=1,padding=(3-1)//2,bias=False)
        self.conv_5 = nn.Conv2d(2 * (in_planes // ratio), in_planes // ratio,kernel_size=5,stride=1,padding=(5-1)//2,bias=False)
        self.conv_7 = nn.Conv2d(2 * (in_planes // ratio), in_planes // ratio,kernel_size=7,stride=1,padding=(7-1)//2,bias=False)
        self.conv__3 = nn.Conv2d(192, 64,kernel_size=1,stride=1,padding=0,bias=False)

    def forward(self, x): # torch.Size([64, 512, 7, 7])
        cattn = self.se(x)
        # cattn = self.ca(x)  # torch.Size([64, 512, 1, 1])
        # cattn = self.dwattn(x)
        _, _, h, w = x.size()

        x_h_avg = torch.mean(x, dim=3, keepdim=True)
        x_h_max, _ = torch.max(x, dim=3, keepdim=True)
        x_h_max = x_h_max

        x_w_avg = torch.mean(x, dim=2, keepdim=True)
        x_w_max, _ = torch.max(x, dim=2, keepdim=True)

        x_h_avg = self.relu_h(self.bn_h(self.fc_h(x_h_avg)))
        x_h_max = self.relu_h(self.bn_h(self.fc_h(x_h_max)))

        x_w_avg = self.relu_w(self.bn_w(self.fc_w(x_w_avg)))
        x_w_max = self.relu_w(self.bn_w(self.fc_w(x_w_max)))

        x_h_cat_sp = torch.cat([x_h_avg, x_h_max], dim=1)
        s3 = self.conv_3(x_h_cat_sp)
        s5 = self.conv_5(x_h_cat_sp)
        s7 = self.conv_7(x_h_cat_sp)
        s = torch.cat([s3,s5,s7],dim=1)
        x_w_cat_sp = torch.cat([x_w_avg, x_w_max], dim=1)
        s30 = self.conv_3(x_w_cat_sp)
        s50 = self.conv_5(x_w_cat_sp)
        s70 = self.conv_7(x_w_cat_sp)
        s0 = torch.cat([s30,s50,s70],dim=1)
        x_h_w = s * s0
        x_h_w = self.conv__3(x_h_w)
        # x_h_cat_sp = self.conv_h_sptial(torch.cat([x_h_avg, x_h_max], dim=1))
        # x_w_cat_sp = self.conv_w_sptial(torch.cat([x_w_avg, x_w_max], dim=1))
        # x_h_w = x_h_cat_sp * x_w_cat_sp   # torch.Size([64, 64, 7, 7])

        x_general = self.fc_general(x_h_w) # torch.Size([64, 512, 7, 7])
        x_general = self.sigmoid(x_general) * x

        x_general = x_general + cattn
        x2 = self.sigmoid(self.pa(x, x_general))
        # x_general = x_general + x2
        # res = x * self.sigmoid(x_general)
        return x * x2
        # return x2

        # return res

class DWConv_BN_RELU(nn.Module):
    def __init__(self, channels, kernel_size=3, relu=True, stride=1, padding=1):
        super(DWConv_BN_RELU, self).__init__()

        self.dwconv = nn.Conv2d(channels, channels, groups=channels, stride=stride,
                                kernel_size=kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.relu = relu

    def forward(self, x):
        x = self.dwconv(x)
        x = self.bn(x)
        if self.relu:
            x = F.relu(x)
        return x


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class AGAB(nn.Module):
    def __init__(self, channel, reduction=16):
        super(AGAB, self).__init__()
        self.conv = nn.Conv2d(channel, 1, 1, bias=False)
        self.softmax = nn.Softmax(dim=2)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.PReLU(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):  # torch.Size([128, 512, 7, 7])
        b, c, h, w = x.size()
        input_x = x  # torch.Size([128, 512, 7, 7])
        input_x = input_x.view(b, c, h*w).unsqueeze(1)  # torch.Size([128, 1, 512, 49])
        mask = self.conv(x)
        mask = self.conv(x).view(b, 1, h*w)  # torch.Size([128, 1, 49])
        mask = self.softmax(mask).unsqueeze(-1)  # torch.Size([128, 1, 49, 1])
        y = torch.matmul(input_x, mask)  # torch.Size([128, 1, 512, 1])
        y = torch.matmul(input_x, mask).view(b, c)  # torch.Size([128, 512])

        y = self.fc(y).view(b, c, 1, 1)  # torch.Size([128, 512, 1, 1])
        # return x * y.expand_as(x)
        # return x * y.expand_as(x)
        return x * y
        # return y

class PixelAttention(nn.Module):
    def __init__(self, dim):
        super(PixelAttention, self).__init__()
        self.pa2 = nn.Conv2d(2 * dim, dim, 7, padding=3, padding_mode='reflect', groups=dim, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, pattn1):
        B, C, H, W = x.shape
        x = x.unsqueeze(dim=2) # B, C, 1, H, W
        pattn1 = pattn1.unsqueeze(dim=2) # B, C, 1, H, W
        x2 = torch.cat([x, pattn1], dim=2) # B, C, 2, H, W
        x2 = Rearrange('b c t h w -> b (c t) h w')(x2)
        pattn2 = self.pa2(x2)
        pattn2 = self.sigmoid(pattn2)
        return pattn2

class ExtractFeature(nn.Module):
    def __init__(self,hashbit):
        super(ExtractFeature, self).__init__()
        self.resnet = resnet18(pretrained=True)
        for param in self.resnet.parameters():
            param.requires_grad = True

        self.f0conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.f01conv = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=7, stride=4, padding=3)
        self.f23conv = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=7, stride=4, padding=3)
        self.f2conv = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.fusionconv = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1)
        self.fusionconv1 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, bias=False)
        self.hagm_fusion = HAGM(512)
        self.hagm_f4 = HAGM(512)
        self.fc = nn.Linear(512, hashbit)
        self.tanh = nn.Tanh()
        self.head = nn.Linear(hashbit, 19) if 19 > 0 else nn.Identity()
        self.CBAM = cbam_block(512,16,7)

    def forward(self, img):
        x = self.resnet.conv1(img)  # [B, 64, 128, 128]
        x = self.resnet.bn1(x)
        f0 = self.resnet.relu(x)  # torch.Size([64, 64, 112, 112])
        x = self.resnet.maxpool(f0)  # [B, 64, 64, 64]

        f1 = self.resnet.layer1(x)  # torch.Size([64, 64, 56, 56])
        f2 = self.resnet.layer2(f1)  # torch.Size([64, 128, 28, 28])
        # x = self.ca_att(x)
        f3 = self.resnet.layer3(f2)  # [torch.Size([64, 256, 14, 14])
        f4 = self.resnet.layer4(f3)  # torch.Size([64, 512, 7, 7])
        # x = self.ca_att(x)

        # multi scale module
        f0 = self.f0conv(f0)  ## torch.Size([64, 128, 56, 56])
        f_0_1 = torch.cat([f0, f1], dim=1)  # torch.Size([64, 128, 56, 56])
        f_0_1 = self.f01conv(f_0_1)  # torch.Size([64, 128, 14, 14])

        f2 = self.f2conv(f2)  # torch.Size([64, 128, 14, 14])
        # f3 = self.ca_att_f3(f3)
        f_2_3 = torch.cat([f2, f3], dim=1)  # torch.Size([64, 384, 14, 14])
        # f_2_3 = self.f01conv(f_2_3)
        # f_2_3 = self.ca_att_f23(f_2_3)
        f_fusion = torch.cat([f_0_1, f_2_3], dim=1)
        f_fusion = self.fusionconv(f_fusion)  # torch.Size([64, 512, 7, 7])

        # f4 = torch.cat([f_fusion, f4], dim=1)
        # f4 = self.fusionconv1(f4)
        f4 = f_fusion + f4

        strategy_1 = 1  # 0:early fusion     1:later fusion
        if strategy_1 == 1:   #
            f_fusion = self.hagm_fusion(f_fusion)
            f_fusion_score = torch.sigmoid(f_fusion)

            f4 = self.hagm_f4(f4)
            f4_score = torch.sigmoid(f4)

            final = f4 * f_fusion_score + f4_score * f_fusion  # [B, 512, 8, 8]
            final = final + f4

            x = self.resnet.avgpool(final)  # [B, 512, 1, 1]
            x = torch.flatten(x, 1)  # [B, 512]
            x = self.fc(x)  # torch.Size([64, 16])
            x = self.tanh(x)  # torch.Size([64, 16])
            x = F.normalize(x, dim=1)
        elif strategy_1 == 2:   # resnet
            x = self.resnet.avgpool(f4)  # [B, 512, 1, 1]
            x = torch.flatten(x, 1)  # [B, 512]
            x = self.fc(x)  # torch.Size([64, 16])
            x = self.tanh(x)  # torch.Size([64, 16])
            x = F.normalize(x, dim=1)
        elif strategy_1 == 3:   # SE
            f_fusion = self.hagm_fusion(f_fusion)
            f_fusion_score = torch.sigmoid(f_fusion)

            f4 = self.hagm_f4(f4)
            f4_score = torch.sigmoid(f4)

            final = f4 * f_fusion_score + f4_score * f_fusion  # [B, 512, 8, 8]
            final = final + f4

            x = self.resnet.avgpool(final)  # [B, 512, 1, 1]
            x = torch.flatten(x, 1)  # [B, 512]
            x = self.fc(x)  # torch.Size([64, 16])
            x = self.tanh(x)  # torch.Size([64, 16])
            x = F.normalize(x, dim=1)
        elif strategy_1 == 4:   # CBAM
            f_fusion = self.CBAM(f_fusion)
            f_fusion_score = torch.sigmoid(f_fusion)

            f4 = self.CBAM(f4)
            f4_score = torch.sigmoid(f4)

            final = f4 * f_fusion_score + f4_score * f_fusion  # [B, 512, 8, 8]
            final = final + f4

            x = self.resnet.avgpool(final)  # [B, 512, 1, 1]
            x = torch.flatten(x, 1)  # [B, 512]
            x = self.fc(x)  # torch.Size([64, 16])
            x = self.tanh(x)  # torch.Size([64, 16])
            x = F.normalize(x, dim=1)
        else:
            f_fusion = f_fusion + f4
            ww = self.hagm_fusion(f_fusion)
            x = f4 * ww + f_fusion * (1-ww) + f4 + f_fusion
            x = self.resnet.avgpool(x)  # [B, 512, 1, 1]
            x = torch.flatten(x, 1)  # [B, 512]
            x = self.fc(x)  # torch.Size([64, 16])
            x = self.tanh(x)  # torch.Size([64, 16])
            x = F.normalize(x, dim=1)
        return x