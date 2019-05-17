"""
ResUNet整体上网络结构基于VNet，做出的修改如下：
将原本555的卷积核换成了333
在除了第一个和最后一个block中添加了dropout
去掉了编码器部分的最后一个16倍降采样的stage
为了弥补这么做带来的感受野的损失，在编码器的最后两个stage加入了混合空洞卷积
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from config import num_organ, dropout_rate, refine_box_list, box_edge_radio
import math

def get_core(inputs, box, box_edge):
    s = inputs.shape
    i = 0
    e1,e2 = box_edge[1], box_edge[2]
    x1 = math.floor(box[i, 1]-e1)
    x2 = math.ceil(box[i, 1]+e1)
    y1 = math.floor(box[i, 2]-e2)
    y2 = math.ceil(box[i, 2]+e2)

    ans = torch.zeros(s[0], s[1], s[2], x2-x1, y2-y1).cuda()
    for i in range(inputs.shape[0]):
        row = inputs[i]
        x1 = math.floor(box[i, 1]-e1)
        x2 = math.ceil(box[i, 1]+e1)
        y1 = math.floor(box[i, 2]-e2)
        y2 = math.ceil(box[i, 2]+e2)
        ans[i] = row[:, :, x1:x2, y1:y2]
    return ans

# 定义单个3D FCN
class ResUNet(nn.Module):
    """
    共9332094个可训练的参数, 九百三十万左右
    """
    def __init__(self, training, inchannel, stage):
        """
        :param training: 标志网络是属于训练阶段还是测试阶段
        :param inchannel 网络最开始的输入通道数量
        :param stage 标志网络属于第一阶段，还是第二阶段
        """
        super().__init__()

        self.training = training
        self.stage = stage
        if stage == 'stage1':
            self.gf = 16
        elif stage == 'stage2':
            self.gf = 16
        else:
            self.gf = 16

        self.encoder_stage1 = nn.Sequential(
            nn.Conv3d(inchannel, self.gf, 3, 1, padding=1),
            nn.PReLU(self.gf),
        )

        self.encoder_stage2 = nn.Sequential(
            nn.Conv3d(self.gf*2, self.gf*2, 3, 1, padding=1),
            nn.PReLU(self.gf*2),

            nn.Conv3d(self.gf*2, self.gf*2, 3, 1, padding=1),
            nn.PReLU(self.gf*2),
        )

        self.encoder_stage3 = nn.Sequential(
            nn.Conv3d(self.gf*4, self.gf*4, 3, 1, padding=1),
            nn.PReLU(self.gf*4),

            nn.Conv3d(self.gf*4, self.gf*4, 3, 1, padding=1),
            nn.PReLU(self.gf*4),

            nn.Conv3d(self.gf*4, self.gf*4, 3, 1, padding=1),
            nn.PReLU(self.gf*4),
        )

        self.encoder_stage4 = nn.Sequential(
            nn.Conv3d(self.gf*8, self.gf*8, 3, 1, padding=1),
            nn.PReLU(self.gf*8),

            nn.Conv3d(self.gf*8, self.gf*8, 3, 1, padding=1),
            nn.PReLU(self.gf*8),

            nn.Conv3d(self.gf*8, self.gf*8, 3, 1, padding=1),
            nn.PReLU(self.gf*8),
        )

        self.decoder_stage1 = nn.Sequential(
            nn.Conv3d(self.gf*8, self.gf*16, 3, 1, padding=1),
            nn.PReLU(self.gf*16),

            nn.Conv3d(self.gf*16, self.gf*16, 3, 1, padding=1),
            nn.PReLU(self.gf*16),

            nn.Conv3d(self.gf*16, self.gf*16, 3, 1, padding=1),
            nn.PReLU(self.gf*16),
        )

        self.decoder_stage2 = nn.Sequential(
            nn.Conv3d(self.gf*8 + self.gf*4, self.gf*8, 3, 1, padding=1),
            nn.PReLU(self.gf*8),

            nn.Conv3d(self.gf*8, self.gf*8, 3, 1, padding=1),
            nn.PReLU(self.gf*8),

            nn.Conv3d(self.gf*8, self.gf*8, 3, 1, padding=1),
            nn.PReLU(self.gf*8),
        )

        self.decoder_stage3 = nn.Sequential(
            nn.Conv3d(self.gf*4 + self.gf*2, self.gf*4, 3, 1, padding=1),
            nn.PReLU(self.gf*4),

            nn.Conv3d(self.gf*4, self.gf*4, 3, 1, padding=1),
            nn.PReLU(self.gf*4),
        )

        self.decoder_stage4 = nn.Sequential(
            nn.Conv3d(self.gf*2 + self.gf, self.gf*2, 3, 1, padding=1),
            nn.PReLU(self.gf*2),
        )

        self.down_conv1 = nn.Sequential(
            nn.Conv3d(self.gf, self.gf*2, 2, 2),
            nn.PReLU(self.gf*2)
        )

        self.down_conv2 = nn.Sequential(
            nn.Conv3d(self.gf*2, self.gf*4, 2, 2),
            nn.PReLU(self.gf*4)
        )

        self.down_conv3 = nn.Sequential(
            nn.Conv3d(self.gf*4, self.gf*8, 2, 2),
            nn.PReLU(self.gf*8)
        )

        self.down_conv4 = nn.Sequential(
            nn.Conv3d(self.gf*8, self.gf*16, 3, 1, padding=1),
            nn.PReLU(self.gf*16)
        )

        self.up_conv2 = nn.Sequential(
            nn.ConvTranspose3d(self.gf*16, self.gf*8, 2, 2),
            nn.PReLU(self.gf*8)
        )

        self.up_conv3 = nn.Sequential(
            nn.ConvTranspose3d(self.gf*8, self.gf*4, 2, 2),
            nn.PReLU(self.gf*4)
        )

        self.up_conv4 = nn.Sequential(
            nn.ConvTranspose3d(self.gf*4, self.gf*2, 2, 2),
            nn.PReLU(self.gf*2)
        )

        self.map = nn.Sequential(
            nn.Conv3d(self.gf*2, num_organ + 1, 1),
            nn.Softmax(dim=1)
        )


    def forward(self, inputs):

        if self.stage is 'stage1':
            long_range1 = self.encoder_stage1(inputs) + inputs
        else:
            long_range1 = self.encoder_stage1(inputs)

        short_range1 = self.down_conv1(long_range1)

        long_range2 = self.encoder_stage2(short_range1) + short_range1
        long_range2 = F.dropout(long_range2, dropout_rate, self.training)

        short_range2 = self.down_conv2(long_range2)

        long_range3 = self.encoder_stage3(short_range2) + short_range2
        long_range3 = F.dropout(long_range3, dropout_rate, self.training)

        short_range3 = self.down_conv3(long_range3)

        long_range4 = self.encoder_stage4(short_range3) + short_range3
        long_range4 = F.dropout(long_range4, dropout_rate, self.training)

        short_range4 = self.down_conv4(long_range4)

        outputs = self.decoder_stage1(long_range4) + short_range4
        outputs_bottle = F.dropout(outputs, dropout_rate, self.training)

        short_range6 = self.up_conv2(outputs_bottle)

        outputs = self.decoder_stage2(torch.cat([short_range6, long_range3], dim=1)) + short_range6
        outputs = F.dropout(outputs, dropout_rate, self.training)

        short_range7 = self.up_conv3(outputs)

        outputs = self.decoder_stage3(torch.cat([short_range7, long_range2], dim=1)) + short_range7
        outputs = F.dropout(outputs, dropout_rate, self.training)

        short_range8 = self.up_conv4(outputs)

        outputs = self.decoder_stage4(torch.cat([short_range8, long_range1], dim=1)) + short_range8

        outputs = self.map(outputs)

        # 返回概率图
        return outputs

# 定义最终的级连3D FCN
class Net(nn.Module):
    def __init__(self, training):
        super().__init__()

        self.training = training

        self.stage1 = ResUNet(training=training, inchannel=1, stage='stage1')
        self.stage2 = ResUNet(training=training, inchannel=num_organ + 1 + 1, stage='stage2')
        self.stage3 = ResUNet(training=training, inchannel= (num_organ+1)*2 + 1, stage='stage3')

    def forward(self, inputs, box_id=None):
        # 首先取出core part
        x = inputs.shape[-3]
        y = inputs.shape[-2]
        z = inputs.shape[-1]

        #------------
        # STAGE 1
        #------------
        inputs_stage1 = F.upsample(inputs, (x, y//2, z//2), mode='trilinear')
        # 得到第一阶段的粗略结果
        output_stage1 = self.stage1(inputs_stage1)
        output_stage1 = F.upsample(output_stage1, (x, y, z), mode='trilinear')

        #------------
        # STAGE 2
        #------------
        # 将第一阶段的结果与原始输入数据进行拼接作为第二阶段的输入
        inputs_stage2 = torch.cat((output_stage1.detach(), inputs), dim=1)
        # 得到第二阶段的结果
        output_stage2 = self.stage2(inputs_stage2)

        if box_id is None:
            box_id = np.random.randint(low=6, high=9, size=(inputs.shape[0]))

        box = np.array(refine_box_list)[box_id]
        box[:, 0] *= x
        box[:, 1] *= y
        box[:, 2] *= z
        box_edge = x*box_edge_radio, y*box_edge_radio, z*box_edge_radio

        #------------
        # STAGE 3
        #------------
        inputs_core = get_core(inputs.clone().detach(), box, box_edge)
        output_stage2_core = get_core(output_stage2.clone().detach(), box, box_edge)
        output_stage2_down = F.upsample(output_stage2.detach(), (x, y//2, z//2), mode='trilinear')

        # 将第二阶段的结果与原始输入数据进行拼接作为第三阶段的输入
        inputs_stage3 = torch.cat((output_stage2_core.detach(), output_stage2_down, inputs_core), dim=1)
        # 得到第三阶段的结果
        output_stage3 = self.stage3(inputs_stage3)
        return output_stage1, output_stage2, output_stage3, output_stage2_core, torch.tensor(box_id).cuda()

    def env(self, inputs, stage2, box_id, loss_func, gt=None):
        x = inputs.shape[-3]
        y = inputs.shape[-2]
        z = inputs.shape[-1]
        box = np.array(refine_box_list)[box_id]
        box = np.expand_dims(box, axis=0)
        box[:, 0] *= x
        box[:, 1] *= y
        box[:, 2] *= z
        box_edge = x*box_edge_radio, y*box_edge_radio, z*box_edge_radio

        #------------
        # STAGE 3
        #------------
        inputs_core = get_core(inputs.clone().detach(), box, box_edge)
        output_stage2_core = get_core(stage2.clone().detach(), box, box_edge)
        output_stage2_down = F.upsample(stage2.detach(), (x, y//2, z//2), mode='trilinear')

        inputs_stage3 = torch.cat((output_stage2_core, output_stage2_down, inputs_core), dim=1)
        output_stage3 = self.stage3(inputs_stage3)
        if gt is not None:
            dice_bf = loss_func.env(stage2, gt)
        stage2[:, :, :,
            math.floor(box[0, 1]-box_edge[1]):math.ceil(box[0, 1]+box_edge[1]),
            math.floor(box[0, 2]-box_edge[2]):math.ceil(box[0, 2]+box_edge[2])
            ] = output_stage3

        if gt is not None:
            # get reward
            dice_af = loss_func.env(stage2, gt)
            return stage2, (dice_af-dice_bf)
        else:
            return stage2

# 网络参数初始化函数
def init(module):
    if isinstance(module, nn.Conv3d) or isinstance(module, nn.ConvTranspose3d):
        nn.init.kaiming_normal(module.weight.data, 0.25)
        nn.init.constant(module.bias.data, 0)


net = Net(training=True)
net.apply(init)

