import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np

def conv(in_planes, out_planes, kernel_size=3):
    return nn.Sequential(
        nn.Conv2d(in_planes,
                  out_planes,
                  kernel_size=kernel_size,
                  padding=(kernel_size - 1) // 2,
                  stride=2), nn.ReLU(inplace=True))

class VisualNet(nn.Module):
    '''
    Encode imgs into visual features
    '''
    def __init__(self):
        super(VisualNet, self).__init__()

        conv_planes = [16, 32, 64, 128, 256, 512, 512]
        self.conv1 = conv(6, conv_planes[0], kernel_size=7)
        self.conv2 = conv(conv_planes[0], conv_planes[1], kernel_size=5)
        self.conv3 = conv(conv_planes[1], conv_planes[2])
        self.conv4 = conv(conv_planes[2], conv_planes[3])
        self.conv5 = conv(conv_planes[3], conv_planes[4])
        self.conv6 = conv(conv_planes[4], conv_planes[5])
        self.conv7 = conv(conv_planes[5], conv_planes[6])
        self.conv8 = nn.Conv2d(conv_planes[6], conv_planes[6], kernel_size=1)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    init.zeros_(m.bias)

    def forward(self, imgs):
        visual_fea = []
        for i in range(len(imgs) - 1):
            input = torch.cat(imgs[i:i + 2], 1)
            out_conv1 = self.conv1(input)
            out_conv2 = self.conv2(out_conv1)
            out_conv3 = self.conv3(out_conv2)
            out_conv4 = self.conv4(out_conv3)
            out_conv5 = self.conv5(out_conv4)
            out_conv6 = self.conv6(out_conv5)
            out_conv7 = self.conv7(out_conv6)
            out_conv8 = self.conv8(out_conv7)

            visual_fea.append(out_conv8.mean(3).mean(2))
        return torch.stack(visual_fea, dim=1)


class ImuNet(nn.Module):
    '''
    Encode imus into imu feature
    '''
    def __init__(self, input_size=6, hidden_size=512):
        super(ImuNet, self).__init__()
        self.rnn = nn.LSTM(input_size=input_size,
                           hidden_size=hidden_size,
                           num_layers=2,
                           batch_first=True)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.LSTM):
                init.xavier_normal_(m.all_weights[0][0], gain=np.sqrt(1))
                init.xavier_normal_(m.all_weights[0][1], gain=np.sqrt(1))
                init.xavier_normal_(m.all_weights[1][0], gain=np.sqrt(1))
                init.xavier_normal_(m.all_weights[1][1], gain=np.sqrt(1))

    def forward(self, imus):
        self.rnn.flatten_parameters()
        x = imus
        B, t, N, _ = x.shape  # B T N 6
        x = x.reshape(B * t, N, -1)  # B T*N 6
        out, (h, c) = self.rnn(x)  # B*T 1000
        out = out[:, -1, :]
        return out.reshape(B, t, -1)


class FuseModule(nn.Module):
    def __init__(self, channels, reduction):
        super(FuseModule, self).__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channels // reduction, channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x

class PoseNet(nn.Module):
    '''
    Fuse both features and output the 6 DOF camera pose
    '''
    def __init__(self, input_size=1024):
        super(PoseNet, self).__init__()

        self.se = FuseModule(input_size, 16)

        self.rnn = nn.LSTM(input_size=input_size,
                           hidden_size=1024,
                           num_layers=2,
                           batch_first=True)

        self.fc1 = nn.Sequential(nn.Linear(1024, 6))

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.LSTM):
                init.xavier_normal_(m.all_weights[0][0], gain=np.sqrt(1))
                init.xavier_normal_(m.all_weights[0][1], gain=np.sqrt(1))
                init.xavier_normal_(m.all_weights[1][0], gain=np.sqrt(1))
                init.xavier_normal_(m.all_weights[1][1], gain=np.sqrt(1))
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight.data, gain=np.sqrt(1))
                if m.bias is not None:
                    init.zeros_(m.bias)

    def forward(self, visual_fea, imu_fea):
        self.rnn.flatten_parameters()
        if imu_fea is not None:
            B, t, _ = imu_fea.shape
            imu_input = imu_fea.view(B, t, -1)
            visual_input = visual_fea.view(B, t, -1)
            inpt = torch.cat((visual_input, imu_input), dim=2)
        else:
            inpt = visual_fea
        inpt = self.se(inpt)
        out, (h, c) = self.rnn(inpt)
        out = 0.01 * self.fc1(out)
        return out

if __name__ == '__main__':
    image_model = VisualNet()
    imgs = [torch.rand(4, 3, 256, 832)] * 5
    img_fea = image_model(imgs)
    print(img_fea.shape)

    imu_model = ImuNet()
    imus = torch.rand(4, 4, 11, 6)
    imu_fea = imu_model(imus)
    print(imu_fea.shape)

    pose_modle = PoseNet()
    pose = pose_modle(img_fea, imu_fea)
    print(pose.shape)
