import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from resnet import resnet50
import config

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        # basic parameters
        self.num_anchors = len(config.aspect_ratios) * len(config.scale_ratios)
        self.num_classes = config.num_classes
        self.num_head_layers = config.num_head_layers

        # Aggregation input channels
        self.agg_dim1 = 32
        self.agg_dim2 = 64
        self.agg_dim3 = 128
        self.agg_dim4 = 256

        # backbone network
        self.backbone = resnet50(True)

        # fist part - feature extraction - fe
        self.fe_conv1 = nn.Conv2d(3072, 128, 1)
        self.fe_bn1 = nn.BatchNorm2d(128)
        self.fe_relu1 = nn.ReLU()

        self.fe_conv2 = nn.Conv2d(128, 128, 3, padding=1)
        self.fe_bn2 = nn.BatchNorm2d(128)
        self.fe_relu2 = nn.ReLU()

        self.fe_conv3 = nn.Conv2d(640, 64, 1)
        self.fe_bn3 = nn.BatchNorm2d(64)
        self.fe_relu3 = nn.ReLU()

        self.fe_conv4 = nn.Conv2d(64, 64, 3 ,padding=1)
        self.fe_bn4 = nn.BatchNorm2d(64)
        self.fe_relu4 = nn.ReLU()

        self.fe_conv5 = nn.Conv2d(320, 64, 1)
        self.fe_bn5 = nn.BatchNorm2d(64)
        self.fe_relu5 = nn.ReLU()

        self.fe_conv6 = nn.Conv2d(64, 32, 3, padding=1)
        self.fe_bn6 = nn.BatchNorm2d(32)
        self.fe_relu6 = nn.ReLU()

        self.fe_conv7 = nn.Conv2d(2048, 256, 3, padding=1)
        self.fe_bn7 = nn.BatchNorm2d(256)
        self.fe_relu7 = nn.ReLU()

        self.fe_unpool1 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.fe_unpool2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.fe_unpool3 = nn.Upsample(scale_factor=2, mode='bilinear')

        # second part - feature aggregration - fa
        ## for channel = 32
        self.fa1_conv1 = nn.Conv2d(self.agg_dim1, 128, kernel_size=1, stride=1, padding=0, bias=False)
        self.fa1_conv1_bn = nn.BatchNorm2d(128)
        self.fa1_conv2_dilation = nn.Conv2d(self.agg_dim1, 128, kernel_size=3, stride=1, padding=2, dilation=2, bias=False)
        self.fa1_conv2_bn = nn.BatchNorm2d(128)
        self.fa1_conv3 = nn.Conv2d(self.agg_dim1, 128, kernel_size=1, stride=1, padding=0, bias=False)
        self.fa1_conv3_bn = nn.BatchNorm2d(128)
        self.fa1_conv4_1 = nn.Conv2d(self.agg_dim1, 128, kernel_size=(1,5), stride=1, padding=(0, 4), dilation=2, bias=False)
        self.fa1_conv4_1_bn = nn.BatchNorm2d(128)
        self.fa1_conv4_2 = nn.Conv2d(128, 128, kernel_size=(5,1), stride=1, padding=(4, 0), dilation=2, bias=False)
        self.fa1_conv4_2_bn = nn.BatchNorm2d(128)

        ## for channel = 64
        self.fa2_conv1 = nn.Conv2d(self.agg_dim2, 128, kernel_size=1, stride=1, padding=0, bias=False)
        self.fa2_conv1_bn = nn.BatchNorm2d(128)
        self.fa2_conv2_dilation = nn.Conv2d(self.agg_dim2, 128, kernel_size=3, stride=1, padding=2, dilation=2, bias=False)
        self.fa2_conv2_bn = nn.BatchNorm2d(128)
        self.fa2_conv3 = nn.Conv2d(self.agg_dim2, 128, kernel_size=1, stride=1, padding=0, bias=False)
        self.fa2_conv3_bn = nn.BatchNorm2d(128)
        self.fa2_conv4_1 = nn.Conv2d(self.agg_dim2, 128, kernel_size=(1,5), stride=1, padding=(0, 4), dilation=2, bias=False)
        self.fa2_conv4_1_bn = nn.BatchNorm2d(128)
        self.fa2_conv4_2 = nn.Conv2d(128, 128, kernel_size=(5,1), stride=1, padding=(4, 0), dilation=2, bias=False)
        self.fa2_conv4_2_bn = nn.BatchNorm2d(128)

        ## for channel = 128
        self.fa3_conv1 = nn.Conv2d(self.agg_dim3, 128, kernel_size=1, stride=1, padding=0, bias=False)
        self.fa3_conv1_bn = nn.BatchNorm2d(128)
        self.fa3_conv2_dilation = nn.Conv2d(self.agg_dim3, 128, kernel_size=3, stride=1, padding=2, dilation=2, bias=False)
        self.fa3_conv2_bn = nn.BatchNorm2d(128)
        self.fa3_conv3 = nn.Conv2d(self.agg_dim3, 128, kernel_size=1, stride=1, padding=0, bias=False)
        self.fa3_conv3_bn = nn.BatchNorm2d(128)
        self.fa3_conv4_1 = nn.Conv2d(self.agg_dim3, 128, kernel_size=(1,5), stride=1, padding=(0, 4), dilation=2, bias=False)
        self.fa3_conv4_1_bn = nn.BatchNorm2d(128)
        self.fa3_conv4_2 = nn.Conv2d(128, 128, kernel_size=(5,1), stride=1, padding=(4, 0), dilation=2, bias=False)
        self.fa3_conv4_2_bn = nn.BatchNorm2d(128)

        ## for channel = 256
        self.fa4_conv1 = nn.Conv2d(self.agg_dim4, 128, kernel_size=1, stride=1, padding=0, bias=False)
        self.fa4_conv1_bn = nn.BatchNorm2d(128)
        self.fa4_conv2_dilation = nn.Conv2d(self.agg_dim4, 128, kernel_size=3, stride=1, padding=2, dilation=2, bias=False)
        self.fa4_conv2_bn = nn.BatchNorm2d(128)
        self.fa4_conv3 = nn.Conv2d(self.agg_dim4, 128, kernel_size=1, stride=1, padding=0, bias=False)
        self.fa4_conv3_bn = nn.BatchNorm2d(128)
        self.fa4_conv4_1 = nn.Conv2d(self.agg_dim4, 128, kernel_size=(1,5), stride=1, padding=(0, 4), dilation=2, bias=False)
        self.fa4_conv4_1_bn = nn.BatchNorm2d(128)
        self.fa4_conv4_2 = nn.Conv2d(128, 128, kernel_size=(5,1), stride=1, padding=(4, 0), dilation=2, bias=False)
        self.fa4_conv4_2_bn = nn.BatchNorm2d(128)

        # third part - aggregation & attention - aa
        self.aa_conv1 = nn.Conv2d(512, 512, kernel_size=3, padding=2, stride=1, dilation=2, bias=False)
        self.aa_conv1_bn = nn.BatchNorm2d(512)
        self.aa_conv2 = nn.Conv2d(2, 2, kernel_size=3, padding=2, stride=1, dilation=2, bias=False)
        self.aa_conv2_bn = nn.BatchNorm2d(2)
        self.aa_conv3 = nn.Conv2d(2, 2, kernel_size=3, padding=2, stride=1, dilation=2)
        self.softmax = nn.Softmax2d()
        self.deconv_fa3 = nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1, bias=False, groups=64)
        self.deconv_fa4 = nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1, bias=False, groups=64)
        self.deconv_mask = nn.ConvTranspose2d(512, 2, kernel_size=16, padding=4, stride=8, groups=2)

        # forth part - predict class & location
        self.loc_head = self._make_head(self.num_anchors * 4)
        self.cls_head = self._make_head(self.num_anchors * self.num_classes)
  
    def _make_head(self, output_dim):
        layers = []

        layers.append(nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1))
        layers.append(nn.ReLU(True))
        for _ in range(self.num_head_layers - 1):
            layers.append(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU(True))
        layers.append(nn.Conv2d(256, output_dim, kernel_size=3, stride=1, padding=1))
        return nn.Sequential(*layers)

    
    def forward(self, images):
        # first part - feature extraction
        _, f = self.backbone(images)
        h4_ = f[3]  # bs 2048 16 16
        h4 = self.fe_relu7(self.fe_bn7(self.fe_conv7(h4_))) # bs 256 16 16
        h3_ = self.fe_relu1(self.fe_bn1(self.fe_conv1(torch.cat((self.fe_unpool1(h4_), f[2]), 1))))
        h3 = self.fe_relu2(self.fe_bn2(self.fe_conv2(h3_))) # bs 128 32 32
        h2_ = self.fe_relu3(self.fe_bn3(self.fe_conv3(torch.cat((self.fe_unpool2(h3), f[1]), 1))))
        h2 = self.fe_relu4(self.fe_bn4(self.fe_conv4(h2_))) # bs 64 64 64
        h1_ = self.fe_relu5(self.fe_bn5(self.fe_conv5(torch.cat((self.fe_unpool3(h2), f[0]), 1))))
        h1 = self.fe_relu6(self.fe_bn6(self.fe_conv6(h1_))) # bs 32 128 128
        
        # second part - feature aggregation
        ## for chennel = 32
        fa1_c1_out = F.relu(self.fa1_conv1_bn(self.fa1_conv1(h1)))
        fa1_c2_out = F.relu(self.fa1_conv2_bn(self.fa1_conv2_dilation(h1)))
        fa1_c3_out = F.relu(self.fa1_conv3_bn(self.fa1_conv3(F.max_pool2d(h1, kernel_size=3, stride=1, padding=1))))
        fa1_c4_out = F.relu(self.fa1_conv4_1_bn(self.fa1_conv4_1(h1)))
        fa1_c4_out = F.relu(self.fa1_conv4_2_bn(self.fa1_conv4_2(fa1_c4_out)))
        fa1_ = [fa1_c1_out, fa1_c2_out, fa1_c3_out, fa1_c4_out]
        fa1 = torch.cat(fa1_, dim=1)

        ## for chennel = 64
        fa2_c1_out = F.relu(self.fa2_conv1_bn(self.fa2_conv1(h2)))
        fa2_c2_out = F.relu(self.fa2_conv2_bn(self.fa2_conv2_dilation(h2)))
        fa2_c3_out = F.relu(self.fa2_conv3_bn(self.fa2_conv3(F.max_pool2d(h2, kernel_size=3, stride=1, padding=1))))
        fa2_c4_out = F.relu(self.fa2_conv4_1_bn(self.fa2_conv4_1(h2)))
        fa2_c4_out = F.relu(self.fa2_conv4_2_bn(self.fa2_conv4_2(fa2_c4_out)))
        fa2_ = [fa2_c1_out, fa2_c2_out, fa2_c3_out, fa2_c4_out]
        fa2 = torch.cat(fa2_, dim=1)

        ## for chennel = 128
        fa3_c1_out = F.relu(self.fa3_conv1_bn(self.fa3_conv1(h3)))
        fa3_c2_out = F.relu(self.fa3_conv2_bn(self.fa3_conv2_dilation(h3)))
        fa3_c3_out = F.relu(self.fa3_conv3_bn(self.fa3_conv3(F.max_pool2d(h3, kernel_size=3, stride=1, padding=1))))
        fa3_c4_out = F.relu(self.fa3_conv4_1_bn(self.fa3_conv4_1(h3)))
        fa3_c4_out = F.relu(self.fa3_conv4_2_bn(self.fa3_conv4_2(fa3_c4_out)))
        fa3_ = [fa3_c1_out, fa3_c2_out, fa3_c3_out, fa3_c4_out]
        fa3 = torch.cat(fa3_, dim=1)

        ## for chennel = 256
        fa4_c1_out = F.relu(self.fa4_conv1_bn(self.fa4_conv1(h4)))
        fa4_c2_out = F.relu(self.fa4_conv2_bn(self.fa4_conv2_dilation(h4)))
        fa4_c3_out = F.relu(self.fa4_conv3_bn(self.fa4_conv3(F.max_pool2d(h4, kernel_size=3, stride=1, padding=1))))
        fa4_c4_out = F.relu(self.fa4_conv4_1_bn(self.fa4_conv4_1(h4)))
        fa4_c4_out = F.relu(self.fa4_conv4_2_bn(self.fa4_conv4_2(fa4_c4_out)))
        fa4_ = [fa4_c1_out, fa4_c2_out, fa4_c3_out, fa4_c4_out]
        fa4 = torch.cat(fa4_, dim=1)

        # third part - class & localition
        ## downsample and upsample
        down_fa1 = F.max_pool2d(fa1, kernel_size=3, stride=2, padding=1)  # bs 512 64 64
        up_fa3 = self.deconv_fa3(fa3) # bs 512 64 64
        down_fa2 = F.max_pool2d(fa2, kernel_size=3, stride=2, padding=1) # bs 512 32 32
        up_fa4 = self.deconv_fa4(fa4) # bs 512 32 32

        ## aggregate high level features
        agg1 = down_fa1 + fa2 + up_fa3 # bs 512 64 64
        agg2 = down_fa2 + fa3 + up_fa4 # bs 512 32 32

        ## generate mask & attention
        mask = down_fa1 + fa2 + up_fa3 # bs 512 64 64
        mask = F.relu(self.aa_conv1_bn(self.aa_conv1(mask))) # bs 512 64 64
        mask = self.deconv_mask(mask) # bs 2 512 512
        mask = F.relu(self.aa_conv2_bn(self.aa_conv2(mask))) # bs 2 512 512
        mask = self.aa_conv3(mask) # bs 2 512 512
        attention = self.softmax(mask) # bs 2 512 512
        attention = attention[:,1:2,:,:] # bs 1 512 512 
        attention = F.avg_pool2d(attention, kernel_size=4, stride=4) # bs 1 128 128

        ## generate attention agg feature
        attention = F.avg_pool2d(attention, kernel_size=2, stride=2) # bs 1 64 64
        att_agg1 = attention * agg1 # bs 512 64 64
        attention = F.avg_pool2d(attention, kernel_size=2, stride=2) # bs 1 32 32
        att_agg2 = attention * agg2 # bs 512 32 32

        # predict class & location
        fms = [att_agg1, att_agg2]
        loc_preds = []
        cls_preds = []
        for fm in fms:
            loc_pred = self.loc_head(fm) # bs 4*24 64 64
            cls_pred = self.cls_head(fm) # bs 1*24 64 64
            loc_pred = loc_pred.permute(0,2,3,1).contiguous().view(images.size(0), -1, 4)                 # [N, 4*24,H,W] -> [N,H,W, 4*24] -> [N,H*W*24, 4]
            # bs 98304 4
            # bs 24576 4
            cls_pred = cls_pred.permute(0,2,3,1).contiguous().view(images.size(0), -1, self.num_classes)  # [N,1*24,H,W] -> [N,H,W,1*24] -> [N,H*W*24,1]
            # bs 98304 1
            # bs 24576 1
            loc_preds.append(loc_pred)
            cls_preds.append(cls_pred)
        return torch.cat(loc_preds,1), torch.cat(cls_preds,1), mask
                # bs 122880 4           bs 122880 1             bs 2 512 512

