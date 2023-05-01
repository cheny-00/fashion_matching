from typing import Callable, List, Optional, Type, Union
import torch
from torch import Tensor
import torch.nn as nn
from torchvision.models import ResNet18_Weights,ResNet50_Weights, ResNet
from torchvision.models.resnet import BasicBlock, Bottleneck

from center_loss import CenterLoss


    
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.base_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', weights=ResNet18_Weights.DEFAULT)
        # self.base_model.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        n_features = self.base_model.fc.in_features
        self.base_model = nn.Sequential(*list(self.base_model.children())[:-1])
        # self.fc = nn.Sequential(
        #     nn.Linear(n_features * 2, 256),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(256, 1),
        # )
        # self.fc = nn.Sequential(
        #     nn.Dropout(p=0.5),
        #     nn.Linear(n_features * 2, 512),
        #     nn.BatchNorm1d(512),
        #     nn.ReLU(),

        #     nn.Dropout(p=0.5),
        #     nn.Linear(512, 64),
        #     nn.BatchNorm1d(64),
        #     nn.Sigmoid(),
        #     nn.Dropout(p=0.5),

        #     nn.Linear(64, 1),
        # )

        # self.sigmoid = nn.Sigmoid()

    def forward_once(self, x):
        x = self.base_model(x)
        # x = x.view(x.size(0), -1)
        return x

    def forward(self, input1, input2):
        x = self.forward_once(input1)
        y = self.forward_once(input2)

        output = None
        # output = torch.cat((x, y), 1)
        # output = self.fc(output)
        # output = self.sigmoid(output)
        return x, y, output

class BackBone(nn.Module):
    def __init__(self, model_name=None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # if model_name is None: # map different model with pretrained weight
        #     model_name = 'resnet18'
        # self.base_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', weights=ResNet50_Weights.DEFAULT)
        # self.base_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V2)
        self.base_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        if hasattr(self.base_model, "fc"):
            self.base_model.fc = torch.nn.Identity()
        elif hasattr(self.base_model, "classifier"):
            self.base_model.classifier = torch.nn.Identity()
        
        self.base_model.avgpool = torch.nn.Identity()
        self.gap = nn.AdaptiveAvgPool2d(1)
        
    def forward(self, x):
        out = self.base_model(x)
        global_feat = self.gap(out)
        global_feat = global_feat.view(global_feat.shape[0], -1)
        
        return out, global_feat
        

class EmbedModel(nn.Module):
    
    def __init__(self, params, num_classes, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        d_model = params['d_model']
        
        self.backbone = BackBone()
        
        self.bn = nn.BatchNorm1d(d_model)
        self.bn.bias.requires_grad_(False)
        
        self.fc_query = nn.Linear(d_model, num_classes, bias=False)
        self.fc_query.apply(weights_init_classifier)
        self.center_loss_fn = CenterLoss(num_classes=num_classes, feat_dim=d_model, use_gpu=params['device']=='cuda')
    
    
    def forward(self, x):
        _, features = self.backbone(x)
        bn_features = self.bn(features)
        cls_score = self.fc_query(bn_features)
        return features, cls_score



def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode="fan_out")
        nn.init.constant_(m.bias, 0.0)
    elif classname.find("Conv") != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find("BatchNorm") != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)
            

class LocalResNet(ResNet):
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.fc(x)

        return x
    
# from https://github.com/pytorch/vision/blob/main/torchvision/models/_utils.py#L235pytorch

def _ovewrite_named_param(kwargs, param, new_value) -> None:
    if param in kwargs:
        if kwargs[param] != new_value:
            raise ValueError(f"The parameter '{param}' expected value {new_value} but got {kwargs[param]} instead.")
    else:
        kwargs[param] = new_value

def _resnet(
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    weights,
    progress,
    **kwargs
) -> ResNet:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = LocalResNet(block, layers, **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model

def resnet18(*, weights: Optional[ResNet18_Weights] = None, progress: bool = True, **kwargs) -> ResNet:

    weights = ResNet18_Weights.verify(weights)

    return _resnet(BasicBlock, [2, 2, 2, 2], weights, progress, **kwargs)

def resnet50(*, weights: Optional[ResNet50_Weights] = None, progress: bool = True, **kwargs) -> ResNet:
    weights = ResNet50_Weights.verify(weights)

    return _resnet(Bottleneck, [3, 4, 6, 3], weights, progress, **kwargs)