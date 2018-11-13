import torch
import torch.nn as nn
import torchvision.models as models
import pdb

from basemodel.model import models as basemodels

class Net(nn.Module):
    def __init__(self, num_classes=100, norm=True, scale=True, options=None):
        super(Net,self).__init__()
        self.extractor = Extractor(options)
        self.embedding = Embedding(options)
        if options is None:
          self.classifier = Classifier(num_classes)
        else:
          self.classifier = Classifier(options['num_classes'])
        self.options = options
        self.s = nn.Parameter(torch.FloatTensor([10]))
        self.norm = norm
        self.scale = scale

    def forward(self, x):
        x = self.extractor(x)
        x = self.embedding(x)
        if self.norm:
            x = self.l2_norm(x)
        if self.scale:
            x = self.s * x
        x = self.classifier(x)
        return x

    def extract(self, x):
        x = self.extractor(x)
        x = self.embedding(x)
        x = self.l2_norm(x)
        return x

    def l2_norm(self,input):
        input_size = input.size()
        buffer = torch.pow(input, 2)

        normp = torch.sum(buffer, 1).add_(1e-10)
        norm = torch.sqrt(normp)

        _output = torch.div(input, norm.view(-1, 1).expand_as(input))

        output = _output.view(input_size)

        return output

    def weight_norm(self):
        w = self.classifier.fc.weight.data
        norm = w.norm(p=2, dim=1, keepdim=True)
        self.classifier.fc.weight.data = w.div(norm.expand_as(w))

class Extractor(nn.Module):
    def __init__(self, options=None):
        super(Extractor,self).__init__()
        self.options = options
        if options is None:
          basenet = models.resnet50(pretrained=True)
          self.extractor = nn.Sequential(*list(basenet.children())[:-1])
        else:
          # pdb.set_trace()
          basenet = basemodels.__dict__[options['arch']](num_classes=options['num_classes'], size_fm_2nd_head=28, options=options)
          self.extractor = basenet


    def forward(self, x):
        if self.options is None:
          x = self.extractor(x)
        else:
          # only takes star_representation (ret from two_heads -> forward)
          _, _, _, x = self.extractor(x)
        x = x.view(x.size(0), -1)
        return x

class Embedding(nn.Module):
    def __init__(self, options=None):
        super(Embedding,self).__init__()

        if options is None:
          self.fc = nn.Linear(2048, 256)
        else:
          self.fc = nn.Linear(options['D_star_embed'], 256)

    def forward(self, x):
        x = self.fc(x)
        return x

class Classifier(nn.Module):
    def __init__(self, num_classes):
        super(Classifier,self).__init__()
        self.fc = nn.Linear(256, num_classes, bias=False)

    def forward(self, x):
        x = self.fc(x)
        return x
