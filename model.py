import  torch
from torch import nn
from transformers import ViltConfig, ViltModel
from torchvision import models

class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()
        model = models.resnet152()
        modules = list(model.children())[:-2]
        self.model = nn.Sequential(*modules)
        self.pool1 = nn.AdaptiveAvgPool2d(output_size=(4,1))
        self.pool2 = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.fc = nn.Linear(2048, 512)

    def forward(self, x):
        x = self.model(x)
        x = self.pool1(x)

        img_embeds = torch.flatten(x, start_dim=2)
        img_embeds = img_embeds.transpose(1, 2).contiguous()
        img_embeds = self.fc(img_embeds)

        img_v = self.pool2(x)
        img_v = torch.flatten(img_v, 1)
        img_v = self.fc(img_v)

        return img_v, img_embeds



class SpecEncoder(nn.Module):
    def __init__(self):
        super(SpecEncoder, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=4)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=4)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=4)
        self.conv4 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=4)
        self.conv5 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=4)

        self.pool = nn.MaxPool1d(kernel_size=4)
        self.advpool1 = nn.AdaptiveAvgPool1d(output_size=32)
        self.advpool2 = nn.AdaptiveAvgPool1d(output_size=1)
        self.fc = nn.Linear(512, 512)


    def forward(self, x):
        x = x.unsqueeze(dim = 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool(x)
        x = self.conv5(x)

        spec_embeds = self.advpool1(x)
        spec_embeds = spec_embeds.transpose(1, 2).contiguous()

        spec_v = self.advpool2(x)
        spec_v = torch.squeeze(spec_v)
        spec_v = self.fc(spec_v)

        return spec_v, spec_embeds


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.image_encoder = ImageEncoder()
        self.spec_encoder = SpecEncoder()
        config = ViltConfig(max_position_embeddings=65,hidden_size=512,
                            num_attention_heads=8,num_hidden_layers=8,intermediate_size=2048)
        self.fusion = ViltModel(config)
        self.classifier = nn.Linear(512, 5)
        self.t = 0.07
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 512))

    def forward(self, specs, imgs):

        imgs_v, imgs_embeds = self.image_encoder(imgs)
        imgs_v = imgs_v / imgs_v.norm(p=2, dim=-1, keepdim=True)

        specs_v, specs_embeds = self.spec_encoder(specs)
        specs_v = specs_v / specs_v.norm(p=2, dim=-1, keepdim=True)

        batch_size, img_seq_len, hidden_size = imgs_embeds.shape

        cls_token = self.cls_token.expand(batch_size, -1, -1)
        specs_embeds = torch.cat((cls_token, specs_embeds), dim=1)

        align_logits = torch.div(torch.matmul(imgs_v, specs_v.T), self.t)
        pixel_values = torch.ones([batch_size, 3, 4, 1])  # for transformers 4.18.0
        output = self.fusion(inputs_embeds=specs_embeds, image_embeds=imgs_embeds, pixel_values = pixel_values)

        cls = output.pooler_output
        fuse_logits = self.classifier(cls)

        return align_logits , fuse_logits




