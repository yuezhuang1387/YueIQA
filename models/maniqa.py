import sys
sys.path.append('./')
import torch
import torch.nn as nn
# 注意，此处使用的是项目中的老timm包，环境中不要通过pip再安装timm
import timm
from timm.models.vision_transformer import Block
from models.swin import SwinTransformer
from torch import nn
from einops import rearrange


class TABlock(nn.Module):
    def __init__(self, dim, drop=0.1):
        super().__init__()
        self.c_q = nn.Linear(dim, dim)
        self.c_k = nn.Linear(dim, dim)
        self.c_v = nn.Linear(dim, dim)
        self.norm_fact = dim ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.proj_drop = nn.Dropout(drop)

    def forward(self, x):
        _x = x
        B, C, N = x.shape
        q = self.c_q(x)
        k = self.c_k(x)
        v = self.c_v(x)

        attn = q @ k.transpose(-2, -1) * self.norm_fact
        attn = self.softmax(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, C, N)
        x = self.proj_drop(x)
        x = x + _x
        return x


class SaveOutput:
    def __init__(self):
        self.outputs = []
    
    def __call__(self, module, module_in, module_out):
        '''
        保存每个transformer block经过forward后的输出，vit-base depth=12，transformer block有12个
        :param module: module.__class__ # <class 'timm.models.vision_transformer.Block'>
        :param module_in: 元组，长度1，module_in[0]才为实际输入tensor，module_in[0].shape: torch.Size([N, 785, 768])
        :param module_out: 直接为实际输出tensor，torch.Size([N, 785, 768])
        :return:
        '''
        self.outputs.append(module_out)
    
    def clear(self):
        self.outputs = []


class MANIQA(nn.Module):
    def __init__(self, embed_dim=72, num_outputs=1, patch_size=8, drop=0.1,
                 depths=[2, 2], window_size=4, dim_mlp=768, num_heads=[4, 4],
                 img_size=224, num_tab=2, scale=0.8, **kwargs):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.input_size = img_size // patch_size
        self.patches_resolution = (img_size // patch_size, img_size // patch_size)

        self.vit = timm.create_model('vit_base_patch8_224', pretrained=True)
        # hook机制在多卡训练时存在问题
        # self.save_output = SaveOutput()
        # hook_handles = []
        # for layer in self.vit.modules():
        #     if isinstance(layer, Block):
        #         # 对transformer block注册hook，vit-base的depth=12，有12个transformer block
        #         handle = layer.register_forward_hook(hook=self.save_output)
        #         # 保存handle对象
        #         hook_handles.append(handle)

        self.tablock1 = nn.ModuleList()
        for i in range(num_tab):
            tab = TABlock(self.input_size ** 2)
            self.tablock1.append(tab)

        self.conv1 = nn.Conv2d(embed_dim * 4, embed_dim, 1, 1, 0)
        self.swintransformer1 = SwinTransformer(
            patches_resolution=self.patches_resolution,
            depths=depths,
            num_heads=num_heads,
            embed_dim=embed_dim,
            window_size=window_size,
            dim_mlp=dim_mlp,
            scale=scale
        )

        self.tablock2 = nn.ModuleList()
        for i in range(num_tab):
            tab = TABlock(self.input_size ** 2)
            self.tablock2.append(tab)

        self.conv2 = nn.Conv2d(embed_dim, embed_dim // 2, 1, 1, 0)
        self.swintransformer2 = SwinTransformer(
            patches_resolution=self.patches_resolution,
            depths=depths,
            num_heads=num_heads,
            embed_dim=embed_dim // 2,
            window_size=window_size,
            dim_mlp=dim_mlp,
            scale=scale
        )

        self.fc_score = nn.Sequential(
            nn.Linear(embed_dim // 2, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(embed_dim // 2, num_outputs),
            nn.ReLU()
        )
        self.fc_weight = nn.Sequential(
            nn.Linear(embed_dim // 2, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(embed_dim // 2, num_outputs),
            nn.Sigmoid()
        )

    def extract_feature(self, transformer_block_output_list):
        # 取四个阶段的输出，并且去掉cls标识符
        # print(f'长度为: {len(transformer_block_output_list)}')  # 12
        x6 = transformer_block_output_list[-4][:, 1:]  # torch.Size([N, 784, 768])
        x7 = transformer_block_output_list[-3][:, 1:]
        x8 = transformer_block_output_list[-2][:, 1:]
        x9 = transformer_block_output_list[-1][:, 1:]
        x = torch.cat((x6, x7, x8, x9), dim=2)
        # torch.Size([N, 785, 768×4=3072])
        return x

    def forward(self, x):
        _x, l = self.vit(x)  # _x 不需要，只是为了得到vit中transformer block阶段的输出
        # self.vit执行完推理后每个transformer block的输出就保存到了self.save_output.outputs这个list中，12个输出
        x = self.extract_feature(l)  # torch.Size([N, 785, 768×4=3072])

        # stage 1
        x = rearrange(x, 'b (h w) c -> b c (h w)', h=self.input_size, w=self.input_size)
        for tab in self.tablock1:
            x = tab(x)
        x = rearrange(x, 'b c (h w) -> b c h w', h=self.input_size, w=self.input_size)
        x = self.conv1(x)
        x = self.swintransformer1(x)

        # stage2
        x = rearrange(x, 'b c h w -> b c (h w)', h=self.input_size, w=self.input_size)
        for tab in self.tablock2:
            x = tab(x)
        x = rearrange(x, 'b c (h w) -> b c h w', h=self.input_size, w=self.input_size)
        x = self.conv2(x)
        x = self.swintransformer2(x)

        x = rearrange(x, 'b c h w -> b (h w) c', h=self.input_size, w=self.input_size)
        score = torch.tensor([]).cuda()

        for i in range(x.shape[0]):
            f = self.fc_score(x[i])
            w = self.fc_weight(x[i])
            _s = torch.sum(f * w) / torch.sum(w)
            score = torch.cat((score, _s.unsqueeze(0)), 0)
        return score

if __name__ == '__main__':
    model = MANIQA(embed_dim=768,
        num_outputs=1,
        dim_mlp=768,
        patch_size=8,
        img_size=224,
        window_size=4,
        depths=[2,2],
        num_heads=[4,4],
        num_tab=2,
        scale=0.13)
    checkpoint = torch.load('/mnt/yue/YueIQA/output/models/model_maniqa/epoch1.pth',map_location='cpu')
    from collections import OrderedDict
    d = OrderedDict()
    print(type(checkpoint)) # <class 'collections.OrderedDict'>
    for k,v in checkpoint.items():
        k_new = k.replace('module.','') # module.vit.cls_token -> vit.cls_token
        d[k_new] = v
    model.load_state_dict(d)
    x = torch.randn(2, 3, 224, 224).cuda()
    # FPS(model,224)
    model.cuda()
    model.eval()
    # torch.save(model.state_dict(),'/mnt/yue/YueIQA/output/models/model_maniqa/cnm.pth')
    y = model(x) #
    print(y.shape) # torch.Size([2])
    s = torch.squeeze(y)
    print(s.shape)