from model import Generator
import torch
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--weights', default='./weights/stylegan2-ffhq-config-f.pt')
parser.add_argument('--device', default='cpu', type=str, help='cuda or cpu')
parser.add_argument('--size', type=int, default=1024, help='output image size of the generator')
opt = parser.parse_args()

device = torch.device(opt.device)

model = Generator(opt.size, 512, 8)
checkpoint = torch.load(opt.weights, weights_only=False)
model.load_state_dict(checkpoint['g_ema'])

model.to(device)
model.eval()

model_path = os.path.splitext(opt.weights)[0] + '.onnx'

dlatent = torch.randn(1, 1, 512)

torch.onnx.export(
    model,
    (dlatent,),
    model_path,
    verbose=False,
    input_names=['input'],
    output_names=['output'],
    opset_version=10
)
