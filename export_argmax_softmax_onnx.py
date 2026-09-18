import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import models


class SegWrapper(nn.Module):
   """Wraps PIDNet to produce the two outputs required by the pipeline:
   - labels: argmax class index per pixel  [B, H, W]  (int64)
   - masks:  softmax probability per class [B, num_classes, H, W]  (float32)
   """
   def __init__(self, backbone):
       super().__init__()
       self.backbone = backbone

   def forward(self, x):
       logits = self.backbone(x)          # [B, C, H, W]
       masks = F.softmax(logits, dim=1)   # [B, C, H, W]
       labels = torch.argmax(logits, dim=1)  # [B, H, W]
       return labels, masks


def parse_args():
   parser = argparse.ArgumentParser(description='Export PIDNet to ONNX')
   parser.add_argument('--a', help='pidnet-s, pidnet-m or pidnet-l', default='pidnet_s', type=str)
   parser.add_argument('--p', help='path to trained .pt checkpoint',
                       default='pretrained_models/cityscapes/PIDNet_S_Simple_Cityscapes_test.pt', type=str)
   parser.add_argument('--num-classes', type=int, default=19)
   parser.add_argument('--height', type=int, default=512)
   parser.add_argument('--width', type=int, default=512)
   parser.add_argument('--o', help='output .onnx path',
                       default='output/holeseg/pidnet_small_holeseg/pretrained_dynamic.onnx', type=str)
   parser.add_argument('--dynamic', action='store_true',
                       help='export with dynamic batch size')
   parser.add_argument('--batch_size', type=int, default=4)
   parser.add_argument('--opset', type=int, default=18)
   return parser.parse_args()


def load_pretrained(model, checkpoint_path):
   pretrained_dict = torch.load(checkpoint_path, map_location='cpu')
   if 'state_dict' in pretrained_dict:
       pretrained_dict = pretrained_dict['state_dict']
   model_dict = model.state_dict()
   pretrained_dict = {
       k[6:]: v for k, v in pretrained_dict.items()
       if k[6:] in model_dict and v.shape == model_dict[k[6:]].shape
   }
   print(f'Loaded {len(pretrained_dict)} parameters from {checkpoint_path}')
   model_dict.update(pretrained_dict)
   model.load_state_dict(model_dict, strict=False)
   return model


def main():
   args = parse_args()

   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   print(f'Using device: {device}')

   backbone = models.pidnet.get_pred_model(args.a, args.num_classes)
   backbone = load_pretrained(backbone, args.p)
   model = SegWrapper(backbone)
   model.to(device)
   model.eval()

   dummy_input = torch.randn(args.batch_size, 3, args.height, args.width, device=device)

   dynamic_axes = None
   if args.dynamic:
       dynamic_axes = {
           'input':  {0: 'batch'},
           'labels': {0: 'batch'},
           'masks':  {0: 'batch'},
       }

   os.makedirs(os.path.dirname(os.path.abspath(args.o)), exist_ok=True)

   with torch.no_grad():
       torch.onnx.export(
           model,
           dummy_input,
           args.o,
           opset_version=args.opset,
           input_names=['input'],
           output_names=['labels', 'masks'],
           dynamic_axes=dynamic_axes,
       )

   print(f'ONNX model saved to: {args.o}')

   try:
       import onnx
       onnx_model = onnx.load(args.o)
       onnx.checker.check_model(onnx_model)
       print('ONNX model check passed.')
   except ImportError:
       print('onnx package not installed — skipping validation.')


if __name__ == '__main__':
   main()
