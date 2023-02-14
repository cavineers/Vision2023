import torch

model = torch.load('/home/sam/Vision2023/training/runs/detect/train11/weights/best.pt')['model'].int().fuse().eval()  # load FP32 model

dummy_input = torch.randn(1, 3, 640, 640)  # (batch_size, channels, height, width)
# # Export to ONNX
torch.onnx.export(model, dummy_input, 'model.onnx', verbose=True, opset_version=11, input_names=['images'], output_names=['output0'],  dynamic_axes={'modelInput' : {0 : 'batch_size'},    # variable length axes 
                                'modelOutput' : {0 : 'batch_size'}})