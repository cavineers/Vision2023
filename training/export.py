import torch

model = torch.load('training/runs/detect/train7/weights/best.pt', map_location=torch.device('cpu'))['model'].float().fuse().eval()  # load FP32 model

dummy_input = torch.randn(1, 3, 640, 640)  # (batch_size, channels, height, width)
# # Export to ONNX
torch.onnx.export(model, dummy_input, 'model.onnx', verbose=True, opset_version=11, input_names=['images'], output_names=['output0'],  dynamic_axes={'modelInput' : {0 : 'batch_size'},    # variable length axes 
                                'modelOutput' : {0 : 'batch_size'}})