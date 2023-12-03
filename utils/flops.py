import torch
import torch.nn as nn
import torch.nn.functional as F
# from ..models.resnet18_cifar import resnet18
# from models.resnet18_cifar import NCM, resnet18

def calculate_flops(model, input_shape):
    total_flops = 0
    dummy_input = torch.randn(input_shape).cuda()

    for name, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d):
            in_channels = layer.in_channels
            dummy_input = torch.randn(1, in_channels, *input_shape[2:]).cuda()

            try:
                output = layer(dummy_input)
                output_shape = output.shape
                kernel_ops = torch.prod(torch.tensor(layer.weight.shape[1:]))
                bias_ops = 1 if layer.bias is not None else 0
                effective_kernel_ops = kernel_ops * (1 - get_sparsity(layer.weight))
                flops = torch.prod(torch.tensor(output_shape)) * effective_kernel_ops
                total_flops += flops

                dummy_input = torch.randn(output_shape, device=layer.weight.device)

            except RuntimeError as e:
                print(f"Error in layer {name}: {e}")

        elif isinstance(layer, nn.Linear):
            # Flatten dummy_input for Linear layers
            dummy_input_flat = dummy_input.view(dummy_input.size(0), -1)
            output = layer(dummy_input_flat)
            output_shape = output.shape

            weight_ops = layer.weight.numel()
            bias_ops = layer.bias.numel() if layer.bias is not None else 0
            flops = weight_ops + bias_ops
            total_flops += flops

            dummy_input = torch.randn(output_shape, device=layer.weight.device)

    return total_flops

def get_sparsity(tensor):
    return float(torch.sum(tensor == 0)) / tensor.numel()