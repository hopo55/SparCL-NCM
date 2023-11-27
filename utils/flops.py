import torch

def calculate_flops(model, input_shape):
    total_flops = 0
    dummy_input = torch.randn(input_shape).cuda()

    # 모델의 각 레이어를 순회합니다.
    for name, layer in model.named_modules():
        # if hasattr(layer, 'weight') and layer.weight is not None:
        #     # 더미 입력을 현재 레이어의 가중치와 같은 장치로 이동
        #     dummy_input = dummy_input.to(layer.weight.device)

        if isinstance(layer, (torch.nn.Conv2d, torch.nn.Linear)):
        # if isinstance(layer, torch.nn.Conv2d):
            # 컨볼루션 레이어의 경우
            output_shape = layer(dummy_input).shape
            kernel_ops = torch.prod(torch.tensor(layer.weight.shape[1:]))  # 커널 크기 곱
            bias_ops = 1 if layer.bias is not None else 0
            ops_per_element = kernel_ops + bias_ops

            # 희소성을 고려하여 연산량 계산
            sparsity = get_sparsity(layer.weight)
            effective_kernel_ops = kernel_ops * (1 - sparsity)

            # FLOPs 계산
            flops = torch.prod(torch.tensor(output_shape)) * effective_kernel_ops
            total_flops += flops

            dummy_input = torch.randn(output_shape, device=layer.weight.device)

        # elif isinstance(layer, torch.nn.Linear):
            # 선형 레이어의 경우
            weight_ops = layer.weight.numel()
            bias_ops = layer.bias.numel() if layer.bias is not None else 0
            ops_per_element = weight_ops + bias_ops

            # 희소성을 고려하여 연산량 계산
            sparsity = get_sparsity(layer.weight)
            effective_weight_ops = weight_ops * (1 - sparsity)

            # FLOPs 계산
            flops = effective_weight_ops + bias_ops
            total_flops += flops

    return total_flops

def get_sparsity(tensor):
    """
    텐서의 희소성을 계산합니다.
    """
    return float(torch.sum(tensor == 0)) / tensor.numel()

# 예시: 모델의 FLOPs 계산
# model_flops = calculate_flops(model, input_shape)
# print(f"Estimated FLOPs: {model_flops}")
