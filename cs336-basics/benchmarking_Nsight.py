import argparse
import timeit
import torch
import numpy as np
import os 
import sys
import pandas as pd
import torch.cuda.nvtx as nvtx

# 尝试导入作业库
try:
    from cs336_basics.model import BasicsTransformerLM
    import cs336_basics.model as model_module
except ImportError:
    print("Error: cs336_basics package not found. Make sure it's in your PYTHONPATH.")
    sys.exit(1)

# --- 1. NVTX 注入逻辑 ---

def get_nvtx_range(name):
    """便捷的 NVTX 范围包装器"""
    return torch.cuda.nvtx.range(name)

def instrument_attention():
    """
    为了回答问题 (e)，我们需要深入拦截注意力机制内部的操作。
    它会替换掉 cs336_basics.model 里的 run_scaled_dot_product_attention。
    """
    original_sdpa = model_module.run_scaled_dot_product_attention

    def annotated_sdpa(Q, K, V, mask=None):
        # 整体 SDPA 范围
        with nvtx.range("sdpa_total"):
            # 问题 (e): 测量 QK^T 的矩阵乘法
            with nvtx.range("attention_matmul_qk"):
                # 注意：这里的计算逻辑应与你作业实现的原始逻辑一致
                scores = torch.matmul(Q, K.transpose(-2, -1)) / (Q.size(-1) ** 0.5)
            
            if mask is not None:
                scores = scores.masked_fill(~mask, float('-inf'))
                
            # 问题 (e): 专门测量 Softmax 操作
            with nvtx.range("attention_softmax"):
                probs = torch.softmax(scores, dim=-1)
                
            # 问题 (e): 测量 AV 的矩阵乘法
            with nvtx.range("attention_matmul_av"):
                output = torch.matmul(probs, V)
                
            return output

    # 动态替换模块中的函数
    model_module.run_scaled_dot_product_attention = annotated_sdpa
    print("✅ 已成功注入 NVTX 标签到 Attention 操作 (用于回答问题 e)")

# --- 2. 模型配置 ---

MODEL_CONFIGS = {
    'small': {'d_model': 768, 'd_ff': 3072, 'num_layers': 12, 'num_heads': 12},
    'medium': {'d_model': 1024, 'd_ff': 4096, 'num_layers': 24, 'num_heads': 16},
    'large': {'d_model': 1280, 'd_ff': 5120, 'num_layers': 36, 'num_heads': 20},
    'xl': {'d_model': 1600, 'd_ff': 6400, 'num_layers': 48, 'num_heads': 25},
    '2.7B': {'d_model': 2560, 'd_ff': 10240, 'num_layers': 32, 'num_heads': 32}
}

def create_model(size_key, context_length, device):
    config = MODEL_CONFIGS[size_key]
    model = BasicsTransformerLM(
        vocab_size=10000,
        context_length=context_length,
        **config
    ).to(device)
    
    # 包装 forward 方法以回答问题 (a)
    original_forward = model.forward
    def wrapped_forward(*args, **kwargs):
        with nvtx.range("forward_pass"):
            return original_forward(*args, **kwargs)
    model.forward = wrapped_forward
    
    return model

# --- 3. 核心运行逻辑 ---

def run_benchmark(model, context_length, num_warmup, num_steps, mode, device):
    # 准备数据
    input_data = torch.randint(0, 10000, (4, context_length), device=device)
    
    # 如果是训练模式，准备优化器
    optimizer = None
    if mode == "forward_backward":
        # 使用 AdamW 以回答问题 (d)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    print(f"开始 {num_warmup} 次预热...")
    for _ in range(num_warmup):
        if mode == "forward":
            with torch.no_grad():
                _ = model(input_data)
        else:
            optimizer.zero_grad()
            output = model(input_data)
            loss = output.sum()
            loss.backward()
            optimizer.step()
    
    if device.type == 'cuda': torch.cuda.synchronize()

    print(f"开始执行测量步骤 ({num_steps} 次)...")
    durations = []
    
    for i in range(num_steps):
        # 这一行标记能让 nsys 自动识别每一轮迭代
        with nvtx.range(f"step_{i}"):
            start_time = timeit.default_timer()
            
            if mode == "forward":
                with torch.no_grad():
                    _ = model(input_data)
            else:
                # 完整训练步 (回答问题 b, d)
                optimizer.zero_grad()
                
                # 前向已经在 model.forward 里标记了
                output = model(input_data)
                
                with nvtx.range("compute_loss"):
                    loss = output.sum()
                
                with nvtx.range("backward_pass"):
                    loss.backward()
                
                with nvtx.range("optimizer_step"):
                    optimizer.step()
            
            if device.type == 'cuda': torch.cuda.synchronize()
            durations.append((timeit.default_timer() - start_time) * 1000)
            
    return durations

# --- 4. 主函数 ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=str, default="small", choices=MODEL_CONFIGS.keys())
    parser.add_argument("--context_length", type=int, default=512)
    parser.add_argument("--mode", type=str, default="forward", choices=["forward", "forward_backward"])
    parser.add_argument("--num_warmup", type=int, default=3)
    parser.add_argument("--num_steps", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    
    # 注入 NVTX
    instrument_attention()

    # 初始化模型
    print(f"正在初始化模型: {args.size}, context_length: {args.context_length}...")
    try:
        model = create_model(args.size, args.context_length, device)
    except torch.cuda.OutOfMemoryError:
        print(f"❌ OOM: 模型 {args.size} 在长度 {args.context_length} 下内存溢出")
        return

    # 运行基准测试
    durations = run_benchmark(
        model, args.context_length, args.num_warmup, args.num_steps, args.mode, device
    )

    # 打印结果 (回答问题 a)
    avg_ms = np.mean(durations)
    std_ms = np.std(durations)
    print(f"\n--- 测量结果 ---")
    print(f"模型尺寸: {args.size}")
    print(f"上下文长度: {args.context_length}")
    print(f"模式: {args.mode}")
    print(f"平均耗时: {avg_ms:.2f} ms")
    print(f"标准差: {std_ms:.2f} ms")

if __name__ == "__main__":
    main()