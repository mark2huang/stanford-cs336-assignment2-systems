import argparse
import timeit
import torch
import numpy as np
import sys
import pandas as pd
import torch.nn as nn
import os 
import torch.cuda.memory as memory



sys.path.append(os.path.join(os.path.dirname(__file__),"..","cs336-basics"))

# step1:   在Terminal:uv run python  uv 会读取项目根目录的 pyproject.toml 文件并安装对应python版本和依赖包，然后自动构建cs336_basics

# step2:  导入assignment2中助教写好的cs336_basics.model 

try:
    from cs336_basics.model import BasicsTransformerLM
except ImportError:
    print("Error: cs336_basics package not found. Please make sure it's installed.")
    sys.exit(1)

# step2:  导入assignment1中自己写好的run_transformer_lm
assignment1_root = "/Users/hcb/Desktop/AI/跟着斯坦福/CS336/assignment1-basics-main"
sys.path.insert(0,assignment1_root)
print(f"sys.path={sys.path}")
from tests.adapters import run_transformer_lm

# --- 包装类 ---
class MyAssignment1Model(nn.Module):
    def __init__(self, vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta):
        super().__init__()
        self.weights = nn.ParameterDict()
        self.config = {
            'vocab_size': vocab_size, 'context_length': context_length,
            'd_model': d_model, 'num_layers': num_layers,
            'num_heads': num_heads, 'd_ff': d_ff, 'rope_theta': rope_theta
        }
        
        # 定义一个辅助函数来处理命名
        def safe_name(name):
            return name.replace(".", "__") # 将点替换为双下划线
        
        # --- 初始化权重 ---
        # Token Embeddings
        self.weights[safe_name('token_embeddings.weight')] = nn.Parameter(torch.randn(vocab_size, d_model))
        
        # Layers
        for i in range(num_layers):
            # 注意这里所有的 key 都通过 safe_name 转换了
            self.weights[safe_name(f'layers.{i}.attn.q_proj.weight')] = nn.Parameter(torch.randn(d_model, d_model))
            self.weights[safe_name(f'layers.{i}.attn.k_proj.weight')] = nn.Parameter(torch.randn(d_model, d_model))
            self.weights[safe_name(f'layers.{i}.attn.v_proj.weight')] = nn.Parameter(torch.randn(d_model, d_model))
            self.weights[safe_name(f'layers.{i}.attn.output_proj.weight')] = nn.Parameter(torch.randn(d_model, d_model))
            self.weights[safe_name(f'layers.{i}.ln1.weight')] = nn.Parameter(torch.ones(d_model))
            self.weights[safe_name(f'layers.{i}.ffn.w1.weight')] = nn.Parameter(torch.randn(d_ff, d_model))
            self.weights[safe_name(f'layers.{i}.ffn.w2.weight')] = nn.Parameter(torch.randn(d_model, d_ff))
            self.weights[safe_name(f'layers.{i}.ffn.w3.weight')] = nn.Parameter(torch.randn(d_ff, d_model))
            self.weights[safe_name(f'layers.{i}.ln2.weight')] = nn.Parameter(torch.ones(d_model))
            
        # Final Norm & Head
        self.weights[safe_name('ln_final.weight')] = nn.Parameter(torch.ones(d_model))
        self.weights[safe_name('lm_head.weight')] = nn.Parameter(torch.randn(vocab_size, d_model))
        
    def forward(self, in_indices):
        # 在执行前，把双下划线还原回点号，构造出作业 1 函数需要的 weights 字典
        dotted_weights = {k.replace("__", "."): v for k, v in self.weights.items()}
        
        return run_transformer_lm(
            **self.config,
            weights=dotted_weights, # 传入还原后的字典
            in_indices=in_indices
        )
    


# step3:按cs336_spring2025_assignment2_systems.pdf中1.1.2配置model大小
MODEL_CONFIGS = {
    'small': {'d_model': 768, 'd_ff': 3072, 'num_layers': 12, 'num_heads': 12},
    'medium': {'d_model': 1024, 'd_ff': 4096, 'num_layers': 24, 'num_heads': 16},
    'large': {'d_model': 1280, 'd_ff': 5120, 'num_layers': 36, 'num_heads': 20},
    'xl': {'d_model': 1600, 'd_ff': 6400, 'num_layers': 48, 'num_heads': 25},
    '2.7B': {'d_model': 2560, 'd_ff': 10240, 'num_layers': 32, 'num_heads': 32}
}


# step4:按cs336_spring2025_assignment2_systems.pdf中1.1.2配置训练语料大小
VOCAB_SIZE = 10000
BATCH_SIZE = 4


# step5:定义创建模型函数
def create_model_with_random_weights(select_config, context_length, use_assignment1=False):
    if use_assignment1:
        print("Created MyAssignment1Model")
        return MyAssignment1Model(
            VOCAB_SIZE, context_length, 
            select_config['d_model'], select_config['num_layers'],
            select_config['num_heads'], select_config['d_ff'], 10000.0
        )
    else:
        # 原有的作业 2 助教模型
        from cs336_basics.model import BasicsTransformerLM
        print("Created BasicsTransformerLM from cs336 instructor")
        return BasicsTransformerLM(vocab_size=VOCAB_SIZE,
                              context_length=context_length,
                              d_model=select_config['d_model'],
                              num_heads=select_config['num_heads'],
                              num_layers=select_config['num_layers'],
                              d_ff=select_config['d_ff'],
                              rope_theta=10000.0)
        


# step6:定义模型前向传播
def forward_pass(model,input_data):
    output=model(input_data)
    return output

# setp7: 创建数据随机批次
def create_random_data(context_length, batch_size=BATCH_SIZE, vocab_size=VOCAB_SIZE, device='cpu'):
    """
    # 生成一个形状为 (bacth_size, context_length) 的矩阵
    # 里面的每个数字都是 0 到 VOCAB_SIZE 之间的随机整数
    参数:
        context_length: 序列长度(token数量)
        batch_size: 批次大小
        vocab_size: 词汇表大小
        device: 设备(cpu或cuda)
    
    返回:
        torch.Tensor: 形状为(batch_size, context_length)的随机整数张量
    """
    # 生成随机整数张量，范围[0, vocab_size)
    return torch.randint(low=0, high=vocab_size, size=(batch_size, context_length), device=device)


# setp8: 核心内存分析函数
def run_benchmark(model, context_length, num_warmup_steps, mode,device,is_mixed_precision=False,profile_memory=False):

    input_data = create_random_data(context_length, BATCH_SIZE, VOCAB_SIZE, device)

    # 如果包含反向传播，需要准备优化器
    optimizer = None
    if mode == "forward_backward":
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    

    # 定义“一个步骤”要做的事情，方便复用
    def run_one_step():
        from contextlib import nullcontext
        context = torch.autocast(device_type='cuda', dtype=torch.bfloat16) if is_mixed_precision else nullcontext()
        if mode=="forward":
            with torch.no_grad(): # 前向测试建议开启 no_grad 以免统计多余内存
                with context:
                    _ = model(input_data)
        else:
            optimizer.zero_grad()
            with context: 
                output = forward_pass(model,input_data)
                loss = output.sum() 
            loss.backward()
            optimizer.step()

    # --- Warm-up 阶段 ---
    # TODO: 循环 num_warmup_steps 次，不记录时间
    for _ in range(num_warmup_steps):
        run_one_step()
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elif device.type=='mps':
            torch.mps.synchronize()            

    # --- memory测量阶段 注意：只跑 1-2 步，让生成的图表更易读，内存分析模式下，计时不重要，这是和benchmarking.py的区别---
    if profile_memory and device.type=='cuda':
        print("Start recording memory history.")
        torch.cuda.memory._record_memory_history(max_entries=1000000) 
        for _ in range(1):
            run_one_step()
        torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")# Save a pickle file to be loaded by PyTorch's online tool.
        torch.cuda.memory._record_memory_history(enabled=None) # Stop recording history.
        print("Memory Snapshot Saved!")
        return [0]
    return [0]




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=str, choices=MODEL_CONFIGS.keys(), default="small")
    parser.add_argument("--context_length", type=int, default=512)
    parser.add_argument("--mode", type=str, choices=["forward", "forward_backward"], default="forward")
    parser.add_argument("--num_warmup", type=int, default=5)
    parser.add_argument("--device", type=str, choices=["cuda", "mps","cpu"], default="cpu")
    args = parser.parse_args()

    #手动指定执行具体的模型
    #比如想要使用cpu执行的话就是uv run python benchmarking.py --size small --mode forward --device cpu
    device = torch.device(args.device) # 将字符串转换为 torch.device 对象
    select_config=MODEL_CONFIGS[args.size]
    model = create_model_with_random_weights(select_config,args.context_length)
    model.to(device)
    run_benchmark(model, args.context_length, args.num_warmup,args.mode,device,is_mixed_precision=True,profile_memory=True)
    print(f"context_length: {args.context_length}")
    print(f"num_warmup: {args.num_warmup}")
    print(f"mode: {args.mode}")
    print(f"size: {args.size}")
    print(f"device: {device}")
    

  
    

if __name__ == "__main__":
    main()


