import argparse
import timeit
import torch
import numpy as np
import os 
import sys
import pandas as pd
import torch.nn as nn

sys.path.append(os.path.join(os.path.dirname(__file__),"..","cs336-basics"))

# step1:   在Terminal:uv run python  uv 会读取项目根目录的 pyproject.toml 文件并安装对应python版本和依赖包，然后自动构建cs336_basics

# step2:  导入assignment2中助教写好的cs336_basics.model 
"""
try:
    from cs336_basics.model import BasicsTransformerLM
except ImportError:
    print("Error: cs336_basics package not found. Please make sure it's installed.")
    sys.exit(1)
"""
# step2:  导入assignment1中自己写好的run_transformer_lm
assignment1_root = "/Users/hcb/Desktop/AI/跟着斯坦福/CS336/assignment1-basics-main"
sys.path.append(assignment1_root)
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
def create_model_with_random_weights(select_config, context_length, use_assignment1=True):
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
        print("Created BasicsTransformerLM")
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


# setp8: 核心计时函数
def run_benchmark(model, context_length, num_warmup_steps, num_measured_steps, mode,device):

    input_data = create_random_data(context_length, BATCH_SIZE, VOCAB_SIZE, device)

    # 如果包含反向传播，需要准备优化器
    optimizer = None
    if mode == "forward_backward":
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    

    # 定义“一个步骤”要做的事情，方便复用
    def run_one_step():
        if mode=="forward":
            with torch.no_grad(): # 前向测试建议开启 no_grad 以免统计多余内存
                _ = model(input_data)
        else:
            optimizer.zero_grad() 
            output = forward_pass(model,input_data)
            loss = output.sum() 
            loss.backward()
            optimizer.step()

    # --- Warm-up 阶段 ---
    # TODO: 循环 num_warmup_steps 次，不记录时间
    for i in range(num_warmup_steps):
        run_one_step()
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elif device.type=='mps':
            torch.mps.synchronize()            

    # --- 测量阶段 ---
    # TODO: 循环 num_measured_steps 次
    records=[]
    for i in range(num_measured_steps):
        time_begin=timeit.default_timer()
        run_one_step()
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elif device.type=='mps':
            torch.mps.synchronize()  
        time_end=timeit.default_timer()
        records.append((time_end-time_begin)*1000)
    return records


# step9: 计算并打印平均值和标准差 (Problem 1.1.3 b)
def calculate_stats(records):
    avg_records=np.mean(records)
    std_records=np.std(records)
    print(f"avg_records: {avg_records:.6f}")
    print(f"std_records: {std_records:.6f}")
    return avg_records,std_records

# step10:panda自动化执行并制表
def run_all_configs(device):
    all_results = []
    
    # 遍历所有模型尺寸
    for size_name in ['small']:
        # 遍历两种模式
        for mode in ["forward"]:
            print(f"Running benchmark for {size_name} - {mode}...")
            
            try:
                # 1. 初始化当前尺寸的模型
                config = MODEL_CONFIGS[size_name]
                device = torch.device(device) # 你在 Mac 上用 mps
                model = create_model_with_random_weights(config, context_length=512)
                model.to(device)
                
                # 2. 运行跑分 
                records = run_benchmark(model, 512, 5, 10, mode, device)
                avg_ms, std_ms = np.mean(records), np.std(records)
                
                # 3. 将结果存入列表
                all_results.append({
                    "Model Size": size_name,
                    "Mode": mode,
                    "Avg (ms)": round(avg_ms, 2),
                    "Std (ms)": round(std_ms, 2)
                })
                
                # 及时清理显存
                del model
                torch.mps.empty_cache() 
                
            except Exception as e:
                # 如果 2.7B 内存溢出了，记录为 OOM
                print(f"Failed to run {size_name}: {e}")
                all_results.append({
                    "Model Size": size_name,
                    "Mode": mode,
                    "Avg (ms)": "OOM",
                    "Std (ms)": "N/A"
                })

    # 4. 转换为 DataFrame
    df = pd.DataFrame(all_results)

    return df
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=str, choices=MODEL_CONFIGS.keys(), default="small")
    parser.add_argument("--context_length", type=int, default=512)
    parser.add_argument("--mode", type=str, choices=["forward", "forward_backward"], default="forward")
    parser.add_argument("--num_warmup", type=int, default=5)
    parser.add_argument("--num_steps", type=int, default=10)
    parser.add_argument("--device", type=str, choices=["cuda", "mps","cpu"], default="cpu")
    args = parser.parse_args()

    # 1. 初始化模型并根据指令设备选择硬件
    #比如想要使用cpu执行的话就是uv run python benchmarking.py --size small --mode forward --device cpu
    device = torch.device(args.device) # 将字符串转换为 torch.device 对象
    
    #如果想一个具体的模型配置执行，可以采用下面的语句
    """
    select_config=MODEL_CONFIGS[args.size]
    model = create_model_with_random_weights(select_config,args.context_length)
    model.to(device)
    # 2. 调用 run_benchmark
    records=run_benchmark(model, args.context_length, args.num_warmup, args.num_steps, args.mode,device)
    avg_records,std_records=calculate_stats(records)
    # 3. 输出结果
    print(f"context_length: {args.context_length}")
    print(f"num_warmup: {args.num_warmup}")
    print(f"num_steps: {args.num_steps}")
    print(f"mode: {args.mode}")
    print(f"size: {args.size}")
    print(f"device: {device}")
    """

    #如果想自动化执行所有模型大小的跑分，可以释放下面的注释
    
    df_results = run_all_configs(device)
    print("\n--- Final Results Table ---")
    print(df_results.to_markdown())
    # 将数据透视，让表格更好看
    pivot_df = df_results.pivot(index="Model Size", columns="Mode", values="Avg (ms)")
    # 按照预定义的顺序排序（否则 pandas 会按字母排序，small 会排在 medium 后面）
    size_order = ['small', 'medium', 'large', 'xl', '2.7B']
    pivot_df = pivot_df.reindex(size_order)

    print("\n--- Pivot Table for Writeup ---")
    print(pivot_df.to_markdown())

    # 1. 保存原始数据（方便后续查阅）
    df_results.to_csv("benchmark_results_raw.csv", index=False)
    
    # 2. 保存透视后的汇总表（这就是你写报告要用的表）
    pivot_df.to_csv("benchmark_summary_pivot.csv")

    print("\n[Success] 表格已自动保存至当前文件夹:")
    print("- benchmark_results_raw.csv")
    print("- benchmark_summary_pivot.csv")
    

if __name__ == "__main__":
    main()


"""
Q1:PDF 专门问了关于 Warm-up 的问题。你需要做一个对比：
使用 --num_warmup 5(正常情况)
使用 --num_warmup 0(不预热)
观察点：如果不预热，第一步的时间通常会非常长（因为 GPU 需要初始化 Kernel),这会导致平均值被拉高。
uv run python benchmarking.py --size small --mode forward --num_warmup 0 --device mps 

avg_records: 280.622967
std_records: 66.030199
context_length: 512
num_warmup: 0
num_steps: 10
mode: forward
size: small
device: mps


uv run python benchmarking.py --size small --mode forward --num_warmup 5 --device mps
avg_records: 259.492258
std_records: 1.071463
context_length: 512
num_warmup: 5
num_steps: 10
mode: forward
size: small
device: mps


Q2:Forward 和 Backward 分别花了多少时间？标准差是否很小？
Running benchmark for small - forward...
Running benchmark for small - forward_backward...
Running benchmark for medium - forward...
Running benchmark for medium - forward_backward...

--- Final Results Table ---
|    | Model Size   | Mode             |   Avg (ms) |   Std (ms) |
|---:|:-------------|:-----------------|-----------:|-----------:|
|  0 | small        | forward          |     259.64 |       0.75 |
|  1 | small        | forward_backward |     793.81 |       1.22 |
|  2 | medium       | forward          |     766.87 |       4.38 |
|  3 | medium       | forward_backward |    8616.75 |    7933.04 |

--- Pivot Table for Writeup ---
| Model Size   |   forward |   forward_backward |
|:-------------|----------:|-------------------:|
| small        |    259.64 |             793.81 |
| medium       |    766.87 |            8616.75 |
| large        |    nan    |             nan    |
| xl           |    nan    |             nan    |
| 2.7B         |    nan    |             nan    |


small model forward_backward ≈ 3 times slower than forward 并且计算波动小
medium model forward_backward ≈ 11 times slower than forward 并且计算波动大,这反映了在本地统一内存环境下,大模型计算触发了系统的内存交换机制(Swap)或热限制,导致计算延迟出现了显著的非线性波动

"""