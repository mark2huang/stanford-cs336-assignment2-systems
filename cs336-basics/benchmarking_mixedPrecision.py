import torch
import os

s = torch.tensor(0,dtype=torch.float32)
for i in range(1000):
    s += torch.tensor(0.01,dtype=torch.float32)
print(s)

s = torch.tensor(0,dtype=torch.float16)
for i in range(1000):
    s += torch.tensor(0.01,dtype=torch.float16)
print(s)

s = torch.tensor(0,dtype=torch.float32)
for i in range(1000):
    s += torch.tensor(0.01,dtype=torch.float16)
print(s)

s = torch.tensor(0,dtype=torch.float32)
for i in range(1000):
    x = torch.tensor(0.01,dtype=torch.float16)
    s += x.type(torch.float32)
print(s)


"""
案例1是传统FP32计算,精度高但慢

案例2是从FP32降到FP16,当累加到某个点,再加一个很小的数,由于FP16精度不足以表示微小的增量,会发生“下溢”或被舍入

案例3和案例4是混合精度,我们在训练时,可以使用 FP16 计算梯度
但在更新参数时，必须把梯度加回到 FP32 的主权重上，只有这样，那些微小的梯度更新才不会被“舍入”掉，模型才能学到东西。
"""





"""
with torch.autocast(device="cuda", dtype=torch.float16):
    # 1. 矩阵乘法
    a = torch.matmul(Q, K) 
    # 2. 加上一个偏置
    b = a + bias
    # 3. 计算 Softmax
    c = torch.softmax(b, dim=-1)
autocast(dtype=torch.float16/torch.float32)函数的含义是:当系统决定要降精度时,请降到这个特定的类型(FP16 还是 BF16)”。
而具体“哪些算子要降”,是由 autocast 内部的清单决定。

白名单(自动转为低精度)：主要是矩阵乘法(matmul)、卷积(conv2d)、线性层(linear)。这些操作在 Tensor Cores 上跑得飞快，且对微小的精度损失不敏感。

黑名单(强制留在 FP32)：主要是对数值范围敏感的函数，比如 softmax、layer_norm、exp、log、pow。这些函数如果用 FP16 跑，非常容易出现 NaN(溢出)或者结果完全错误。

中立操作：比如 add(加法)。它们通常遵循输入张量的类型，如果输入有一个是 FP32,结果就是 FP32。
"""



import torch.nn as nn
class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.ln(x)
        x = self.fc2(x)
        return x
    




import argparse
import timeit
import torch
import numpy as np
import sys
import pandas as pd
import torch.nn as nn
import os 
from contextlib import nullcontext

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


# setp8: 核心计时函数
def run_benchmark(model, context_length, num_warmup_steps, num_measured_steps, mode,device,mixedPrecision):

    input_data = create_random_data(context_length, BATCH_SIZE, VOCAB_SIZE, device)

    #确定精度上下文
    if mixedPrecision=='true':
        if device.type=='cuda':
            device_type='cuda'
        else:
            device_type='cpu'
        autocast_ctx=torch.autocast(device_type=device_type,dtype=torch.bfloat16)
    else:
        autocast_ctx=nullcontext()

    # 如果包含反向传播，需要准备优化器
    optimizer = None
    if mode == "forward_backward":
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    

    # 定义“一个步骤”要做的事情，方便复用
    def run_one_step():
        if mode=="forward":
            with torch.no_grad(): # 前向测试开启 no_grad 以免统计多余内存
                with autocast_ctx: #开启混合精度
                    _ = model(input_data)
        else:
            optimizer.zero_grad() 
            with autocast_ctx: #开启混合精度
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
                device = torch.device(device) 
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
    parser.add_argument("--mixedPrecision",type=str,choices=["true","false"],default="false")
    args = parser.parse_args()

    #手动指定执行具体的模型
    #比如想要使用cpu执行的话就是uv run python benchmarking.py --size small --mode forward --device cpu
    device = torch.device(args.device) # 将字符串转换为 torch.device 对象
    select_config=MODEL_CONFIGS[args.size]
    model = create_model_with_random_weights(select_config,args.context_length)
    model.to(device)
    records=run_benchmark(model, args.context_length, args.num_warmup, args.num_steps, args.mode,device,args.mixedPrecision)
    avg_records,std_records=calculate_stats(records)
    print(f"context_length: {args.context_length}")
    print(f"num_warmup: {args.num_warmup}")
    print(f"num_steps: {args.num_steps}")
    print(f"mode: {args.mode}")
    print(f"size: {args.size}")
    print(f"device: {device}")
    print(f"device.type={device.type}")
    

    #自动化执行所有模型大小的跑分
    """
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
    """
    

if __name__ == "__main__":
    main()




"""
Writeup:
Suppose we are training the model on a GPU and that the model parameters are originally in
FP32. We’d like to use autocasting mixed precision with FP16. What are the data types of:
• the model parameters within the autocast context,FP32     注：当代码运行到第 10 行执行 fc1(x)（矩阵乘法）时，它会在后台临时创建一个参数的 FP16 副本进行计算。计算一结束，这个副本就被丢弃了。从用户的视角看，或者从 model.parameters() 的属性看，参数永远是 FP32。它不会在模型里持久化存一份 FP16 的 weight。
• the output of the first feed-forward layer (ToyModel.fc1),FP16
• the output of layer norm (ToyModel.ln),FP32
• the model’s predicted logits,FP32
• the loss,FP32
• and the model’s gradients? FP32
Deliverable: The data types for each of the components listed above.



b) You should have seen that FP16 mixed precision autocasting treats the layer normalization layer
differently than the feed-forward layers. What parts of layer normalization are sensitive to mixed
precision? If we use BF16 instead of FP16, do we still need to treat layer normalization differently?
Why or why not?
Deliverable: A 2-3 sentence response.
An:
不能使用FP16,因为layerNorm其中要计算向量的方差,涉及平方的操作,如果使用FP16,容易出现数值溢出的问题
也不能使用BP16,虽然 BF16 的动态范围和 FP32 一样大，不容易溢出，但 BF16 的精度（尾数位）非常低（只有 7 位)。在计算均值和方差这种对精确度要求极高的统计量时,BF16 可能会引入不可忽视的噪声。

(c) Modify your benchmarking script to optionally run the model using mixed precision with BF16.
Time the forward and backward passes with and without mixed-precision for each language model
size described in §1.1.2. Compare the results of using full vs. mixed precision, and comment on
any trends as model size changes. You may find the nullcontext no-op context manager to be
useful.
Deliverable: A 2-3 sentence response with your timings and commentary.
随着模型尺寸（从 small 到 xl)的增加,你会发现：
加速比增加:大模型中矩阵乘法(Matmul)占总计算量的比例更高,BF16 能通过 Tensor Cores 带来更显著的吞吐量提升。
显存节省：虽然参数仍是 FP32,但**激活值(Activations)**变为了 16 位，显著降低了显存压力，使得在 FP32 下 OOM 的长序列在 BF16 下可以运行。

"""


