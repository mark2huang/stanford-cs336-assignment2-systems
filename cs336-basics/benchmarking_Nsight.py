import argparse
import timeit
import torch
import numpy as np
import os 
import sys
import pandas as pd
import sys
import os
import torch.cuda.nvtx as nvtx
if "/content/cs336-basics" not in sys.path:
    sys.path.append("/content/cs336-basics")


# step1:   在Colab Terminal:uv run python  uv 会读取项目根目录的 pyproject.toml 文件并安装对应python版本和依赖包，然后自动构建cs336_basics

# step2:  导入cs336_basics.model 
try:
    from cs336_basics.model import BasicsTransformerLM
except ImportError:
    print("Error: cs336_basics package not found. Please make sure it's installed.")
    sys.exit(1)


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


# step5:自定义模型
def create_model_with_random_weights(select_config,context_length):
    model=BasicsTransformerLM(vocab_size=VOCAB_SIZE,
                              context_length=context_length,
                              d_model=select_config['d_model'],
                              num_heads=select_config['num_heads'],
                              num_layers=select_config['num_layers'],
                              d_ff=select_config['d_ff'],
                              rope_theta=10000.0)
    return model




# step6:定义模型前向传播
def forward_pass(model,input_data):
    output=model(input_data)
    return output

# setp7: 创建数据随机批次
def create_random_data(context_length, batch_size=BATCH_SIZE, vocab_size=VOCAB_SIZE, device='cpu'):
    """
    # 生成一个形状为 (4, context_length) 的矩阵
    # 里面的每个数字都是 0 到 9999 之间的随机整数
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
    for size_name in ['small','medium']:
        # 遍历两种模式
        for mode in ["forward", "forward_backward"]:
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

# --- Nsight定义带标签的函数 ---
def annotated_attention(Q, K, V, mask=None):
    with nvtx.range("scaled_dot_product_attention"):
        with nvtx.range("compute_attention_scores"):
            # 注意：这里的逻辑要和你原始 model.py 里的实现一致
            scores = torch.matmul(Q, K.transpose(-2, -1)) / (Q.size(-1) ** 0.5)
        with nvtx.range("compute_softmax"):
            probs = torch.softmax(scores, dim=-1)
        with nvtx.range("final_matmul"):
            output = torch.matmul(probs, V)
        return output
    
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
    
    #如果想一条一条指令执行，可以释放下面的注释
    select_config=MODEL_CONFIGS[args.size]

    model = create_model_with_random_weights(select_config,args.context_length)
    model.to(device)
    try:
        import cs336_basics.model
        cs336_basics.model.scaled_dot_product_attention = annotated_attention
        print("✅ NVTX 标签已成功注入 cs336_basics.model")
    except Exception as e:
        print(f"❌ 注入 NVTX 失败: {e}")
    
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
    

    #如果想自动化执行，可以释放下面的注释
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
测试步骤：

# 1. 检查 GPU
!nvidia-smi

# 2. 检查 nsys 是否安装(Colab 通常预装了)

# 3. 更新包管理器并安装 Nsight Systems
!apt-get update -y
!apt-get install -y nsight-systems-cli

# 4. 验证安装
!nsys --version

# 5. 运行基准测试并生成 Nsight Systems 报告
!PYTHONPATH=$PYTHONPATH:/content/cs336-basics nsys profile \
    -o result_medium_fwd \
    -f true \
    -t cuda,nvtx,osrt \
    python cs336-basics/benchmarking_nvtx.py --size medium --mode forward --device cuda --num_warmup 5 --num_steps 1

# 6. 查看生成的报告
!nsys-ui result_medium_fwd.qdrep

# 7. 分析报告
# 打开 Nsight Systems UI 后，根据报告中的指标（如 GPU 利用率、内存使用等）分析模型性能并改进测试。
"""