import os
import torch.nn.functional as F

def pytorch_naive_attention(q, k, v, mask=None):
    """
    实现朴素的 PyTorch Attention: softmax(Q @ K.T / sqrt(dk)) @ V
    
    参数:
        q: Query 张量，形状为 (batch_size, seq_len, d_model)
        k: Key 张量，形状为 (batch_size, seq_len, d_model)
        v: Value 张量，形状为 (batch_size, seq_len, d_model)
        mask: 可选的掩码张量 (例如 causal mask)
        
    返回:
        output: 注意力计算后的输出，形状为 (batch_size, seq_len, d_model)
    """
    #step1:calculate attentionScore
    attentionScore=q@k.transpose(-2,-1)

    #step2:normalize
    d_k=q.shape[-1]
    attentionScore=attentionScore/d_k**0.5

    #step3:mask
    if mask is not None:
        attentionScore=attentionScore.masked_fill(~mask,float('-inf'))

    #step4:softmax
    attentionScore=torch.softmax(attentionScore,dim=-1)

    #step5:calculate output
    output=attentionScore@v
    return output



import time
import torch
import pandas as pd
import numpy as np

def benchmark_attention():
    batch_size = 8
    d_model_list = [16, 32, 64, 128]
    #seq_len_list = [256, 1024, 4096, 8192, 16384]
    seq_len_list=[256,1024]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = []

    for d_model in d_model_list:
        for seq_len in seq_len_list:
            print(f"Testing: d_model={d_model}, seq_len={seq_len}...")
            
            # 清理显存缓存
            if device == "cuda":
                torch.cuda.empty_cache()
            
            try:
                # 构造数据，需要 requires_grad 来测试后向传播
                q = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)
                k = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)
                v = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)

                # --- 1. Warm up ---
                for _ in range(10):
                    _ = pytorch_naive_attention(q, k, v)

                # --- 2. Measure Forward Pass ---
                if device == "cuda": torch.cuda.synchronize()
                start_fwd = time.perf_counter()
                
                for _ in range(100):
                    out = pytorch_naive_attention(q, k, v)
                
                if device == "cuda": torch.cuda.synchronize()
                end_fwd = time.perf_counter()
                avg_fwd = (end_fwd - start_fwd) / 100 * 1000  # 转为 ms

                # --- 3. Measure Memory ---
                # 在后向传播开始前记录显存
                mem_usage = torch.cuda.memory_allocated(device) / (1024 ** 2) # MB

                # --- 4. Measure Backward Pass ---
                grad_output = torch.randn_like(out)
                if device == "cuda": torch.cuda.synchronize()
                start_bwd = time.perf_counter()
                
                for _ in range(100):
                    out.backward(grad_output, retain_graph=True)
                
                if device == "cuda": torch.cuda.synchronize()
                end_bwd = time.perf_counter()
                avg_bwd = (end_bwd - start_bwd) / 100 * 1000 # ms

                results.append({
                    "d_model": d_model,
                    "seq_len": seq_len,
                    "fwd_ms": f"{avg_fwd:.3f}",
                    "bwd_ms": f"{avg_bwd:.3f}",
                    "mem_mb": f"{mem_usage:.2f}"
                })

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"OOM at seq_len={seq_len}")
                    results.append({
                        "d_model": d_model,
                        "seq_len": seq_len,
                        "fwd_ms": "OOM",
                        "bwd_ms": "OOM",
                        "mem_mb": "OOM"
                    })
                else:
                    raise e

    # 打印结果表格 (作业 1.1.2 建议使用 pandas 打印表格)
    df = pd.DataFrame(results)
    print("\nBenchmark Results:")
    print(df.to_markdown()) # 需要 pip install tabulate

if __name__ == "__main__":
    benchmark_attention()