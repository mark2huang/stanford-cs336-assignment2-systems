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
    seq_len_list = [256, 1024, 4096, 8192, 16384]
    
    if torch.cuda.is_available():
        device = "cuda"
        print(f"Is CUDA available: {torch.cuda.is_available()}")
        print(f"GPU Device Name: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        print(f"Total Memory: {props.total_memory / 1024**2:.2f} MB")
        print(f"Processor Count: {props.multi_processor_count}")
        print(f"Compute Capability: {props.major}.{props.minor}")
    else:
        device ="cpu"
    results = []

    compiled_naive_attention = torch.compile(pytorch_naive_attention)
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
                    warm_up_out = compiled_naive_attention(q, k, v)
                    warm_up_out.backward(torch.randn_like(warm_up_out))
                    
                # --- 2. Measure Forward Pass ---
                if device == "cuda": torch.cuda.synchronize()
                start_fwd = time.perf_counter()
                
                for _ in range(100):
                    out = compiled_naive_attention(q, k, v)
                
                if device == "cuda": torch.cuda.synchronize()
                end_fwd = time.perf_counter()
                avg_fwd = (end_fwd - start_fwd) / 100 * 1000  # 转为 ms

                # --- 3. Measure Memory ---
                # 在后向传播开始前记录显存
                mem_usage = torch.cuda.memory_allocated(device) / (1024 ** 2) # MB

                # --- 4. Measure Backward Pass ---
                grad_output = torch.randn_like(out)
                if device == "cuda": torch.cuda.synchronize()
                start = time.perf_counter()
                
                for _ in range(100):
                    out=compiled_naive_attention(q,k,v)
                    out.backward(grad_output)
                if device == "cuda": torch.cuda.synchronize()
                end = time.perf_counter()
                avg_fwd_bwd = (end - start) / 100 * 1000 # ms 注意这里包含前向和反向的时间
                real_bwd_ms=avg_fwd_bwd-avg_fwd

                results.append({
                    "d_model": d_model,
                    "seq_len": seq_len,
                    "fwd_ms": f"{avg_fwd:.3f}",
                    "bwd_ms": f"{real_bwd_ms:.3f}",
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

    # 打印结果表格 (作业 1.3.1 建议使用 pandas 打印表格)
    df = pd.DataFrame(results)
    print("\nBenchmark Results:")
    print(df.to_markdown()) # 需要 pip install tabulate

if __name__ == "__main__":
    benchmark_attention()



#1.3.1 Example - Weighted Sum
def weighted_sum(x,weight):
    # 此处假设 x 的维度是 [..., D]，weight 的维度是 [D],维度不匹配， 广播+点积
    return (weight *x).sum(axis=-1)


import triton
import triton.language as tl
@triton.jit
def weight_sum_fwd(
    x_ptr,weight_ptr, #输入指针
    output_ptr, #输出指针
    x_row_stride,x_stride_dim,#步长，告诉我们如何在张量的每个轴上移动一个元素
    weight_stride_dim,#通常为1
    output_stride_row,#通常为1
    ROWS,D,#矩阵总行数和维度
    ROWS_TILE_SIZE:tl.constexpr,
    D_TILE_SIZE: tl.constexpr,#分块大小：必须在编译时已知 对于3*2的矩阵ROWS_TILE_SIZE=1，D_TILE_SIZE=2
):
    #每个实例将计算一组行分块的加权和
    # tl.program_id(0) 指示当前在哪个线程块
    row_tile_idx=tl.program_id(0)
    
    #块指针（Block pointers） 让我们能选择内存中的一个N维区域并移动它
    """
    块指针需要知道：
    - 指向张量的第一个元素的指针
    - 张量的总形状（以处理越界访问）
    - 每个维度的步长（以正确使用内存布局）
    - 起始块的N维坐标(offsets)
    - 每次加载/存储时使用的块形状
    - 维度在内存中的排列顺序
    """
    x_block_ptr=tl.make_block_ptr(
        x_ptr,
        shape=(ROWS,D,),
        strides=(x_row_stride,x_stride_dim),
        offsets=(row_tile_idx * ROWS_TILE_SIZE,0),
        block_shape=(ROWS_TILE_SIZE,D_TILE_SIZE),
        order=(1,0),
    )

    weight_block_ptr=tl.make_block_ptr(
        weight_ptr,
        shape=(D,),
        strides=(weight_stride_dim,),
        offsets=(0,),
        block_shape=(D_TILE_SIZE,),
        orser=(0,),
    )

    output_block_ptr=tl.make_block_ptr(
        output_ptr,
        shape=(ROWS,),
        strides=(output_stride_row,),
        offsets=(row_tile_idx * ROWS_TILE_SIZE,),
        block_shape=(ROWS_TILE_SIZE,),
        orser=(0,),
    )
    #初始化一个缓冲区用于写入结果
    output=tl.zeros((ROWS_TILE_SIZE,),dtype=tl.float32)
    #在D维度上进行循环
    for i in range(tl.cdiv(D,D_TILE_SIZE)):
        #加载当前的块指针内容并带有边界检查
        row=tl.load(x_block_ptr,boundary_check=(0,1),padding_option="zero")
        weight=tl.load(weight_block_ptr,boundary_check=(0,),padding_option="zero")

        #计算该行分块的加权和
        output+=tl.sum(row*weight[None,:],axis=1)

        #将指针移动到下一个分块
        x_block_ptr=x_block_ptr.advance((0,D_TILE_SIZE))
        weight_block_ptr=weight_block_ptr.advance((D_TILE_SIZE,))
    #将结果写回输出快指针
    tl.store(output_block_ptr,output,boundary_check=(0,))

class WeightedSumFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,weight):
        #x=(Batch, Seq_len, D) such as (8, 512, 128)
        D,output_dims=x.shape[-1],x.shape[:-1]
        input_shape=x.shape
        # 把前面所有的维度（Batch, Seq_len）全部压扁成一个维度。(8, 512, 128) 会变成 (4096, 128),这样我们的 Triton 内核只需要处理一个 2D 矩阵 就行了
        x=rearrange(x,"...d ->(...)d")
        """
        这一步将展平后的 x 和 weight 保存到 ctx 对象中。
        当你之后运行 loss.backward() 时,PyTorch 会把这两个张量传给 backward 函数，用来计算梯度。
        """
        ctx.save_for_backward(x,weight)

        assert len(weight.shape)==1 and weight.shape[0]==D,"Dimension mismatch"
        assert x.is_cuda and weight.is_cuda,"Expected CUDA tensor"
        assert x.is_contiguous(),"Our pointer arithmetic will assume contiguous x" #is_contiguous (连续性)：这是最关键的！Triton 的指针算法基于步长（Stride）。如果张量在内存里是不连续的（比如经过了转置但没调 contiguous()），指针计算就会乱掉，读到错误的数据。

        ctx.D_TILE_SIZE=triton.next_power_of_2(D)#Triton 的 block_shape 必须是 2 的幂,如果D=128，就会计算出128/16=8
        ctx.ROWS_TILE_SIZE=16
        ctx.input_shape=input_shape

        y=torch.empty(output_dims,device=x.device)

        n_rows=y.numel()

        """
        grid 定义：(cdiv(n_rows, ctx.ROWS_TILE_SIZE), )。这决定了启动多少个线程块。如果总共有 160 行，每块处理 16 行，就启动 10 个线程块。
        传递步长 (stride)：手动把 PyTorch 张量的内存步长传给内核。这就是我们在内核里计算地址的依据。
        """
        weight_sum_fwd[(tl.cdiv(n_rows,ctx.ROWS_TILE_SIZE),)](
            x.weight,
            y,
            x.stride(0),x.stride(1),
            weight.stride(0),
            y.stride(0),
            ROWS=n_rows,D=D,
            ROWS_TILE_SIZE=ctx.ROWS_TILE_SIZE,D_TILE_SIZE=ctx.D_TILE_SIZE,
        )
        return y.view(input_shape[:-1])#view把它还原成原始的形状
    
    
    



    


    










