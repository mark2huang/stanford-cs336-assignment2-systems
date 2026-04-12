
import timeit
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
class MyOverlapDDP(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
        self.world_size = dist.get_world_size()
        
        # 1. 初始化时 Broadcast 参数
        for param in module.parameters():
            dist.broadcast(param.data,src=0)
        # 2. 句柄存储：创建一个列表，用来存放所有正在后台运行的异步通信任务（Future Handles）
        self.handles = []
        # 3.调用钩子函数
        self._register_hooks()
    
    def _register_hooks(self):
        # 这是一个内部工厂函数，用来制造专属的钩子
        def create_hook(p):
            # 这才是真正挂在参数上的钩子函数
            def hook_fn(*args):
                # 1. 异步发起全归约操作（因为外层有 create_hook，所以这里的 p 是被锁定的专属参数）
                handle = dist.all_reduce(p.grad.data, op=dist.ReduceOp.SUM, async_op=True)
                # 2. 将句柄存入列表，立刻返回，不阻塞当前 GPU 线程
                self.handles.append(handle)
                
            return hook_fn # 把造好的专属钩子吐出去

        # 遍历所有参数，挨个挂上门铃
        for param in self.module.parameters():
            if param.requires_grad:
                # 调用工厂函数，给当前 param 造一个专属钩子
                exclusive_hook = create_hook(param)
                # 挂载钩子
                param.register_post_accumulate_grad_hook(exclusive_hook)


    def forward(self, *inputs, **kwargs):
        # DDP 容器本身不参与计算，只是个壳子
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        #  等待所有后台任务完成
        # TODO: 循环 self.handles 调用 .wait()
        for handle in self.handles:
            handle.wait()
        self.handles=[]
        # 清空 handles 列表，为下一轮做准备
        
        #  梯度平均化
        for param in self.module.parameters():
            if param.grad is not None:
                param.grad.data/=self.world_size
                



import timeit
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
class DummyTransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.ln_2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        x = x + self.attn(self.ln_1(x), self.ln_1(x), self.ln_1(x))[0]
        x = x + self.mlp(self.ln_2(x))
        return x

class DummyXLTransformer(nn.Module):
    def __init__(self, vocab_size=10000, d_model=1600, d_ff=6400, num_layers=48, num_heads=25):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            DummyTransformerBlock(d_model, num_heads, d_ff) for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x):
        x = self.embedding(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits
    
def benchmark_overlap_ddp(rank, world_size):
    # 1. 基础设置
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    if rank == 0:
        print("Initializing Overlap DDP Model (12 layers)...")

    # 2. 初始化模型并使用你的 MyOverlapDDP 包装！
    # 注意：MyOverlapDDP 的 __init__ 里已经帮你做了 broadcast 参数对齐
    base_model = DummyXLTransformer(num_layers=12).to(device)
    ddp_model = MyOverlapDDP(base_model)
    
    # 优化器包装的是 ddp_model 的参数
    optimizer = optim.SGD(ddp_model.parameters(), lr=1e-4)

    # 伪造一点假数据
    batch_size = 1
    seq_len = 128
    data = torch.randint(0, 10000, (batch_size, seq_len)).to(device)

    if rank == 0:
        print("Starting warm-up...")

    # 3. 预热几步 (注意要使用 ddp 模型的流程)
    for _ in range(3):
        optimizer.zero_grad()
        output = ddp_model(data)
        loss = output.sum()
        loss.backward() # 此时后台已经在偷偷发通信了！
        ddp_model.finish_gradient_synchronization() # 等待最后一部分发完
        optimizer.step()

    if rank == 0:
        print("Starting Benchmark...")
        
    torch.cuda.synchronize(device)
    start_step_time = timeit.default_timer()

    # [前向 + 反向]
    optimizer.zero_grad()
    output = ddp_model(data)
    loss = output.sum()
    
    # 反向传播开始（钩子会在这里面陆续被触发）
    loss.backward()

    # 4. === 测量“未被掩盖”的剩余通信时间 ===
    torch.cuda.synchronize(device)
    start_comm_time = timeit.default_timer()
    
    # 因为大部分梯度已经在 backward 时发走了，这里只等最后几个完成即可
    ddp_model.finish_gradient_synchronization()
                
    torch.cuda.synchronize(device)
    end_comm_time = timeit.default_timer()
    # ========================================

    # [优化器更新]
    optimizer.step()

    torch.cuda.synchronize(device)
    end_step_time = timeit.default_timer()

    # 5. 打印结果
    if rank == 0:
        total_time = (end_step_time - start_step_time) * 1000 # 转毫秒
        # 这里的 comm_time 指的是“没有被反向传播重叠掉的等待时间”
        exposed_comm_time = (end_comm_time - start_comm_time) * 1000
        comm_ratio = (exposed_comm_time / total_time) * 100
        
        print(f"\n| Setup (2x T4 GPUs) | Total Step Time | Exposed Comm Time | Comm Ratio |")
        print(f"|--------------------|-----------------|-------------------|------------|")
        print(f"| Overlap DDP (钩子) | {total_time:>10.2f} ms | {exposed_comm_time:>11.2f} ms | {comm_ratio:>9.2f}% |")

    dist.destroy_process_group()

if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    if world_size < 2:
        print(f"This benchmark requires at least 2 GPUs. Found {world_size}.")
    else:
        # 记得把这里改成调用 benchmark_overlap_ddp
        mp.spawn(benchmark_overlap_ddp, args=(2,), nprocs=2, join=True)



"""
2.2 A_Naive_Implementation_of_Distributed_Data_Parallel_Training
旧版通信逻辑:同步这“一整套梯度”,一个一个tensor进行通信(发 Q 的梯度、发 K 的梯度、发 FFN1 的梯度...)

    #with torch.no_grad():
    #    for param in model.parameters():
    #        if param.grad is not None:
    #            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
    #            param.grad.data /= world_size

Initializing Half-XL Model (12 layers) to fit 16GB VRAM...
Starting warm-up...
Starting Benchmark...
| Setup (2x T4 GPUs) | Total Step Time | Comm Time   | Comm Ratio |
|--------------------|-----------------|-------------|------------|
| Naive DDP (逐参)   |     476.35 ms |   250.52 ms |     52.59% |


2.3.1 Reducing the Number of Communication Calls

#Initializing Half-XL Model (12 layers) to fit 16GB VRAM...
#Starting warm-up...
#Starting Benchmark...

#| Setup (2x T4 GPUs) | Total Step Time | Comm Time   | Comm Ratio |
#|--------------------|-----------------|-------------|------------|
#| Naive DDP (逐参)   |     493.74 ms |   271.29 ms |     54.95% |
#通过在通信前将所有梯度展平成单一连续张量，通信时间显著下降。
#因为打包通信消除了数百次独立调用 all_reduce 时产生的巨大内核启动和网络握手延迟开销


单纯靠“减少次数”在超大模型上收益有限，必须靠“重叠”才能压榨出性能
并且在kaggle上碰到内存带宽瓶颈:flatten 和 unflatten 过程中,GPU 必须在显存里把几百块数据搬来搬去凑成一个大块。对于 XL 这种几 GB 规模的梯度，“搬家”的时间可能和“寄信”的时间一样长
它等到反向传播结束后才开始通信。实际上梯度是增量计算的。当某个参数梯度就绪时可以立即通信,从而实现通信与计算的重叠(Overlap)


2.3_Improving_Upon_the_Minimal_DDP_Implementation.py

Initializing Overlap DDP Model (12 layers)...
Starting warm-up...
Starting Benchmark...

| Setup (2x T4 GPUs) | Total Step Time | Exposed Comm Time | Comm Ratio |
|--------------------|-----------------|-------------------|------------|
| Overlap DDP (钩子) |     461.10 ms |       15.01 ms |      3.26% |


通过使用反向传播钩子将梯度通信与反向传播计算重叠，单步总耗时下降至 461.10 毫秒，
且‘暴露的通信时间’从朴素版的大约 250 毫秒骤降至仅 15.01 毫秒（通信占比仅为 3.26%）。
这有力地证明了：绝大部分的网络传输都在后台异步发生，同时 GPU 仍在计算较早网络层的梯度，从而成功地将通信延迟掩盖在了计算时间之下。
"""


