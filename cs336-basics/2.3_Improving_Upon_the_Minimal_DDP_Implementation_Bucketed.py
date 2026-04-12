
import os
import torch
import torch.nn as nn
import torch.distributed as dist

class MyBucketedDDP(nn.Module):
    def __init__(self, module: nn.Module, bucket_size_mb: float = 25.0):
        super().__init__()
        self.module = module
        self.world_size = dist.get_world_size()
        
        # 1. 计算一个桶最多能装多少个 float32 数字 (1 MB = 1024*1024 bytes, 1 float32 = 4 bytes)
        self.bucket_size_elements = int((bucket_size_mb * 1024 * 1024) / 4)
        
        # 2. 存放所有构建好的桶
        self.buckets = []
        # 3. 存放异步通信句柄
        self.handles = []
        
        # 4. 初始化阶段：广播对齐参数
        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)
            
        # 5. 核心：构建分桶 (Bucketing)
        self._build_buckets()
        
        # 6. 注册钩子
        self._register_hooks()

    def _build_buckets(self):
        """
        按照反向传播的顺序（即参数的逆序），把参数塞进不同的桶里。
        """
        # 获取所有需要梯度的参数，并且【逆序】！
        # 为什么逆序？因为反向传播是从最后一层往前算的，逆序装桶能保证桶按顺序填满！
        params = [p for p in self.module.parameters() if p.requires_grad]
        params = list(reversed(params))
        
        current_bucket_params = []
        current_bucket_size = 0
        
        for p in params:
            numel = p.numel() # 获取参数的元素个数
            
            # 如果当前桶装不下这个参数了，就把当前桶“封箱”，开启一个新桶
            if current_bucket_size + numel > self.bucket_size_elements and len(current_bucket_params) > 0:
                self._create_and_append_bucket(current_bucket_params)
                current_bucket_params = []
                current_bucket_size = 0
                
            current_bucket_params.append(p)
            current_bucket_size += numel
            
        # 把最后剩下的装不满的参数也封成一个桶
        if len(current_bucket_params) > 0:
            self._create_and_append_bucket(current_bucket_params)
            
        # 为了方便调试，打印一下分桶情况
        if dist.get_rank() == 0:
            print(f"Created {len(self.buckets)} buckets for communication.")

    def _create_and_append_bucket(self, params_list):
        """
        为一个参数列表创建一个真实的“物理桶”。
        """
        total_elements = sum(p.numel() for p in params_list)
        # 在显存中开辟一块连续的一维张量，作为“桶”的物理空间
        # 注意：必须和模型在同一个 device 上
        device = params_list[0].device
        buffer = torch.zeros(total_elements, dtype=torch.float32, device=device)
        
        # 记录下每个参数在这个大 buffer 中的起始和结束位置 (切片索引)
        # 这样未来才能把参数的梯度准确地塞进 buffer，或者从 buffer 读出来
        offsets = []
        current_offset = 0
        for p in params_list:
            offsets.append((current_offset, current_offset + p.numel()))
            current_offset += p.numel()
            
        # 把这个桶的所有信息打包存起来
        bucket_info = {
            "buffer": buffer,               # 物理显存块
            "params": params_list,          # 这个桶包含的参数列表
            "offsets": offsets,             # 切片位置
            "expected_count": len(params_list), # 这个桶一共需要等几个梯度算完
            "ready_count": 0                # 当前已经算好了几个梯度
        }
        self.buckets.append(bucket_info)

    def _register_hooks(self):
        """
        给每个参数挂上钩子，钩子触发时要把梯度塞进对应的桶里。
        """
        # 辅助工厂函数，用来锁定 param 和它所属的 bucket_idx
        def create_hook(p, bucket_idx, param_idx_in_bucket):
            def hook_fn(*args):
                bucket = self.buckets[bucket_idx]
                
                # 1. 梯度拷贝入桶：把 p.grad 的数据拍平，复制到 buffer 的对应切片中
                start, end = bucket["offsets"][param_idx_in_bucket]
                bucket["buffer"][start:end].copy_(p.grad.data.view(-1))
                
                # 2. 桶的就绪计数器 + 1
                bucket["ready_count"] += 1
                
                # 3. 核心判断：如果这个桶里的所有参数梯度都算完了，立刻发车！
                if bucket["ready_count"] == bucket["expected_count"]:
                    # 异步发车！
                    handle = dist.all_reduce(bucket["buffer"], op=dist.ReduceOp.SUM, async_op=True)
                    self.handles.append(handle)
                    
            return hook_fn

        # 遍历所有桶，给里面的每个参数打上专属烙印
        for b_idx, bucket in enumerate(self.buckets):
            for p_idx, p in enumerate(bucket["params"]):
                hook = create_hook(p, b_idx, p_idx)
                p.register_post_accumulate_grad_hook(hook)

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        # 1. 等待所有后台发车的桶完成通信
        for handle in self.handles:
            handle.wait()
        self.handles.clear()
        
        # 2. 将桶里的平均梯度，拆解还原回各个参数的 .grad 中
        for bucket in self.buckets:
            # 梯度求平均 (全归约是求和，这里除以卡数)
            bucket["buffer"] /= self.world_size
            
            # 拆解写回
            for p, (start, end) in zip(bucket["params"], bucket["offsets"]):
                # 从大 buffer 中切片，变回原来矩阵的形状，写回 p.grad
                reduced_grad = bucket["buffer"][start:end].view(p.grad.shape)
                p.grad.data.copy_(reduced_grad)
                
            # 极其重要：为下一个 step 重置计数器！！！
            bucket["ready_count"] = 0



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
    ddp_model = MyBucketedDDP(base_model)
    
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
Initializing Overlap DDP Model (12 layers)...
Created 75 buckets for communication.
Starting warm-up...
Starting Benchmark...

| Setup (2x T4 GPUs) | Total Step Time | Exposed Comm Time | Comm Ratio |
|--------------------|-----------------|-------------------|------------|
| Overlap DDP (钩子) |     474.62 ms |       28.39 ms |      5.98% |


“在我的测试中，分桶 DDP 的总耗时为 474.62 毫秒，暴露的通信时间为 28.39 毫秒。出乎意料的是,这比不分桶的重叠版本(461.10 毫秒）略慢。这个看似反直觉的结果实际上符合实现的底层机制：
首先，分桶在 GPU 上引入了不容忽视的内存拷贝开销（将独立梯度展平存入桶缓冲区）。在 T4 这种显存带宽受限的 GPU 上，这种开销可能会抵消减少通信调用带来的收益。
其次，我默认的 25MB 桶大小导致生成了 75 个桶。为了让分桶方法显著优于不分桶版本，我需要对 bucket_size_mb 进行超参数搜索
（例如增加到 100MB 或 250MB)，以找到“减少内核启动延迟（更少的桶）”与“保持高重叠效率（更小的桶）”之间的最佳平衡点。
此外，使用高带宽互连（如 NVLink)而不是 PCIe 也会转移瓶颈，从而清晰地展示分桶的优越性。”
"""

