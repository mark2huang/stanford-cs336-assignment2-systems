import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torchvision

#toy model
class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel,self).__init__()
        self.net=nn.Sequential(
            nn.Linear(10,20),
            nn.ReLU(),
            nn.Linear(20,5)
        )
    def forward(self,x):
        return self.net(x)
    

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # 使用 Gloo 后端方便 CPU 测试
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


#多GPU训练逻辑
def train_native_ddp(rank,world_size,data,target):

    """
    rank: 当前进程编号
    world_size: 总进程数
    model: 你的神经网络模型
    dataloader: 数据加载器（注意 DDP 下通常使用 DistributedSampler）
    optimizer: 优化器
    criterion: 损失函数
    """

    setup(rank,world_size)

    #实例化模型
    model=ToyModel()
    #加载和单GPU一样的初始化权重
    state_dict = torch.load("initial_weights.pth", weights_only=True)
    model.load_state_dict(state_dict)
    optimizer=optim.SGD(model.parameters(),lr=0.01)
    criterion=nn.MSELoss()

    #step1:broadcast 参数分发对齐，所有GPU都拿到一样的参数
    for param in model.parameters():
        dist.broadcast(param.data,src=0)
    #step2:data sharing 数据分布，假设有4个GPU，每个GPU拿到25%数据
    dataset=TensorDataset(data,target)
    sampler=torch.utils.data.distributed.DistributedSampler(dataset,num_replicas=world_size,rank=rank)
    dataloader=torch.utils.data.DataLoader(dataset,batch_size=2,sampler=sampler)

    #step3:训练
    optimizer.zero_grad()
    for batch_data,batch_target in dataloader:
        output=model(batch_data)
        loss=criterion(output,batch_target)
        loss.backward()
    #step4:同步梯度
    with torch.no_grad():
        for param in model.parameters():
            if param.grad is not None:
                #all-reduce 把所有进程该参数的梯度相加
                dist.all_reduce(param.grad.data,op=dist.ReduceOp.SUM)
                param.grad.data/=world_size
    optimizer.step()

    # 返回更新后的权重用于验证（仅在 Rank 0 返回）
    if rank == 0:
        print(f"DDP Training Step Complete on Rank 0.")
        # 保存模型状态，稍后和单机结果对比
        torch.save(model.state_dict(), "ddp_model.pth")
        print("save model success!")

    cleanup()



#单GPU训练逻辑
def train_single_process(data,target):
    model=ToyModel()
    optimizer=optim.SGD(model.parameters(),lr=0.01)
    criterion=nn.MSELoss()
    #加载和多GPU一样的初始化权重
    state_dict=torch.load("initial_weights.pth")
    model.load_state_dict(state_dict)

    optimizer.zero_grad()
    output=model(data)
    loss=criterion(output,target)
    loss.backward()

    optimizer.step()
    torch.save(model.state_dict(), "single_model.pth")
    print("Single Process Training Step Complete.")

if __name__ == "__main__":
    world_size=4
    data=torch.randn(8,10)
    target=torch.randn(8,5)

    initial_model=ToyModel()
    torch.save(initial_model.state_dict(),"initial_weights.pth")

    #多GPU训练
    mp.spawn(train_native_ddp,args=(world_size,data,target),nprocs=world_size,join=True)
    """
    PyTorch 会启动 4 个独立的 Python 进程。它在调用你的 train_naive_ddp 函数时,会自动把进程的索引(0 到 3)作为第一个参数塞进去。
    所以，这 4 个进程实际上是这样被启动的：
    进程 0 运行:train_naive_ddp(0, 4, data, target) <-- 这里的 0 是自动生成的 rank
    进程 1 运行:train_naive_ddp(1, 4, data, target)
    进程 2 运行:train_naive_ddp(2, 4, data, target)
    进程 3 运行:train_naive_ddp(3, 4, data, target)
    """

    #单GPU训练
    train_single_process(data,target)

    ddp_state=torch.load("ddp_model.pth")
    single_state=torch.load("single_model.pth")

    all_match=True
    for key in ddp_state:
        if not torch.allclose(ddp_state[key], single_state[key], atol=1e-6):
            print(f"❌ 权重不匹配: {key}")
            all_match = False
            break
    
    if all_match:
        print("✅ 校验成功!DDP 权重与单进程完全一致。")

"""
DDP Training Step Complete on Rank 0.
Single Process Training Step Complete.
✅ 校验成功!DDP 权重与单进程完全一致。
"""






#以下请在有NVIDIA GPU的机器上测试，我是放到kaggle上使用GPU T4*2
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
#add test DummyXLTransformer

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
    
def benchmark_naive_ddp(rank, world_size):
    # 【补充】：必须告诉系统主机在哪，否则 NCCL 无法建联
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    if rank == 0:
        print("Initializing Half-XL Model (12 layers) to fit 16GB VRAM...")

    # 如果 Kaggle T4 显存炸了，把下面的 num_layers 临时改成 12
    model = DummyXLTransformer(num_layers=12).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    for param in model.parameters():
        dist.broadcast(param.data, src=0)

    batch_size = 1
    seq_len = 128
    data = torch.randint(0, 10000, (batch_size, seq_len)).to(device)

    if rank == 0:
        print("Starting warm-up...")

    for _ in range(3):
        output = model(data)
        loss = output.sum()
        loss.backward()
        for param in model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad.data)
        optimizer.step()
        optimizer.zero_grad()

    if rank == 0:
        print("Starting Benchmark...")
        
    torch.cuda.synchronize(device)
    start_step_time = timeit.default_timer()

    optimizer.zero_grad()
    output = model(data)
    loss = output.sum()
    loss.backward()

    torch.cuda.synchronize(device)
    start_comm_time = timeit.default_timer()

# 2.2 A_Naive_Implementation_of_Distributed_Data_Parallel_Training
# 旧版通信逻辑：同步这“一整套梯度”，一个一个tensor进行通信（发 Q 的梯度、发 K 的梯度、发 FFN1 的梯度...）

    #with torch.no_grad():
    #    for param in model.parameters():
    #        if param.grad is not None:
    #            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
    #            param.grad.data /= world_size


#Initializing Half-XL Model (12 layers) to fit 16GB VRAM...
#Starting warm-up...
#Starting Benchmark...
#| Setup (2x T4 GPUs) | Total Step Time | Comm Time   | Comm Ratio |
#|--------------------|-----------------|-------------|------------|
#| Naive DDP (逐参)   |     476.35 ms |   250.52 ms |     52.59% |

#测试在 Kaggle 双 T4 环境下进行。然而，完整的 FP32 XL 模型加 AdamW 优化器需要超过 24GB 显存，
#导致了 OOM。为了完成测试，我将模型砍到 12 层并将 Adam 换成 SGD 丢弃了优化器状态。这不仅成功在 16GB 内跑通了测试，展示了逐个参数调用 all_reduce 时庞大的延迟开销。


# 2.3.1 Reducing the Number of Communication Calls
# 新版：Flat DDP 通信逻辑：torch._utils._flatten_dense_tensors 把所有梯度的数值拼成一个巨大的、扁平的一维 Tensor，然后只发起 1 次 all_reduce
    with torch.no_grad():
        # 1. 收集所有非空的梯度张量
        grads = [param.grad.data for param in model.parameters() if param.grad is not None]
        # 2. 核心魔法：把几百个小 Tensor 展平拼接成一个连续的 1D 大 Tensor
        flat_grads = torch._utils._flatten_dense_tensors(grads)
        # 3. 极速发车：只调用【1 次】 all_reduce！
        dist.all_reduce(flat_grads, op=dist.ReduceOp.SUM)
        flat_grads /= world_size
        # 4. 拆解还原：把算好的大 Tensor 拆分回原来的形状
        unflattened_grads = torch._utils._unflatten_dense_tensors(flat_grads, grads)
        # 5. 写回原来的梯度中
        for g, unflat_g in zip(grads, unflattened_grads):
            g.copy_(unflat_g)

#Initializing Half-XL Model (12 layers) to fit 16GB VRAM...
#Starting warm-up...
#Starting Benchmark...

#| Setup (2x T4 GPUs) | Total Step Time | Comm Time   | Comm Ratio |
#|--------------------|-----------------|-------------|------------|
#| Naive DDP (逐参)   |     493.74 ms |   271.29 ms |     54.95% |
#通过在通信前将所有梯度展平成单一连续张量，通信时间显著下降。
#因为打包通信消除了数百次独立调用 all_reduce 时产生的巨大内核启动和网络握手延迟开销




    torch.cuda.synchronize(device)
    end_comm_time = timeit.default_timer()

    # 【补充】：必须跑完 step 才是完整的一个训练步
    optimizer.step()
    torch.cuda.synchronize(device)
    end_step_time = timeit.default_timer()

    # 【补充】：最后必须把结果打印出来！
    if rank == 0:
        total_time = (end_step_time - start_step_time) * 1000
        comm_time = (end_comm_time - start_comm_time) * 1000
        comm_ratio = (comm_time / total_time) * 100
        
        print(f"\n| Setup (2x T4 GPUs) | Total Step Time | Comm Time   | Comm Ratio |")
        print(f"|--------------------|-----------------|-------------|------------|")
        print(f"| Naive DDP (逐参)   | {total_time:>10.2f} ms | {comm_time:>8.2f} ms | {comm_ratio:>9.2f}% |")

    dist.destroy_process_group()

if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    if world_size < 2:
        print(f"This benchmark requires at least 2 GPUs. Found {world_size}.")
    else:
        mp.spawn(benchmark_naive_ddp, args=(2,), nprocs=2, join=True)







