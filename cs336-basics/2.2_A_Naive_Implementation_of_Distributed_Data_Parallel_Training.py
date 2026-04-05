import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

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
    PyTorch 会启动 4 个独立的 Python 进程。它在调用你的 train_naive_ddp 函数时，会自动把进程的索引（0 到 3）作为第一个参数塞进去。
    所以，这 4 个进程实际上是这样被启动的：
    进程 0 运行：train_naive_ddp(0, 4, data, target) <-- 这里的 0 是自动生成的 rank
    进程 1 运行：train_naive_ddp(1, 4, data, target)
    进程 2 运行：train_naive_ddp(2, 4, data, target)
    进程 3 运行：train_naive_ddp(3, 4, data, target)
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
        print("✅ 校验成功！DDP 权重与单进程完全一致。")

"""
DDP Training Step Complete on Rank 0.
Single Process Training Step Complete.
✅ 校验成功！DDP 权重与单进程完全一致。
"""



#以下请在有NVIDIA GPU的机器上测试
import timeit
# 1. 导入你第一节课写的模型（或者根据参数临时搭一个 XL 尺寸的 Transformer）
from cs336_basics.model import Transformer, TransformerConfig 

def benchmark_naive_ddp(rank, world_size):
    # Kaggle 上记得用 nccl
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    # 2. 初始化 XL 模型并移到 GPU
    # 注意：如果 Kaggle T4 显存炸了（OOM），把 batch_size 设为 1，seq_len 设小一点（比如 128）
    config = TransformerConfig(d_model=1600, d_ff=6400, num_layers=48, num_heads=25, vocab_size=10000)
    model = Transformer(config).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    # Broadcast 同步初始参数
    for param in model.parameters():
        dist.broadcast(param.data, src=0)

    # 伪造一点假数据
    data = torch.randint(0, 10000, (1, 128)).to(device)
    target = torch.randint(0, 10000, (1, 128)).to(device)

    # 预热几步 (Warm-up)
    for _ in range(3):
        output = model(data)
        loss = output.sum() # 简化计算
        loss.backward()
        for param in model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad.data)
        optimizer.step()
        optimizer.zero_grad()

    # 3. 开始正式测速
    torch.cuda.synchronize(device)
    start_step_time = timeit.default_timer()

    # [前向 + 反向]
    output = model(data)
    loss = output.sum()
    loss.backward()

    # 4. === 测量纯通信时间 ===
    torch.cuda.synchronize(device)
    start_comm_time = timeit.default_timer()
    
    with torch.no_grad():
        for param in model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                param.grad.data /= world_size
                
    torch.cuda.synchronize(device)
    end_comm_time = timeit.default_timer()
    # ========================

    # [优化器更新]
    optimizer.step()

    torch.cuda.synchronize(device)
    end_step_time = timeit.default_timer()

    # 5. 打印结果
    if rank == 0:
        total_time = (end_step_time - start_step_time) * 1000 # 转毫秒
        comm_time = (end_comm_time - start_comm_time) * 1000
        comm_ratio = (comm_time / total_time) * 100
        
        print(f"| XL Model (2 GPUs) | Total Step Time | Comm Time | Comm Ratio |")
        print(f"|-------------------|-----------------|-----------|------------|")
        print(f"| Naive DDP         | {total_time:.2f} ms     | {comm_time:.2f} ms | {comm_ratio:.2f}%     |")


"""
Comm Ratio(通信时间占比)可能高达 70% 甚至 90%!
因为 XL 模型有几百上千个参数矩阵。
在 Naive DDP 里，你是一个一个参数发起 all_reduce 的。
每一次 all_reduce 都要在两张显卡之间建立一次网络通信（发送指令、握手、传输）这会产生极其庞大的 Overhead(网络延迟开销)。
"""

    





