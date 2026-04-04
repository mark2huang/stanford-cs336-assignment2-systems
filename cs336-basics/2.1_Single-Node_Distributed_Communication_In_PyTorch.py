import os
import numpy as np
import torch
import torch.distributed as dist  # 导入PyTorch的分布式通信模块
import torch.multiprocessing as mp # 导入PyTorch的多进程模块
import timeit

#uv run python 2.1_Single-Node_Distributed_Communication_In_PyTorch.py
def setup(rank, world_size):
    # 这两行环境变量是告诉所有进程，作为“主节点（Master）”的进程在哪里。
    # 因为这里是单机多进程（Single-Node），所以IP地址是本机 "localhost"，端口随便挑了一个空闲的 "29500"。
    # 所有进程在初始化时，都会去连接这个地址和端口来互相建立联系。
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    os.environ["GLOO_SOCKET_IFNAME"] = "lo0"  # 消除 Mac 上的网络警告
    
    # 初始化进程组。这一步是分布式训练的核心。
    # "gloo" 告诉PyTorch使用CPU网络进行通信；
    # rank 告诉当前进程它的编号是多少（0, 1, 2, 或 3）；
    # world_size 告诉当前进程总共有多少个兄弟进程参与。
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def run_benchmark(rank, world_size,data_size_mb):
    
    # 生成本地数据。每个进程都会独立执行这一步。
    #data = torch.randint(0, 10, (3,))
    num_elements=int((data_size_mb*1024*1024)/4) #1MB 对应的 float32 元素个数是 1024 * 1024 / 4 = 262,144 个
    data=torch.rand(num_elements,dtype=torch.float32)
    
    # 3. 核心通信操作：all_reduce（全归约）
    # PyTorch会在后台把这4个进程里的 data 收集起来，默认做加法（Sum）操作。
    # async_op=False 表示这是一个“同步”操作，代码会在这里阻塞，直到加法算完并且把结果发回给每个进程为止。
    # 执行完这行后，每个进程手里的 data 变量都会变成那 4 个原始 data 加起来的总和。

    records=[]
    #warm up
    for i in range(5):
        dist.all_reduce(data, async_op=False)
    
    num_iters=10
    for i in range(num_iters):
        time_begin=timeit.default_timer()
        dist.all_reduce(data, async_op=False)
        time_end=timeit.default_timer()
    
    avg_time_ms=((time_end-time_begin)*1000)/num_iters
    records.append(avg_time_ms)


    if rank == 0:
        print(f"| {'Gloo':<7} | {'CPU':<6} | {world_size:<10} | {data_size_mb:<14} | {avg_time_ms:>13.3f} |")


def distributed_demo(rank, world_size, data_sizes):
    setup(rank, world_size)
    for size_mb in data_sizes:
        run_benchmark(rank, world_size, size_mb)
    dist.destroy_process_group() # 跑完清理进程组


if __name__ == "__main__":
    data_sizes_mb=[1,10,100,1000] #1MB,10MB,100MB,1000MB

    print(f"| {'Backend':<7} | {'Device':<6} | {'World Size':<10} | {'Data Size (MB)':<14} | {'Avg Time (ms)':<13} |")
    print(f"|{'-'*9}|{'-'*8}|{'-'*12}|{'-'*16}|{'-'*15}|")
    # 启动多进程。
    # fn 指定每个进程要运行的目标函数；
    # args 是传递给目标函数的额外参数（注意 rank 会被自动作为第一个参数传入，所以这里只传 world_size）；
    # nprocs 是要启动的进程数量；
    # join=True 表示主程序会阻塞在这里，等待所有子进程执行完毕后再退出。
    for world_size in [2,4,6]:
        print(f"\n--- Starting benchmark for world_size = {world_size} ---")
        mp.spawn(fn=distributed_demo, args=(world_size, data_sizes_mb), nprocs=world_size, join=True)

"""
| Backend | Device | World Size | Data Size (MB) | Avg Time (ms) |
|---------|--------|------------|----------------|---------------|

--- Starting benchmark for world_size = 2 ---
| Gloo    | CPU    | 2          | 1              |         0.072 |
| Gloo    | CPU    | 2          | 10             |         0.347 |
| Gloo    | CPU    | 2          | 100            |         3.547 |
| Gloo    | CPU    | 2          | 1000           |        35.320 |

--- Starting benchmark for world_size = 4 ---
| Gloo    | CPU    | 4          | 1              |         0.121 |
| Gloo    | CPU    | 4          | 10             |         0.653 |
| Gloo    | CPU    | 4          | 100            |         6.356 |
| Gloo    | CPU    | 4          | 1000           |        64.110 |

--- Starting benchmark for world_size = 6 ---
| Gloo    | CPU    | 6          | 1              |         0.257 |
| Gloo    | CPU    | 6          | 10             |         1.051 |
| Gloo    | CPU    | 6          | 100            |        10.641 |
| Gloo    | CPU    | 6          | 1000           |       109.227 |

"""



import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import timeit

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def run_benchmark(rank, world_size, data_size_mb):
    # 根据 MB 计算对应的 float32 元素数量
    num_elements = int((data_size_mb * 1024 * 1024) / 4)
    
    # 给每个进程分配到对应的 GPU 显卡上
    device = torch.device(f"cuda:{rank}")
    
    data = torch.rand(num_elements, dtype=torch.float32, device=device)

    # 1. 预热 (Warm-up)：执行 5 次
    for _ in range(5):
        dist.all_reduce(data, async_op=False)
        
    # 在开始计时前，必须确保前面所有的 GPU 操作都执行完了！
    torch.cuda.synchronize(device)
    
    # 2. 正式测速
    num_iters = 10
    time_begin = timeit.default_timer()
    
    for _ in range(num_iters):
        dist.all_reduce(data, async_op=False)
        
    # 在结束计时前，必须等待这一次的 all-reduce 彻底在 GPU 上完成！
    torch.cuda.synchronize(device)
    time_end = timeit.default_timer()
    
    avg_time_ms = ((time_end - time_begin) * 1000) / num_iters

    if rank == 0:
        print(f"| {'NCCL':<7} | {'GPU':<6} | {world_size:<10} | {data_size_mb:<14} | {avg_time_ms:>13.3f} |")

def distributed_demo(rank, world_size, data_sizes):
    setup(rank, world_size)
    for size_mb in data_sizes:
        run_benchmark(rank, world_size, size_mb)
    dist.destroy_process_group()

if __name__ == "__main__":
    # 检查 Kaggle 环境有几张 GPU
    num_gpus = torch.cuda.device_count()
    if num_gpus < 2:
        print(f"当前环境只有 {num_gpus} 张 GPU，请确保开启了多 GPU 环境（如 T4 x2）")
    else:
        data_sizes_mb = [1, 10, 100, 1000]
        
        print(f"| {'Backend':<7} | {'Device':<6} | {'World Size':<10} | {'Data Size (MB)':<14} | {'Avg Time (ms)':<13} |")
        print(f"|{'-'*9}|{'-'*8}|{'-'*12}|{'-'*16}|{'-'*15}|")
        
        for world_size in [2]: 
            mp.spawn(fn=distributed_demo, args=(world_size, data_sizes_mb), nprocs=world_size, join=True)

"""

!python benchmark_gpu.py
| Backend | Device | World Size | Data Size (MB) | Avg Time (ms) |
|---------|--------|------------|----------------|---------------|
| NCCL    | GPU    | 2          | 1              |         0.323 |
| NCCL    | GPU    | 2          | 10             |         2.675 |
| NCCL    | GPU    | 2          | 100            |        25.809 |
| NCCL    | GPU    | 2          | 1000           |       259.092 |

"""



"""
writeup.pdf:

Commentary on Benchmark Results:

Surprisingly, the CPU (Gloo) backend on an Apple M1 Pro outperformed the GPU (NCCL) backend on Kaggle's dual T4 GPUs. 
This occurs because the M1 Pro leverages a highly optimized Unified Memory Architecture (UMA) with massive bandwidth for inter-process communication,
whereas Kaggle's T4 GPUs lack NVLink and are severely bottlenecked by traversing the virtualized PCIe bus and host memory (achieving only ~3.8 GB/s). 
This highlights that while NCCL is optimized for GPU topologies, physical interconnects (PCIe vs. UMA vs. NVLink) dictate the actual communication overhead.



令人惊讶的是,Apple M1 Pro 上的 CPU (Gloo) 后端性能击败了 Kaggle 双 T4 GPU 上的 GPU (NCCL) 后端。
这是由于 M1 Pro 利用了高度优化的统一内存架构UMA,在进程间通信时拥有巨大的内存带宽；而 Kaggle 的 T4 GPU 缺少 NVLink 硬件，通信时必须跨越虚拟化的 PCIe 总线和主机内存，受到了严重的瓶颈限制（实测带宽仅约 3.8 GB/s)
这表明，尽管 NCCL 专为 GPU 拓扑优化,但底层的物理互连结构(PCIe vs 统一内存 vs NVLink)才是决定实际通信开销的根本因素。

"""
