from __future__ import annotations

from typing import Type

import torch

class FlashAttentionPytorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        # --- 维度对齐处理 ---
        is_3d = False
        if Q.ndim == 3:
            is_3d = True
            Q = Q.unsqueeze(1) # (B, N, d) -> (B, 1, N, d)
            K = K.unsqueeze(1)
            V = V.unsqueeze(1)

        B, H, Nq, d = Q.shape
        Nk = K.shape[2]
        Bq, Bk = 128, 128
        
        O = torch.zeros_like(Q)
        # 统计量必须用 float32 保证指数运算不溢出
        logsumexp = torch.zeros((B, H, Nq), device=Q.device, dtype=torch.float32)
        scale = d ** -0.5

        # 外层循环
        for i in range(0, Nq, Bq):
            q_tile = Q[:, :, i : i + Bq, :] 
            curr_Bq = q_tile.shape[2]

            m_i = torch.full((B, H, curr_Bq), float("-inf"), device=Q.device, dtype=torch.float32)
            l_i = torch.zeros((B, H, curr_Bq), device=Q.device, dtype=torch.float32)
            O_i = torch.zeros((B, H, curr_Bq, d), device=Q.device, dtype=torch.float32)

            # 内层循环
            for j in range(0, Nk, Bk):
                k_tile = K[:, :, j : j + Bk, :] 
                v_tile = V[:, :, j : j + Bk, :]
                
                S = (q_tile @ k_tile.transpose(-1, -2)) * scale
                
                # --- 在线 Softmax 更新 ---
                m_block = torch.max(S, dim=-1).values # 修正：必须是 .values
                m_old = m_i
                m_i = torch.max(m_block, m_old)
                
                alpha = torch.exp(m_old - m_i)
                p_tilde = torch.exp(S - m_i.unsqueeze(-1)) 
                
                l_i = alpha * l_i + torch.sum(p_tilde, dim=-1)
                O_i = alpha.unsqueeze(-1) * O_i + torch.matmul(p_tilde, v_tile)

            # 写入当前 Query 块的结果
            O_final_tile = O_i / l_i.unsqueeze(-1)
            O[:, :, i : i + Bq, :] = O_final_tile.to(Q.dtype)
            
            # --- 修正：索引必须是切片 i:i+Bq ---
            logsumexp[:, :, i : i + Bq] = m_i + torch.log(l_i)

        # --- 关键修正：为了通过测试，需要根据输入维度调整保存的 L 的形状 ---
        save_l = logsumexp.squeeze(1) if is_3d else logsumexp
        
        # 保存供反向传播使用的张量
        ctx.save_for_backward(Q, K, V, O, save_l)
        ctx.is_causal = is_causal
        
        # 返回结果同样根据需要还原维度
        return O.squeeze(1) if is_3d else O

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError("Backward pass not implemented yet.")
    

def get_flashattention_autograd_function_pytorch() -> Type:
    """
    Returns a torch.autograd.Function subclass that implements FlashAttention2.
    The expectation is that this class will implement FlashAttention2
    using only standard PyTorch operations (no Triton!).

    Returns:
        A class object (not an instance of the class)
    """
    # For example: return MyFlashAttnAutogradFunctionClass
    return FlashAttentionPytorch


def get_flashattention_autograd_function_triton() -> Type:
    """
    Returns a torch.autograd.Function subclass that implements FlashAttention2
    using Triton kernels.
    The expectation is that this class will implement the same operations
    as the class you return in get_flashattention_autograd_function_pytorch(),
    but it should do so by invoking custom Triton kernels in the forward
    and backward passes.

    Returns:
        A class object (not an instance of the class)
    """
    # For example: return MyTritonFlashAttentionAutogradFunctionClass
    raise NotImplementedError


def get_ddp_individual_parameters(module: torch.nn.Module) -> torch.nn.Module:
    """
    Returns a torch.nn.Module container that handles
    parameter broadcasting and gradient synchronization for
    distributed data parallel training.

    This container should overlaps communication with backprop computation
    by asynchronously communicating gradients as they are ready
    in the backward pass. The gradient for each parameter tensor
    is individually communicated.

    Args:
        module: torch.nn.Module
            Underlying model to wrap with DDP.
    Returns:
        Instance of a DDP class.
    """
    # For example: return DDPIndividualParameters(module)
    raise NotImplementedError


def ddp_individual_parameters_on_after_backward(ddp_model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    """
    Code to run after the backward pass is completed, but before we take
    an optimizer step.

    Args:
        ddp_model: torch.nn.Module
            DDP-wrapped model.
        optimizer: torch.optim.Optimizer
            Optimizer being used with the DDP-wrapped model.
    """
    # For example: ddp_model.finish_gradient_synchronization()
    raise NotImplementedError


def get_ddp_bucketed(module: torch.nn.Module, bucket_size_mb: float) -> torch.nn.Module:
    """
    Returns a torch.nn.Module container that handles
    parameter broadcasting and gradient synchronization for
    distributed data parallel training.

    This container should overlaps communication with backprop computation
    by asynchronously communicating buckets of gradients as they are ready
    in the backward pass.

    Args:
        module: torch.nn.Module
            Underlying model to wrap with DDP.
        bucket_size_mb: The bucket size, in megabytes. If None, use a single
            bucket of unbounded size.
    Returns:
        Instance of a DDP class.
    """
    raise NotImplementedError


def ddp_bucketed_on_after_backward(ddp_model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    """
    Code to run after the backward pass is completed, but before we take
    an optimizer step.

    Args:
        ddp_model: torch.nn.Module
            DDP-wrapped model.
        optimizer: torch.optim.Optimizer
            Optimizer being used with the DDP-wrapped model.
    """
    # For example: ddp_model.finish_gradient_synchronization()
    raise NotImplementedError


def ddp_bucketed_on_train_batch_start(ddp_model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    """
    Code to run at the very start of the training step.

    Args:
        ddp_model: torch.nn.Module
            DDP-wrapped model.
        optimizer: torch.optim.Optimizer
            Optimizer being used with the DDP-wrapped model.
    """
    raise NotImplementedError


def get_sharded_optimizer(params, optimizer_cls: Type[torch.optim.Optimizer], **kwargs) -> torch.optim.Optimizer:
    """
    Returns a torch.optim.Optimizer that handles optimizer state sharding
    of the given optimizer_cls on the provided parameters.

    Arguments:
        params (``Iterable``): an ``Iterable`` of :class:`torch.Tensor` s
            or :class:`dict` s giving all parameters, which will be sharded
            across ranks.
        optimizer_class (:class:`torch.nn.Optimizer`): the class of the local
            optimizer.
    Keyword arguments:
        kwargs: keyword arguments to be forwarded to the optimizer constructor.
    Returns:
        Instance of sharded optimizer.
    """
    raise NotImplementedError
