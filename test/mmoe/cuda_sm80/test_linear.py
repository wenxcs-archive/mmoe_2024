import os
import torch
import mmoe
import mmoe.moe
import mmoe.moe.cuda_sm80
import mmoe.moe.cuda_sm80.moe_linear_Wf8_Af16_Of16_Accf32
from loguru import logger


def test_performance_block1_expert1(m=12800, k=4096, n=1024):
    torch.manual_seed(12345)
    index_size = 128
    expert_num = 16
    # inputs
    w = torch.randn(expert_num, k, m).half().cuda()
    act = torch.randn(n, k).half().cuda()
    outp = torch.zeros(index_size, m).half().cuda()
    # meta
    index = torch.arange(0, index_size, dtype=torch.int32).cuda()
    w_scale = torch.ones(expert_num).half().cuda()
    expert_ids = torch.arange(0, 1, dtype=torch.int32).cuda()
    num_tokens_post_padded = torch.tensor(index_size, dtype=torch.int32).cuda()
    num_valid_tokens = index_size
    topk = 2
    topk_weights = torch.ones(n, topk).half().cuda()

    mmoe.moe.cuda_sm80.moe_linear_Wf8_Af16_Of16_Accf32.moe_linear(
        m, k, n, expert_num, index_size,
        w,
        act,
        outp,
        index,
        w_scale,
        topk_weights,
        expert_ids,
        num_tokens_post_padded,
        num_valid_tokens,
        topk,
        1
    )

    w0 = w[0, :, :]
    a0 = act[:128, :]

    c0 = a0 @ w0

    torch.testing.assert_close(outp, c0)

    times = 100
    all_time = 0.0
    for j in range(10 + times):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
    
        mmoe.moe.cuda_sm80.moe_linear_Wf8_Af16_Of16_Accf32.moe_linear(
            m, k, n, expert_num, index_size,
            w,
            act,
            outp,
            index,
            w_scale,
            topk_weights,
            expert_ids,
            num_tokens_post_padded,
            num_valid_tokens,
            topk,
            1
        )

        end.record()
        torch.cuda.synchronize()
        elapsed_time_ms = start.elapsed_time(end)

        if j >= 10:
            all_time += elapsed_time_ms
        
    avg_time = all_time / times

    logger.info(f"m={m}, k={k}, n={n}, index_size={index_size}, expert_num={expert_num}")
    logger.info(f"Average time(ms):{avg_time}")
    logger.info(f"TFLOPS: {2 * index_size * k * n / (avg_time * 1e-3) / 1e12}")
    logger.info(f"Throughput(G/s): {(m*k*2 + k*index_size*2 + m*index_size*2) / (avg_time * 1e-3) / 1e9}")


def test_performance_block2_expert2(m=12800, k=4096, n=1024, index_size=256):
    torch.manual_seed(12345)
    expert_num = 16
    # inputs
    w = torch.randn(expert_num, k, m).half().cuda()
    act = torch.randn(n, k).half().cuda()
    outp = torch.zeros(index_size, m).half().cuda()
    # meta
    index = torch.arange(0, index_size, dtype=torch.int32).cuda()
    w_scale = torch.ones(expert_num).half().cuda()
    expert_ids = torch.arange(4, 6, dtype=torch.int32).cuda()
    num_tokens_post_padded = torch.tensor(index_size, dtype=torch.int32).cuda()
    num_valid_tokens = index_size
    topk = 2
    topk_weights = torch.ones(n, topk).half().cuda()

    mmoe.moe.cuda_sm80.moe_linear_Wf8_Af16_Of16_Accf32.moe_linear(
        m, k, n, expert_num, index_size,
        w,
        act,
        outp,
        index,
        w_scale,
        topk_weights,
        expert_ids,
        num_tokens_post_padded,
        num_valid_tokens,
        topk,
        1
    )

    outps = torch.chunk(outp, 2, dim=0)

    w0 = w[4, :, :]
    w1 = w[5, :, :]
    a0 = act[:128, :]
    a1 = act[128:256, :]

    c0 = a0 @ w0
    c1 = a1 @ w1

    torch.testing.assert_close(outps[0], c0)
    torch.testing.assert_close(outps[1], c1)

    times = 100
    all_time = 0.0
    for j in range(10 + times):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
    
        mmoe.moe.cuda_sm80.moe_linear_Wf8_Af16_Of16_Accf32.moe_linear(
            m, k, n, expert_num, index_size,
            w,
            act,
            outp,
            index,
            w_scale,
            topk_weights,
            expert_ids,
            num_tokens_post_padded,
            num_valid_tokens,
            topk,
            1
        )

        end.record()
        torch.cuda.synchronize()
        elapsed_time_ms = start.elapsed_time(end)

        if j >= 10:
            all_time += elapsed_time_ms
        
    avg_time = all_time / times

    logger.info(f"m={m}, k={k}, n={n}, index_size={index_size}, expert_num={expert_num}")
    logger.info(f"Average time(ms):{avg_time}")
    logger.info(f"TFLOPS: {2 * index_size * k * n / (avg_time * 1e-3) / 1e12}")
    logger.info(f"Throughput(G/s): {(m*k + k*index_size + m * index_size)*2/ (avg_time * 1e-3) / 1e9}")


def test_performance_block128(m=12800, k=4096, n=1024, expert_num=1, splitk=1):
    torch.manual_seed(12345)
    index_size=expert_num*128
    n = max(n, index_size)
    # inputs
    w = torch.randn(expert_num, k, m).half().cuda()
    act = torch.randn(index_size, k).half().cuda()
    outp = torch.zeros(index_size, m).half().cuda()
    # meta
    index = torch.arange(0, index_size, dtype=torch.int32).cuda()
    w_scale = torch.ones(expert_num).half().cuda()
    #expert_ids = torch.arange(0, expert_num, dtype=torch.int32).cuda()
    expert_ids = torch.zeros(expert_num, dtype=torch.int32).cuda()
    num_tokens_post_padded = torch.tensor(index_size, dtype=torch.int32).cuda()
    num_valid_tokens = index_size
    topk = 2
    topk_weights = torch.ones(n, topk).half().cuda()

    mmoe.moe.cuda_sm80.moe_linear_Wf8_Af16_Of16_Accf32.moe_linear(
        m, k, n, expert_num, index_size,
        w,
        act,
        outp,
        index,
        w_scale,
        topk_weights,
        expert_ids,
        num_tokens_post_padded,
        num_valid_tokens,
        topk,
        splitk
    )

    outps = torch.chunk(outp, expert_num, dim=0)

    for i in range(expert_num):
        idx = expert_ids[i].item()
        w_ = w[idx, :, :]
        a_ = act[idx*128:(idx+1)*128, :]
        c_ = a_ @ w_
        torch.testing.assert_close(outps[i], c_)

    times = 100
    all_time = 0.0
    for j in range(10 + times):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
    
        mmoe.moe.cuda_sm80.moe_linear_Wf8_Af16_Of16_Accf32.moe_linear(
            m, k, n, expert_num, index_size,
            w,
            act,
            outp,
            index,
            w_scale,
            topk_weights,
            expert_ids,
            num_tokens_post_padded,
            num_valid_tokens,
            topk,
            splitk
        )

        end.record()
        torch.cuda.synchronize()
        elapsed_time_ms = start.elapsed_time(end)

        if j >= 10:
            all_time += elapsed_time_ms
        
    avg_time = all_time / times

    logger.info(f"m={m}, k={k}, n={n}, index_size={index_size}, expert_num={expert_num}")
    logger.info(f"Average time(ms):{avg_time}")
    logger.info(f"TFLOPS: {2 * index_size * k * n / (avg_time * 1e-3) / 1e12}")
    logger.info(f"Throughput(G/s): {(m*k + k*index_size + m * index_size)*2/ (avg_time * 1e-3) / 1e9}")



if __name__ == "__main__":
    k = int(os.environ.get("K", 4096))
    m = int(os.environ.get("M", 12800))
    expert_num = int(os.environ.get("EXPERT_NUM", 1))
    splitk = int(os.environ.get("SPLITK", 1))
    test_performance_block128(k=k, m=m, expert_num=expert_num, splitk=splitk)