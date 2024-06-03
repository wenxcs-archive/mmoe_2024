import torch
import mmoe
import mmoe.moe
import mmoe.moe.cuda_sm80
import mmoe.moe.cuda_sm80.moe_linear_Wf8_Af16_Of16_Accf32

def test_performance(m=12800, k=4096, n=1024, index_size=256):
    expert_num = 16
    W = torch.randn(expert_num, m, k).half().cuda()
    act = torch.randn(k, n).half().cuda()
    outp = torch.zeros(m, n).half().cuda()
    index = torch.arange(0, index_size, dtype=torch.int32).cuda()
    w_scale = torch.ones(m).half().cuda()
    expert_ids = torch.arange(0, 2, dtype=torch.int32).cuda()
    num_tokens_post_padded = torch.tensor(index_size, dtype=torch.int32).cuda()
    num_valid_tokens = index_size
    topk = 2
    topk_weights = torch.ones(n, topk).half().cuda()

    mmoe.moe.cuda_sm80.moe_linear_Wf8_Af16_Of16_Accf32.moe_linear(
        W,
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

    print(outp)


test_performance()