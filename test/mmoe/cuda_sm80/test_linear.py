import torch
import mmoe
import mmoe.moe
import mmoe.moe.cuda_sm80
import mmoe.moe.cuda_sm80.moe_linear_Wf8_Af16_Of16_Accf32

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
