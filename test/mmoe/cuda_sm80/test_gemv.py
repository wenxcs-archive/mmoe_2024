import torch
import mmoe
import mmoe.moe
import mmoe.moe.cuda_sm80
import mmoe.moe.cuda_sm80.moe_gemv_Wf8_Af16_Of16_Accf32

def test_performance(m=12800, k=4096, n=1024, index_size=256):
    print(mmoe.moe.cuda_sm80.moe_gemv_Wf8_Af16_Of16_Accf32.gemv)

    W = torch.randn(m, k).float().cuda()
    act = torch.randn(n, k).float().cuda()
    outp = torch.randn(m, n).float().cuda()
    index = torch.randint(0, m, index_size).to(torch.int32).cuda()

    mmoe.moe.cuda_sm80.moe_gemv_Wf8_Af16_Of16_Accf32.gemv(W, act, outp, index)


test_performance()