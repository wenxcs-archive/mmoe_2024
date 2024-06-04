import os
import re
import sys

for (m, k) in [(12800,4096),]:
    for e in range(1, 17):
        for s in [1]:
            if os.path.exists(f"log.txt"):
                os.remove(f"log.txt")
            os.system(f"M={m} K={k} EXPERT_NUM={e} SPLITK={s} nsys nvprof python ../test_linear.py > log.txt 2>&1")

            with open("log.txt", "r") as f:
                lines = f.readlines()
                for line in lines:
                    if "cutlass::gemm::kernel::DefaultGemmUniversal" in line:
                        numbers = re.findall(r'\d+\.?\d*', line)
                        print(f"{m}, {k}, {e}, {s}, {float(numbers[3])*1e-6}")
            
            os.system(f"rm *report*")