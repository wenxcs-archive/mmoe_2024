#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <assert.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <iostream>
#include <fstream>
#include <random>
#include <numeric>

#include "cutlass/cutlass.h"

#include "kernel_default_gemm_universal.h"

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/command_line.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"
#include "helper.h"

///////////////////////////////////////////////////////////////////////////////////////////////////

// The code section below describes datatype for input, output matrices and computation between
// elements in input matrices.
using ElementAccumulator = float;                  // <- data type of accumulator
using ElementComputeEpilogue = ElementAccumulator; // <- data type of epilogue operations
using ElementInputA = cutlass::half_t;             // <- data type of elements in input matrix A
using ElementInputB = cutlass::half_t;             // <- data type of elements in input matrix B
using ElementOutput = cutlass::half_t;                       // <- data type of elements in output matrix D

// The code section below describes matrix layout of input and output matrices.
// Column Major for Matrix A, B and C.
//
using LayoutInputA = cutlass::layout::RowMajor;
using LayoutInputB = cutlass::layout::RowMajor;
using LayoutOutput = cutlass::layout::RowMajor;

// This code section describes whether you want to use tensor cores or regular SIMT cores on GPU SM
using MMAOp = cutlass::arch::OpClassTensorOp;

// This code section describes CUDA SM architecture number
using SmArch = cutlass::arch::Sm80;

// This code section describes the tile size a thread block will compute
using ShapeMMAThreadBlock =
    cutlass::gemm::GemmShape<128, 128, 32>; // <- threadblock tile M = 128, N = 128, K = 32
// This code section describes tile size a warp will compute
using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 32>; // <- warp tile M = 64, N = 64, K = 32
// This code section describes the size of MMA op
using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 16>; // <- MMA Op tile M = 8, N = 8, K = 4
// 16, 8, 8 -> Turing
// 16, 8, 16 -> Ampere

// This code section describes how threadblocks are scheduled on GPU
using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>; // <- ??

// Define the epilogue operation as LinearCombination. This is approximately equal to
//
//    d_ij = alpha * sum_k(a_ik * b_kj) + c_ij
//
using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,                                    // <- data type of output matrix
    128 / cutlass::sizeof_bits<ElementOutput>::value, // <- this is the number of elements per
                                                      // vectorized memory access. For half
                                                      // precision, it's 8 elements. This becomes
                                                      // the vector width of math instructions in
                                                      // epilogue too
    ElementAccumulator,                               // <- data type of accumulator
    ElementComputeEpilogue>;                          // <- data type for alpha in linear combination function

// Number of pipelines you want to use
constexpr int NumStages = 5;
// Ampere -> 4/5
// Turing -> 2

using Gemm = cutlass::gemm::device::GemmUniversal<ElementInputA,
                                                  LayoutInputA,
                                                  ElementInputB,
                                                  LayoutInputB,
                                                  ElementOutput,
                                                  LayoutOutput,
                                                  ElementAccumulator,
                                                  MMAOp,
                                                  SmArch,
                                                  ShapeMMAThreadBlock,
                                                  ShapeMMAWarp,
                                                  ShapeMMAOp,
                                                  EpilogueOp,
                                                  SwizzleThreadBlock,
                                                  NumStages,
                                                  8, /*alignmentA*/
                                                  8, /*alignmentB*/
                                                  cutlass::arch::OpMultiplyAdd,
                                                  cutlass::ComplexTransform::kNone,
                                                  cutlass::ComplexTransform::kNone,
                                                  false, /*GatherA*/
                                                  true,  /*GatherB*/
                                                  true   /*ScatterD*/
                                                  >;
// ================================================================================

int run_gemm(int m, int k, int n, int index_size,
          ElementInputA* tensor_a_ptr,
          ElementInputB* tensor_b_ptr,
          ElementOutput* tensor_c_ptr,
          ElementOutput* tensor_d_ptr,
          int* tensor_indices_ptr,
          ElementInputA* W_scale_ptr,
          ElementInputA* topk_weights_ptr,
          int* expert_ids_ptr,
          int* num_tokens_post_padded_ptr,
          int num_valid_tokens,
          int topk,
          int split_k_slices = 1
        )
{
  // ================================================================================
  // Initialization setup

  // Create a tuple of problem size for matrix multiplication
  cutlass::gemm::GemmCoord problem_size({m,k,n});

  // Create a tuple of problem size for matrix multiplication
  cutlass::gemm::GemmCoord problem_size_real(problem_size.m(),
                                             index_size,
                                             problem_size.k());

  // Initialize alpha/beta for dot product computation
  ElementComputeEpilogue alpha = ElementComputeEpilogue(1);
  ElementComputeEpilogue beta = ElementComputeEpilogue(1);

  // Split K dimension into 1 partitions

  auto tensor_a_layout = LayoutInputA::packed(problem_size.mk());
  auto tensor_b_layout = LayoutInputA::packed(problem_size.nk());
  auto tensor_c_layout = LayoutInputA::packed(problem_size.mn());
  auto tensor_d_layout = LayoutInputA::packed(problem_size.mn());

  static_assert(std::is_same<Gemm::Arguments::debug_flag, int>::value, "Failed");

  // Create a tuple of gemm kernel arguments. This is later passed as arguments to launch
  // instantiated CUTLASS kernel
  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      problem_size_real,                // <- problem size of matrix multiplication
      split_k_slices,                   // <- k-dimension split factor
      {alpha, beta},                    // <- alpha, beta
      tensor_a_ptr,           // <- reference to matrix A on device
      tensor_b_ptr,           // <- reference to matrix B on device
      tensor_c_ptr,           // <- reference to matrix C on device
      tensor_d_ptr, // <- reference to matrix D on device
      tensor_a_layout.capacity(problem_size.mk()),
      tensor_b_layout.capacity(cutlass::make_Coord(index_size, problem_size.k())),
      tensor_c_layout.capacity(problem_size.mn()),
      tensor_d_layout.capacity(problem_size.mn()),
      tensor_a_layout.stride(),
      tensor_b_layout.stride(),
      tensor_c_layout.stride(),
      tensor_d_layout.stride(),
      nullptr,                       // <- pointer to index vector to gather A on device
      tensor_indices_ptr,  // <- pointer to index vector to gather B on device
      tensor_indices_ptr,
      W_scale_ptr,
      topk_weights_ptr,
      expert_ids_ptr,
      num_tokens_post_padded_ptr,
      num_valid_tokens,
      topk
      }; // <- pointer to index vector to scatter D on device

  // Using the arguments, query for extra workspace required for matrix multiplication computation
  size_t workspace_size = Gemm::get_workspace_size(arguments);

  // Allocate workspace memory
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  // Instantiate CUTLASS kernel depending on templates
  Gemm gemm_op;

  // Check the problem size is supported or not
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);

  // Initialize CUTLASS kernel with arguments and workspace pointer
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);

  status = gemm_op();
  CUTLASS_CHECK(status);

  return 0;
}


#include "torch/extension.h"


void moe_linear(
  torch::Tensor W,
  torch::Tensor act,
  torch::Tensor outp,
  torch::Tensor index,
  torch::Tensor W_scale,
  torch::Tensor topk_weights,
  torch::Tensor expert_ids,
  torch::Tensor num_tokens_post_padded,
  int num_valid_tokens,
  int topk,
  int split_k
)
{
  int m = W.size(0);
  int n = act.size(0);
  int k = W.size(1);
  int index_size = index.size(0);

  run_gemm(m, n, k, index_size, 
    (ElementInputA*)W.data_ptr(), 
    (ElementInputB*)act.data_ptr(),
    (ElementOutput*)outp.data_ptr(),
    (ElementOutput*)outp.data_ptr(),
    (int*)index.data_ptr(),
    (ElementInputA*)W_scale.data_ptr(),
    (ElementInputA*)topk_weights.data_ptr(),
    (int*)expert_ids.data_ptr(),
    (int*)num_tokens_post_padded.data_ptr(),
    num_valid_tokens,
    topk,
    split_k
  );
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("moe_linear", &moe_linear, "");
}