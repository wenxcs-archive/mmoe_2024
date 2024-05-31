#pragma once
#include "cutlass/gemm/kernel/default_gemm_universal.h"
#include "kernel_gemm_universal.h"

namespace cutlass {
namespace gemm {
namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Real-valued GEMM kernels
//

template <
    /// Element type for A matrix operand
    typename ElementA,
    /// Layout type for A matrix operand
    typename LayoutA,
    /// Access granularity of A matrix in units of elements
    int kAlignmentA,
    /// Element type for B matrix operand
    typename ElementB,
    /// Layout type for B matrix operand
    typename LayoutB,
    /// Access granularity of B matrix in units of elements
    int kAlignmentB,
    /// Element type for C and D matrix operands
    typename ElementC,
    /// Layout type for C and D matrix operands
    typename LayoutC,
    /// Element type for internal accumulation
    typename ElementAccumulator,
    /// Operator class tag
    typename OperatorClass,
    /// Tag indicating architecture to tune for
    typename ArchTag,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape,
    /// Epilogue output operator
    typename EpilogueOutputOp,
    /// Threadblock-level swizzling operator
    typename ThreadblockSwizzle,
    /// Number of stages used in the pipelined mainloop
    int Stages,
    /// Operation performed by GEMM
    typename Operator,
    /// Use zfill or predicate for out-of-bound cp.async
    SharedMemoryClearOption SharedMemoryClear,
    /// Permute result D
    typename PermuteDLayout,
    /// Permute operand A
    typename PermuteALayout,
    /// Permute operand B
    typename PermuteBLayout
>
struct DefaultGemmUniversal<
  ElementA,
  LayoutA,
  ComplexTransform::kNone,   // transform A
  kAlignmentA,
  ElementB,
  LayoutB,
  ComplexTransform::kNone,   // transform B
  kAlignmentB,
  ElementC,
  LayoutC,
  ElementAccumulator,
  OperatorClass,
  ArchTag,
  ThreadblockShape,
  WarpShape,
  InstructionShape,
  EpilogueOutputOp,
  ThreadblockSwizzle,
  Stages,
  Operator,
  SharedMemoryClear,
  false,
  true,
  true,
  PermuteDLayout,
  PermuteALayout,
  PermuteBLayout,
  typename platform::enable_if< ! cutlass::is_complex<ElementAccumulator>::value>::type
> {

  using DefaultGemmKernel = typename kernel::DefaultGemm<
    ElementA,
    LayoutA,
    kAlignmentA,
    ElementB,
    LayoutB,
    kAlignmentB,
    ElementC,
    LayoutC,
    ElementAccumulator,
    OperatorClass,
    ArchTag,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    EpilogueOutputOp,
    ThreadblockSwizzle,
    Stages,
    true,
    Operator,
    SharedMemoryClear,
    false,
    true,
    true,
    PermuteDLayout,
    PermuteALayout,
    PermuteBLayout
  >::GemmKernel;

  /// Universal kernel without StreamkFeature member type
  template <class SwizzleT, class Enable = void>
  class SelectBase :
    public kernel::MGemmUniversal<
      typename DefaultGemmKernel::Mma,
      typename DefaultGemmKernel::Epilogue,
      SwizzleT>
  {};

  /// Universal kernel with StreamkFeature member type
  template <class SwizzleT>
  class SelectBase<SwizzleT, typename SwizzleT::StreamkFeature> :
    public kernel::GemmUniversalStreamk<
      typename DefaultGemmKernel::Mma,
      typename DefaultGemmKernel::Epilogue,
      SwizzleT>
  {};

  /// Select kernel by ThreadblockSwizzle's support for StreamkFeature
  using GemmKernel = SelectBase<ThreadblockSwizzle>;
};

}  // namespace kernel
}  // namespace gemm
}  // namespace cutlass
