#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <torch/extension.h>

#include <cstdint>
#include <type_traits>
#include <vector>

#include "cutlass/gemm/warp/default_mma_tensor_op.h"
#include "cutlass/arch/memory_sm80.h"
#include "cutlass/layout/tensor_op_multiplicand_sm80.h"

namespace {

constexpr int kTile = 64;
constexpr int kMixed = 2;

using PrefillWarpShape = cutlass::gemm::GemmShape<16, 16, 64>;
using PrefillMacroWarpShape = cutlass::gemm::GemmShape<16, 32, 64>;
using PrefillInstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;
using PrefillLayoutA = cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<8, 32>;
using PrefillM64W8LayoutA = cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<8, 64>;
using PrefillLayoutB = cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<8, 32>;
using PrefillMma = typename cutlass::gemm::warp::DefaultMmaTensorOp<
    PrefillWarpShape,
    PrefillInstructionShape,
    int8_t,
    PrefillLayoutA,
    int8_t,
    PrefillLayoutB,
    int32_t,
    cutlass::layout::RowMajor,
    cutlass::arch::OpMultiplyAddSaturate>::Type;
using PrefillMacroMma = typename cutlass::gemm::warp::DefaultMmaTensorOp<
    PrefillMacroWarpShape,
    PrefillInstructionShape,
    int8_t,
    PrefillLayoutA,
    int8_t,
    PrefillLayoutB,
    int32_t,
    cutlass::layout::RowMajor,
    cutlass::arch::OpMultiplyAddSaturate>::Type;

// Register coordinates are those of CUTLASS's exact M16 row-major iterator.
static_assert(
    std::is_same<typename PrefillMma::LayoutC,
                 cutlass::layout::RowMajor>::value,
    "M16 register-direct requires row-major CUTLASS accumulators");
static_assert(
    PrefillMma::Shape::kM == 16 && PrefillMma::Shape::kN == 16 &&
        PrefillMma::Shape::kK == 64,
    "M16 register-direct requires a 16x16x64 warp shape");
static_assert(
    PrefillMma::InstructionShape::kM == 16 &&
        PrefillMma::InstructionShape::kN == 8 &&
        PrefillMma::InstructionShape::kK == 32,
    "M16 register-direct requires a 16x8x32 MMA instruction");
static_assert(
    PrefillMma::FragmentC::kElements == 8 &&
        PrefillMma::IteratorC::Fragment::kElements == 8,
    "M16 register-direct mapping requires 8 accumulators per lane");
static_assert(
    std::is_same<typename PrefillMma::Policy::OpDelta,
                 cutlass::MatrixShape<1, 1>>::value,
    "M16 register-direct requires unit iterator operation deltas");

// The production M64 register epilogue is deliberately coupled to this
// exact CUTLASS accumulator layout.  A CUTLASS or shape change must fail the
// build instead of silently reusing a stale lane-to-coordinate mapping.
static_assert(
    std::is_same<typename PrefillMacroMma::LayoutC,
                 cutlass::layout::RowMajor>::value,
    "M64 register-direct requires row-major CUTLASS accumulators");
static_assert(
    PrefillMacroMma::Shape::kM == 16 &&
        PrefillMacroMma::Shape::kN == 32 &&
        PrefillMacroMma::Shape::kK == 64,
    "M64 register-direct requires a 16x32x64 warp shape");
static_assert(
    PrefillMacroMma::InstructionShape::kM == 16 &&
        PrefillMacroMma::InstructionShape::kN == 8 &&
        PrefillMacroMma::InstructionShape::kK == 32,
    "M64 register-direct requires a 16x8x32 MMA instruction");
static_assert(
    PrefillMacroMma::FragmentC::kElements == 16,
    "M64 register-direct mapping requires 16 accumulators per lane");

inline void check_cuda(cudaError_t status, const char* where) {
  TORCH_CHECK(status == cudaSuccess, where, " failed: ", cudaGetErrorString(status));
}

__device__ __forceinline__ int8_t sign_extend_int4(uint8_t value) {
  int8_t result = static_cast<int8_t>(value & 0x0f);
  return result >= 8 ? static_cast<int8_t>(result - 16) : result;
}

__device__ __forceinline__ int32_t load_physical_int4x4(
    const uint8_t* tile,
    int physical_offset) {
  const uint16_t packed = *reinterpret_cast<const uint16_t*>(
      tile + physical_offset / 2);
  const uint32_t x0 = static_cast<uint8_t>(
      sign_extend_int4(static_cast<uint8_t>(packed)));
  const uint32_t x1 = static_cast<uint8_t>(
      sign_extend_int4(static_cast<uint8_t>(packed >> 4)));
  const uint32_t x2 = static_cast<uint8_t>(
      sign_extend_int4(static_cast<uint8_t>(packed >> 8)));
  const uint32_t x3 = static_cast<uint8_t>(
      sign_extend_int4(static_cast<uint8_t>(packed >> 12)));
  return static_cast<int32_t>(x0 | (x1 << 8) | (x2 << 16) | (x3 << 24));
}

// K fragments passed to dp4a start on K multiples of four.  Layout-B maps
// those fragments to four consecutive physical int8 values with a four-byte
// aligned first offset, so one packed global load preserves the byte order
// consumed by dp4a.
__device__ __forceinline__ int32_t load_packed_s8x4(
    const int8_t* values,
    int byte_offset) {
  return *reinterpret_cast<const int32_t*>(values + byte_offset);
}

__device__ __forceinline__ void stage_layout_b_w4(
    const uint8_t* tile,
    int8_t* weight,
    int thread) {
  constexpr int kWords = kTile * kTile / 8;
  const uint32_t* source = reinterpret_cast<const uint32_t*>(tile);
  uint2* destination = reinterpret_cast<uint2*>(weight);
  for (int word = thread; word < kWords; word += blockDim.x) {
    const uint32_t packed = source[word];
    uint32_t even = packed & 0x0f0f0f0f;
    uint32_t odd = (packed >> 4) & 0x0f0f0f0f;
    // Each selected sign bit is bit 3 of an independent byte.  Multiplying
    // it by 0x1e produces 0xf0 in that byte with no cross-byte carry.
    even |= (even & 0x08080808) * 0x1e;
    odd |= (odd & 0x08080808) * 0x1e;
    // __byte_perm selector nibbles are ordered from output byte 0 upward.
    // 0x5140 selects even0, odd0, even1, odd1; 0x7362 selects the rest.
    destination[word] = make_uint2(
        __byte_perm(even, odd, 0x5140),
        __byte_perm(even, odd, 0x7362));
  }
}

__device__ __forceinline__ void stage_layout_b_w8(
    const int8_t* tile,
    int8_t* weight,
    int thread) {
  constexpr int kVectors = kTile * kTile / sizeof(uint4);
  const uint4* source = reinterpret_cast<const uint4*>(tile);
  uint4* destination = reinterpret_cast<uint4*>(weight);
  for (int vector = thread; vector < kVectors; vector += blockDim.x) {
    destination[vector] = source[vector];
  }
}

// Build one temporary all-W8 arena in the existing compact tile order. W4
// values are only sign-extended; each tile keeps its original FP32 scale.
// The source artifact remains compact and unchanged, and this work is part of
// the current projection call (including CUDA-graph capture/replay).
__global__ void expand_mixed_payload_to_w8_kernel(
    const uint8_t* __restrict__ state_bits,
    const uint8_t* __restrict__ w4_payload,
    const float* __restrict__ w4_scales,
    const int8_t* __restrict__ w8_payload,
    const float* __restrict__ w8_scales,
    const int32_t* __restrict__ w4_offsets,
    const int32_t* __restrict__ w8_offsets,
    int8_t* __restrict__ dense_w8_payload,
    float* __restrict__ dense_w8_scales,
    int32_t* __restrict__ dense_w8_offsets,
    int n_j,
    int n_k) {
  const int tile = static_cast<int>(blockIdx.x);
  const int j = tile / n_k;
  const int k = tile - j * n_k;
  __shared__ int source_tile;
  __shared__ int source_is_w4;
  if (threadIdx.x == 0) {
    int w4_tile = w4_offsets[j];
    int w8_tile = w8_offsets[j];
    for (int prior_k = 0; prior_k < k; ++prior_k) {
      const int prior_tile = j * n_k + prior_k;
      const bool prior_is_w4 =
          ((state_bits[prior_tile >> 3] >> (prior_tile & 7)) & 1) != 0;
      w4_tile += prior_is_w4;
      w8_tile += !prior_is_w4;
    }
    const bool is_w4 = ((state_bits[tile >> 3] >> (tile & 7)) & 1) != 0;
    source_is_w4 = is_w4;
    source_tile = is_w4 ? w4_tile : w8_tile;
    if (k == 0) {
      dense_w8_offsets[j] = j * n_k;
    }
    if (tile == n_j * n_k - 1) {
      dense_w8_offsets[n_j] = n_j * n_k;
    }
    dense_w8_scales[tile] = is_w4
        ? w4_scales[source_tile]
        : w8_scales[source_tile];
  }
  __syncthreads();

  int8_t* destination = dense_w8_payload +
      static_cast<int64_t>(tile) * kTile * kTile;
  if (source_is_w4) {
    stage_layout_b_w4(
        w4_payload + static_cast<int64_t>(source_tile) * kTile * 32,
        destination, threadIdx.x);
  } else {
    stage_layout_b_w8(
        w8_payload + static_cast<int64_t>(source_tile) * kTile * kTile,
        destination, threadIdx.x);
  }
}

// Fixed M64/K64 uniform-W8 async production stage.
// One 256-thread CTA copies one 16-byte A vector and one 16-byte W vector per
// thread.  A needs a logical-row to LayoutA destination permutation; W is
// already stored in physical LayoutB order.
template <bool FullRows, bool TileMajorA, typename LayoutA>
__device__ __forceinline__ void stage_m64_w8_cp_async(
    const int8_t* x,
    const int8_t* w8_payload,
    int8_t* activation_stage,
    int8_t* weight_stage,
    const LayoutA& layout_a,
    int row_start,
    int n_rows,
    int n_k,
    int w8_tile,
    int k_tile,
    int thread) {
  const int row = thread >> 2;
  const int vector = thread & 3;
  const int global_row = row_start + row;
  const int64_t activation_offset = TileMajorA
      ? (static_cast<int64_t>(global_row / 16) * n_k + k_tile) * 16 * kTile +
          (global_row % 16) * kTile + vector * 16
      : static_cast<int64_t>(global_row) * (n_k * kTile) +
          k_tile * kTile + vector * 16;
  if constexpr (FullRows) {
    cutlass::arch::cp_async<16, cutlass::arch::CacheOperation::Global>(
        activation_stage + layout_a({row, vector * 16}),
        x + activation_offset,
        true);
  } else {
    const bool valid_row = global_row < n_rows;
    const int8_t* activation_source = valid_row
        ? x + activation_offset
        : x;
    cutlass::arch::cp_async_zfill<16, cutlass::arch::CacheOperation::Global>(
        activation_stage + layout_a({row, vector * 16}),
        activation_source,
        valid_row);
  }
  cutlass::arch::cp_async<16, cutlass::arch::CacheOperation::Global>(
      weight_stage + thread * 16,
      w8_payload + static_cast<int64_t>(w8_tile) * kTile * kTile + thread * 16,
      true);
}

// Fixed W4/mixed companion for the M64 two-stage production path.  W8 copies directly
// into expanded LayoutB storage.  W4 copies only its 2 KiB packed payload;
// the caller expands it with the existing stage_layout_b_w4 after wait/sync.
template <bool FullRows, bool TileMajorA, typename LayoutA>
__device__ __forceinline__ void stage_m64_w4w8_cp_async(
    const int8_t* x,
    const uint8_t* w4_payload,
    const int8_t* w8_payload,
    int8_t* activation_stage,
    int8_t* weight_stage,
    uint8_t* packed_w4_stage,
    const LayoutA& layout_a,
    int row_start,
    int n_rows,
    int n_k,
    bool is_w4,
    int payload_tile,
    int k_tile,
    int thread) {
  const int row = thread >> 2;
  const int vector = thread & 3;
  const int global_row = row_start + row;
  const int64_t activation_offset = TileMajorA
      ? (static_cast<int64_t>(global_row / 16) * n_k + k_tile) * 16 * kTile +
          (global_row % 16) * kTile + vector * 16
      : static_cast<int64_t>(global_row) * (n_k * kTile) +
          k_tile * kTile + vector * 16;
  if constexpr (FullRows) {
    cutlass::arch::cp_async<16, cutlass::arch::CacheOperation::Global>(
        activation_stage + layout_a({row, vector * 16}),
        x + activation_offset,
        true);
  } else {
    const bool valid_row = global_row < n_rows;
    const int8_t* activation_source = valid_row
        ? x + activation_offset
        : x;
    cutlass::arch::cp_async_zfill<16, cutlass::arch::CacheOperation::Global>(
        activation_stage + layout_a({row, vector * 16}),
        activation_source,
        valid_row);
  }
  if (is_w4) {
    if (thread < kTile * 32 / 16) {
      cutlass::arch::cp_async<16, cutlass::arch::CacheOperation::Global>(
          packed_w4_stage + thread * 16,
          w4_payload + static_cast<int64_t>(payload_tile) * kTile * 32 +
              thread * 16,
          true);
    }
  } else {
    cutlass::arch::cp_async<16, cutlass::arch::CacheOperation::Global>(
        weight_stage + thread * 16,
        w8_payload + static_cast<int64_t>(payload_tile) * kTile * kTile +
            thread * 16,
        true);
  }
}

// Stage one M16/K64 activation tile and its compact W4/W8 N64/K64 payload.
// Each copy is a 16-byte transaction; the CTA may loop over multiple vectors.
__device__ __forceinline__ void stage_m16_splitk_cp_async(
    const int8_t* x, const uint8_t* w4_payload, const int8_t* w8_payload,
    int8_t* activation_stage, int8_t* weight_stage, uint8_t* packed_w4_stage,
    const PrefillLayoutA& layout_a, int row_start, int n_k, bool is_w4,
    int payload_tile, int k_tile, int thread) {
  for (int vector = thread; vector < 16 * 4; vector += blockDim.x) {
    const int row = vector >> 2;
    const int k_vector = vector & 3;
    const int global_row = row_start + row;
    const int8_t* source = x + static_cast<int64_t>(global_row) * (n_k * kTile) +
        k_tile * kTile + k_vector * 16;
    cutlass::arch::cp_async_zfill<16, cutlass::arch::CacheOperation::Global>(
        activation_stage + layout_a({row, k_vector * 16}), source, true);
  }
  if (is_w4) {
    for (int vector = thread; vector < kTile * 32 / 16; vector += blockDim.x) {
      cutlass::arch::cp_async<16, cutlass::arch::CacheOperation::Global>(
          packed_w4_stage + vector * 16,
          w4_payload + static_cast<int64_t>(payload_tile) * kTile * 32 + vector * 16,
          true);
    }
  } else {
    for (int vector = thread; vector < kTile * kTile / 16; vector += blockDim.x) {
      cutlass::arch::cp_async<16, cutlass::arch::CacheOperation::Global>(
          weight_stage + vector * 16,
          w8_payload + static_cast<int64_t>(payload_tile) * kTile * kTile + vector * 16,
          true);
    }
  }
}

__device__ __forceinline__ int32_t pack_s8x4(
    int8_t x0, int8_t x1, int8_t x2, int8_t x3) {
  return static_cast<int32_t>(static_cast<uint8_t>(x0)) |
      (static_cast<int32_t>(static_cast<uint8_t>(x1)) << 8) |
      (static_cast<int32_t>(static_cast<uint8_t>(x2)) << 16) |
      (static_cast<int32_t>(static_cast<uint8_t>(x3)) << 24);
}

__device__ __forceinline__ int8_t quantize_s8_rn(double value, double scale) {
  int result = __double2int_rn(value / scale);
  result = result < -128 ? -128 : result;
  result = result > 127 ? 127 : result;
  return static_cast<int8_t>(result);
}

__device__ __forceinline__ void fwht64_inplace(
    double* values,
    int rows,
    int thread) {
  const int lane = thread & 31;
  const int warp = thread >> 5;
  const int warps = blockDim.x >> 5;
  for (int row = warp; row < rows; row += warps) {
    double low = values[row * kTile + lane];
    double high = values[row * kTile + lane + 32];
#pragma unroll
    for (int stride = 1; stride < 32; stride <<= 1) {
      const double low_peer = __shfl_xor_sync(0xffffffff, low, stride);
      const double high_peer = __shfl_xor_sync(0xffffffff, high, stride);
      low = (lane & stride) ? low_peer - low : low + low_peer;
      high = (lane & stride) ? high_peer - high : high + high_peer;
    }
    const double low_before_final = low;
    const double high_before_final = high;
    values[row * kTile + lane] = low_before_final + high_before_final;
    values[row * kTile + lane + 32] = low_before_final - high_before_final;
  }
  // Callers read the transformed shared values after this helper returns.
  __syncthreads();
}

// Metadata rows are (mode, source_to_target, left, right).  Keep the inverse
// in the output-owning CTA so the only transform arithmetic is deterministic
// FP64 after the FP32 GEMM contract has completed.
__device__ __forceinline__ void inverse_u64_inplace(
    double* values,
    int rows,
    const int32_t* metadata,
    int thread) {
  if (metadata[0] == 0) {
    return;
  }
  for (int index = thread; index < rows * kTile; index += blockDim.x) {
    values[index] *= static_cast<double>(metadata[3 * kTile + index % kTile]);
  }
  __syncthreads();
  fwht64_inplace(values, rows, thread);
}

__device__ __forceinline__ float inverse_u64_value(
    const double* values,
    const int32_t* metadata,
    int index) {
  if (metadata[0] == 0) {
    return static_cast<float>(values[index]);
  }
  const int source = index % kTile;
  return static_cast<float>(
      values[(index / kTile) * kTile + metadata[kTile + source]] *
      static_cast<double>(metadata[2 * kTile + source]) * 0.125);
}

template <typename OutputT>
__device__ __forceinline__ void store_m64_output(
    OutputT* output, int64_t index, float value);

template <>
__device__ __forceinline__ void store_m64_output<float>(
    float* output, int64_t index, float value) {
  output[index] = value;
}

template <>
__device__ __forceinline__ void store_m64_output<uint16_t>(
    uint16_t* output, int64_t index, float value) {
  output[index] = __bfloat16_as_ushort(__float2bfloat16_rn(value));
}

struct R5RegisterRow {
  double low;
  double high;
};

template <typename scalar_t>
__device__ __forceinline__ R5RegisterRow r5_register_row(
    const scalar_t* x,
    const int32_t* transform_metadata,
    const int32_t* inverse_permutation,
    int row,
    int n_k,
    int k_tile,
    int lane,
    bool is_identity) {
  const int64_t base = static_cast<int64_t>(row) * (n_k * kTile) + k_tile * kTile;
  // Inputs are originally converted through float in the canonical path.
  // Permute those exact float bits first, then convert to double for all
  // transform arithmetic; this halves the permutation shuffle traffic.
  const float source_low = static_cast<float>(x[base + lane]);
  const float source_high = static_cast<float>(x[base + lane + 32]);
  if (is_identity) {
    return {source_low, source_high};
  }

  const int low_source = inverse_permutation[lane];
  const int high_source = inverse_permutation[lane + 32];
  const unsigned int full_warp = 0xffffffff;
  // All lanes execute every shuffle.  Each destination selects the half
  // containing its source after both source halves have been gathered.
  const float low_from_low = __shfl_sync(full_warp, source_low, low_source & 31);
  const float low_from_high = __shfl_sync(full_warp, source_high, low_source & 31);
  const float high_from_low = __shfl_sync(full_warp, source_low, high_source & 31);
  const float high_from_high = __shfl_sync(full_warp, source_high, high_source & 31);
  double low = (low_source < 32 ? low_from_low : low_from_high) *
      static_cast<double>(transform_metadata[2 * kTile + low_source]);
  double high = (high_source < 32 ? high_from_low : high_from_high) *
      static_cast<double>(transform_metadata[2 * kTile + high_source]);
#pragma unroll
  for (int stride = 1; stride < 32; stride <<= 1) {
    const double low_peer = __shfl_xor_sync(full_warp, low, stride);
    const double high_peer = __shfl_xor_sync(full_warp, high, stride);
    low = (lane & stride) ? low_peer - low : low + low_peer;
    high = (lane & stride) ? high_peer - high : high + high_peer;
  }
  const double low_before_final = low;
  const double high_before_final = high;
  const double transformed_low = low_before_final + high_before_final;
  const double transformed_high = low_before_final - high_before_final;
  return {
      transformed_low * static_cast<double>(transform_metadata[3 * kTile + lane]) * 0.125,
      transformed_high * static_cast<double>(transform_metadata[3 * kTile + lane + 32]) * 0.125};
}

template <typename scalar_t, bool TileMajorA>
__global__ void r5_a8_quantize_kernel(
    const scalar_t* __restrict__ x,
    const int32_t* __restrict__ metadata,
    int8_t* __restrict__ x_quantized,
    float* __restrict__ scales,
    int rows,
    int n_k) {
  __shared__ int32_t transform_metadata[4 * kTile];
  __shared__ int32_t inverse_permutation[kTile];
  __shared__ double selected_scale;
  __shared__ double warp_positive[8];
  __shared__ double warp_negative[8];
  const int thread = threadIdx.x;
  const int lane = thread & 31;
  const int warp = thread >> 5;
  const int warps = blockDim.x >> 5;
  const int k_tile = blockIdx.x;
  const int m_tile = blockIdx.y;
  const int tile_rows = rows == 1 ? 1 : 16;
  const int row_start = m_tile * tile_rows;
  for (int index = thread; index < 4 * kTile; index += blockDim.x) {
    transform_metadata[index] = metadata[static_cast<int64_t>(k_tile) * 4 * kTile + index];
  }
  __syncthreads();
  const bool is_identity = transform_metadata[0] == 0;
  if (!is_identity && thread < kTile) {
    inverse_permutation[transform_metadata[kTile + thread]] = thread;
  }
  __syncthreads();

  R5RegisterRow first = {0.0, 0.0};
  R5RegisterRow second = {0.0, 0.0};
  const bool first_valid = warp < tile_rows;
  const bool second_valid = warp + warps < tile_rows;
  if (first_valid) {
    first = r5_register_row(x, transform_metadata, inverse_permutation,
        row_start + warp, n_k, k_tile, lane, is_identity);
  }
  if (second_valid) {
    second = r5_register_row(x, transform_metadata, inverse_permutation,
        row_start + warp + warps, n_k, k_tile, lane, is_identity);
  }

  double positive = 0.0;
  double negative = 0.0;
  if (first_valid) {
    positive = fmax(fmax(positive, first.low), first.high);
    negative = fmax(fmax(negative, -first.low), -first.high);
  }
  if (second_valid) {
    positive = fmax(fmax(positive, second.low), second.high);
    negative = fmax(fmax(negative, -second.low), -second.high);
  }
  const unsigned int full_warp = 0xffffffff;
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    positive = fmax(positive, __shfl_down_sync(full_warp, positive, offset));
    negative = fmax(negative, __shfl_down_sync(full_warp, negative, offset));
  }
  if (lane == 0) {
    warp_positive[warp] = positive;
    warp_negative[warp] = negative;
  }
  __syncthreads();
  if (warp == 0) {
    const int warp_count = (blockDim.x + 31) >> 5;
    positive = lane < warp_count ? warp_positive[lane] : 0.0;
    negative = lane < warp_count ? warp_negative[lane] : 0.0;
    // The launcher uses at most eight warps.  The remaining lanes are zero,
    // the identity for these nonnegative maxima, so offsets 16 and 8 cannot
    // affect lane zero's result.
#pragma unroll
    for (int offset = 4; offset > 0; offset >>= 1) {
      positive = fmax(positive, __shfl_down_sync(full_warp, positive, offset));
      negative = fmax(negative, __shfl_down_sync(full_warp, negative, offset));
    }
    if (lane == 0) {
      // Both maxima start at zero, so all-zero rows already select the same
      // 1e-12 floor without a separate nonzero reduction.
      selected_scale = fmax(fmax(positive / 127.0, negative / 128.0), 1.0e-12);
      scales[m_tile * n_k + k_tile] = static_cast<float>(selected_scale);
    }
  }
  __syncthreads();
  const int64_t first_base = TileMajorA
      ? (static_cast<int64_t>(m_tile) * n_k + k_tile) * 16 * kTile + warp * kTile
      : static_cast<int64_t>(row_start + warp) * (n_k * kTile) + k_tile * kTile;
  const int64_t second_base = TileMajorA
      ? (static_cast<int64_t>(m_tile) * n_k + k_tile) * 16 * kTile +
          (warp + warps) * kTile
      : static_cast<int64_t>(row_start + warp + warps) * (n_k * kTile) +
          k_tile * kTile;
  if (first_valid) {
    x_quantized[first_base + lane] = quantize_s8_rn(first.low, selected_scale);
    x_quantized[first_base + lane + 32] = quantize_s8_rn(first.high, selected_scale);
  }
  if (second_valid) {
    x_quantized[second_base + lane] = quantize_s8_rn(second.low, selected_scale);
    x_quantized[second_base + lane + 32] = quantize_s8_rn(second.high, selected_scale);
  }
}

template <int StateMode>
__device__ __forceinline__ bool tile_is_w4(
    const uint8_t* state_bits, int64_t linear_tile) {
  if constexpr (StateMode == 0) {
    return true;
  } else if constexpr (StateMode == 1) {
    return false;
  } else {
    return ((state_bits[linear_tile >> 3] >> (linear_tile & 7)) & 1) != 0;
  }
}

template <int StateMode>
__global__ void w4w8_a8_decode_kernel(
    const int8_t* __restrict__ x,
    const float* __restrict__ activation_scales,
    const uint8_t* __restrict__ state_bits,
    const uint8_t* __restrict__ w4_payload,
    const float* __restrict__ w4_scales,
    const int8_t* __restrict__ w8_payload,
    const float* __restrict__ w8_scales,
    const int32_t* __restrict__ w4_offsets,
    const int32_t* __restrict__ w8_offsets,
    const int32_t* __restrict__ u_metadata,
    float* __restrict__ output,
    int n_k) {
  __shared__ int8_t activation[kTile];
  __shared__ int8_t weight[kTile * kTile];  // CUTLASS LayoutB physical order
  __shared__ int32_t integer_partial[4 * kTile];
  __shared__ int32_t transform_metadata[4 * kTile];
  __shared__ double values[kTile];
  const int thread = threadIdx.x;
  const int lane = thread & 63;
  const int k_part = thread >> 6;
  const int j = static_cast<int>(blockIdx.x);
  const int output_base = j * kTile;
  const PrefillLayoutB layout_b = PrefillLayoutB::packed({kTile, kTile});
  int w4_cursor = w4_offsets[j];
  int w8_cursor = w8_offsets[j];
  float accumulator = 0.0f;
  for (int index = thread; index < 4 * kTile; index += blockDim.x) {
    transform_metadata[index] = u_metadata[j * 4 * kTile + index];
  }
  __syncthreads();

  for (int k_tile = 0; k_tile < n_k; ++k_tile) {
    if (thread < kTile) {
      activation[thread] = x[k_tile * kTile + thread];
    }
    const bool is_w4 = tile_is_w4<StateMode>(state_bits, j * n_k + k_tile);
    if (is_w4) {
      const uint8_t* tile = w4_payload + static_cast<int64_t>(w4_cursor) * kTile * 32;
      stage_layout_b_w4(tile, weight, thread);
    } else {
      const int8_t* tile = w8_payload + static_cast<int64_t>(w8_cursor) * kTile * kTile;
      stage_layout_b_w8(tile, weight, thread);
    }
    __syncthreads();

    int32_t integer = 0;
#pragma unroll
    const int kk_begin = k_part * 16;
    const int kk_end = (k_part + 1) * 16;
    for (int kk = kk_begin; kk < kk_end; kk += 4) {
      integer = __dp4a(
          pack_s8x4(activation[kk], activation[kk + 1], activation[kk + 2], activation[kk + 3]),
          pack_s8x4(
              weight[layout_b({kk, lane})],
              weight[layout_b({kk + 1, lane})],
              weight[layout_b({kk + 2, lane})],
              weight[layout_b({kk + 3, lane})]),
          integer);
    }
    integer_partial[k_part * kTile + lane] = integer;
    __syncthreads();
    const float weight_scale = is_w4 ? w4_scales[w4_cursor++] : w8_scales[w8_cursor++];
    if (thread < kTile) {
      integer = integer_partial[thread] + integer_partial[kTile + thread] +
          integer_partial[2 * kTile + thread] + integer_partial[3 * kTile + thread];
      const float tile_scale = activation_scales[k_tile] * weight_scale;
      const float scaled = __fmul_rn(static_cast<float>(integer), tile_scale);
      accumulator = __fadd_rn(accumulator, scaled);
    }
    __syncthreads();
  }
  if (thread < kTile) {
    values[thread] = static_cast<double>(accumulator);
  }
  __syncthreads();
  inverse_u64_inplace(values, 1, transform_metadata, thread);
  if (thread < kTile) {
    output[output_base + thread] = inverse_u64_value(values, transform_metadata, thread);
  }
}

template <int StateMode>
__global__ void w4w8_a8_prefill_kernel(
    const int8_t* __restrict__ x,
    const float* __restrict__ activation_scales,
    const uint8_t* __restrict__ state_bits,
    const uint8_t* __restrict__ w4_payload,
    const float* __restrict__ w4_scales,
    const int8_t* __restrict__ w8_payload,
    const float* __restrict__ w8_scales,
    const int32_t* __restrict__ w4_offsets,
    const int32_t* __restrict__ w8_offsets,
    const int32_t* __restrict__ u_metadata,
    float* __restrict__ output,
    int n_k,
    int n_cols) {
  __shared__ __align__(16) int8_t activation[16 * kTile];
  __shared__ __align__(16) int8_t weight[kTile * kTile];
  __shared__ int32_t transform_metadata[4 * kTile];
  __shared__ double values[16 * kTile];
  const int thread = threadIdx.x;
  const int warp = thread >> 5;
  const int j = static_cast<int>(blockIdx.x);
  const int m_tile = blockIdx.y;
  const int row_start = m_tile * 16;
  int w4_cursor = w4_offsets[j];
  int w8_cursor = w8_offsets[j];
  float accumulators[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  const PrefillLayoutA layout_a = PrefillLayoutA::packed({16, kTile});
  const PrefillLayoutB layout_b = PrefillLayoutB::packed({kTile, kTile});
  for (int index = thread; index < 4 * kTile; index += blockDim.x) {
    transform_metadata[index] = u_metadata[j * 4 * kTile + index];
  }
  __syncthreads();

  for (int k_tile = 0; k_tile < n_k; ++k_tile) {
    for (int index = thread; index < 16 * kTile; index += blockDim.x) {
      const int row = index / kTile;
      const int kk = index % kTile;
      activation[layout_a({row, kk})] =
          x[(row_start + row) * (n_k * kTile) + k_tile * kTile + kk];
    }
    const bool is_w4 = tile_is_w4<StateMode>(state_bits, j * n_k + k_tile);
    if (is_w4) {
      const uint8_t* tile = w4_payload + static_cast<int64_t>(w4_cursor) * kTile * 32;
      stage_layout_b_w4(tile, weight, thread);
    } else {
      const int8_t* tile = w8_payload + static_cast<int64_t>(w8_cursor) * kTile * kTile;
      stage_layout_b_w8(tile, weight, thread);
    }
    __syncthreads();

    typename PrefillMma::IteratorA iterator_a(
        {activation, layout_a}, thread & 31);
    typename PrefillMma::IteratorB iterator_b(
        {weight, layout_b}, thread & 31);
    iterator_b.add_tile_offset({0, warp});
    typename PrefillMma::FragmentA a_fragment;
    typename PrefillMma::FragmentB b_fragment;
    typename PrefillMma::FragmentC c_fragment;
    c_fragment.clear();
    PrefillMma mma;
#pragma unroll
    for (int k32 = 0; k32 < 2; ++k32) {
      iterator_a.load(a_fragment);
      iterator_b.load(b_fragment);
      ++iterator_a;
      ++iterator_b;
      mma(c_fragment, a_fragment, b_fragment, c_fragment);
    }
    const float scale = activation_scales[m_tile * n_k + k_tile] *
        (is_w4 ? w4_scales[w4_cursor++] : w8_scales[w8_cursor++]);
#pragma unroll
    for (int item = 0; item < 8; ++item) {
      const float scaled = __fmul_rn(static_cast<float>(c_fragment[item]), scale);
      accumulators[item] = __fadd_rn(accumulators[item], scaled);
    }
    __syncthreads();
  }

  for (int item = 0; item < 8; ++item) {
    const int lane = thread & 31;
    const int row = (lane >> 2) + 8 * ((item / 2) & 1);
    const int n = 16 * warp + 2 * (lane & 3) + 8 * (item / 4) + (item & 1);
    values[row * kTile + n] = static_cast<double>(accumulators[item]);
  }
  __syncthreads();
  inverse_u64_inplace(values, 16, transform_metadata, thread);
  for (int index = thread; index < 16 * kTile; index += blockDim.x) {
    const int row = index / kTile;
    const int n = index % kTile;
    output[(row_start + row) * n_cols + j * kTile + n] =
        inverse_u64_value(values, transform_metadata, index);
  }
}

// Long prefill execution macro-tile.  The numerical contract remains four
// independent M16xK64 activation tiles, while one CTA reuses every N64xK64
// weight tile across all four M16 groups.
template <int StateMode, typename OutputT, bool FullRows, bool TileMajorA>
__global__ __launch_bounds__(256, 4) void w4w8_a8_prefill_m64_kernel(
    const int8_t* __restrict__ x,
    const float* __restrict__ activation_scales,
    const uint8_t* __restrict__ state_bits,
    const uint8_t* __restrict__ w4_payload,
    const float* __restrict__ w4_scales,
    const int8_t* __restrict__ w8_payload,
    const float* __restrict__ w8_scales,
    const int32_t* __restrict__ w4_offsets,
    const int32_t* __restrict__ w8_offsets,
    const int32_t* __restrict__ u_metadata,
    OutputT* __restrict__ output,
    int n_rows,
    int n_k,
  int n_cols) {
  constexpr int kRows = 64;
  using LayoutA = typename std::conditional<StateMode == 1,
      PrefillM64W8LayoutA, PrefillLayoutA>::type;
  using Mma = typename cutlass::gemm::warp::DefaultMmaTensorOp<
      PrefillMacroWarpShape, PrefillInstructionShape, int8_t, LayoutA,
      int8_t, PrefillLayoutB, int32_t, cutlass::layout::RowMajor,
      cutlass::arch::OpMultiplyAddSaturate>::Type;
  static_assert(std::is_same<typename Mma::LayoutC,
                             cutlass::layout::RowMajor>::value &&
                    Mma::FragmentC::kElements == 16,
                "M64 register epilogue requires the original accumulator layout");
  // Two [A4KiB, expanded-W4/W8 4KiB] stages plus two packed-W4 2KiB
  // stages reuse the 16KiB FP32 inverse-U handoff after GEMM.
  constexpr int kWorkspaceBytes = StateMode == 1 ? 16 * 1024 : 20 * 1024;
  __shared__ __align__(16) unsigned char workspace[kWorkspaceBytes];
  int8_t* activation = reinterpret_cast<int8_t*>(workspace);
  int8_t* weight = activation + kRows * kTile;
  uint8_t* packed_w4 = reinterpret_cast<uint8_t*>(workspace + 16 * 1024);
  float* values = reinterpret_cast<float*>(workspace);
  __shared__ int32_t transform_metadata[4 * kTile];
  const int thread = threadIdx.x;
  const int warp = thread >> 5;
  const int lane = thread & 31;
  const int warp_m = warp >> 1;
  const int warp_n = warp & 1;
  const int m_blocks = (n_rows + kRows - 1) / kRows;
  const int group_m = StateMode == 1
      ? (m_blocks >= 32 ? 32 : m_blocks >= 16 ? 16 : m_blocks >= 8 ? 8 : 1)
      : 1;
  const int j = static_cast<int>(blockIdx.x) / group_m;
  const int m_tile = static_cast<int>(blockIdx.y) * group_m +
                     static_cast<int>(blockIdx.x) % group_m;
  const int row_start = m_tile * kRows;
  if (row_start >= n_rows) {
    return;
  }
  int w4_cursor = 0;
  int w8_cursor = 0;
  if constexpr (StateMode != 1) {
    w4_cursor = w4_offsets[j];
  }
  if constexpr (StateMode != 0) {
    w8_cursor = w8_offsets[j];
  }
  float accumulators[16] = {
      0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
      0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  float current_weight_scale = 0.0f;
  const LayoutA layout_a = LayoutA::packed({kRows, kTile});
  const PrefillLayoutB layout_b = PrefillLayoutB::packed({kTile, kTile});
  for (int index = thread; index < 4 * kTile; index += blockDim.x) {
    transform_metadata[index] = u_metadata[j * 4 * kTile + index];
  }
  __syncthreads();

  const bool first_is_w4 = tile_is_w4<StateMode>(state_bits, j * n_k);
  int payload_tile;
  if (first_is_w4) {
    payload_tile = w4_cursor++;
    current_weight_scale = w4_scales[payload_tile];
  } else {
    payload_tile = w8_cursor++;
    current_weight_scale = w8_scales[payload_tile];
  }
  if constexpr (StateMode == 1) {
    stage_m64_w8_cp_async<FullRows, TileMajorA>(
        x, w8_payload, activation, weight, layout_a, row_start, n_rows, n_k,
        payload_tile, 0, thread);
  } else {
    stage_m64_w4w8_cp_async<FullRows, TileMajorA>(
        x, w4_payload, w8_payload, activation, weight, packed_w4, layout_a,
        row_start, n_rows, n_k, first_is_w4, payload_tile, 0, thread);
  }
  cutlass::arch::cp_async_fence();
  cutlass::arch::cp_async_wait<0>();
  __syncthreads();
  if (first_is_w4) {
    stage_layout_b_w4(packed_w4, weight, thread);
    __syncthreads();
  }

#pragma unroll 4
  for (int k_tile = 0; k_tile < n_k; ++k_tile) {
    bool next_is_w4 = false;
    float next_weight_scale = 0.0f;
    int8_t* tile_activation = activation;
    int8_t* tile_weight = weight;
    const int stage = k_tile & 1;
    tile_activation += stage * 2 * kTile * kTile;
    tile_weight = tile_activation + kRows * kTile;
    if (k_tile + 1 < n_k) {
      const int next_stage = stage ^ 1;
      int8_t* next_activation =
          activation + next_stage * 2 * kTile * kTile;
      int8_t* next_weight = next_activation + kRows * kTile;
      uint8_t* next_packed_w4 = packed_w4 + next_stage * kTile * 32;
      next_is_w4 = tile_is_w4<StateMode>(state_bits, j * n_k + k_tile + 1);
      int payload_tile;
      if (next_is_w4) {
        payload_tile = w4_cursor++;
        next_weight_scale = w4_scales[payload_tile];
      } else {
        payload_tile = w8_cursor++;
        next_weight_scale = w8_scales[payload_tile];
      }
      if constexpr (StateMode == 1) {
        stage_m64_w8_cp_async<FullRows, TileMajorA>(
            x, w8_payload, next_activation, next_weight, layout_a, row_start,
            n_rows, n_k, payload_tile, k_tile + 1, thread);
      } else {
        stage_m64_w4w8_cp_async<FullRows, TileMajorA>(
            x, w4_payload, w8_payload, next_activation, next_weight,
            next_packed_w4, layout_a, row_start, n_rows, n_k,
            next_is_w4, payload_tile, k_tile + 1, thread);
      }
      cutlass::arch::cp_async_fence();
    }

    typename Mma::IteratorA iterator_a(
        {tile_activation, layout_a}, thread & 31);
    typename Mma::IteratorB iterator_b(
        {tile_weight, layout_b}, thread & 31);
    iterator_a.add_tile_offset({warp_m, 0});
    iterator_b.add_tile_offset({0, warp_n});
    typename Mma::FragmentA a_fragment;
    typename Mma::FragmentB b_fragment;
    typename Mma::FragmentC c_fragment;
    c_fragment.clear();
    Mma mma;
#pragma unroll
    for (int k32 = 0; k32 < 2; ++k32) {
      iterator_a.load(a_fragment);
      iterator_b.load(b_fragment);
      ++iterator_a;
      ++iterator_b;
      mma(c_fragment, a_fragment, b_fragment, c_fragment);
    }
    const float weight_scale = current_weight_scale;
    // Rows are 16-aligned, so every accumulator owned by this warp belongs
    // to one M16 group and shares both its validity and K64 scale.
    const int group_first_row = row_start + 16 * warp_m;
    if (FullRows || group_first_row < n_rows) {
      const float scale =
          activation_scales[(group_first_row / 16) * n_k + k_tile] * weight_scale;
#pragma unroll
      for (int item = 0; item < 16; ++item) {
        accumulators[item] = __fadd_rn(
            accumulators[item],
            __fmul_rn(static_cast<float>(c_fragment[item]), scale));
      }
    }
    if (k_tile + 1 < n_k) {
      cutlass::arch::cp_async_wait<0>();
      __syncthreads();
      const int next_stage = stage ^ 1;
      if (next_is_w4) {
        int8_t* next_activation =
            activation + next_stage * 2 * kTile * kTile;
        int8_t* next_weight = next_activation + kRows * kTile;
        uint8_t* next_packed_w4 = packed_w4 + next_stage * kTile * 32;
        stage_layout_b_w4(next_packed_w4, next_weight, thread);
        __syncthreads();
      }
      current_weight_scale = next_weight_scale;
    } else {
      __syncthreads();
    }
  }

  const bool has_inverse_u = transform_metadata[0] != 0;
#pragma unroll
  for (int item = 0; item < 16; ++item) {
    const int row =
        16 * warp_m + (lane >> 2) + 8 * ((item / 2) & 1);
    const int n =
        32 * warp_n + 2 * (lane & 3) + 8 * (item / 4) + (item & 1);
    values[row * kTile + n] = accumulators[item];
  }
  // All accumulator fragments reach row-major shared storage before a warp
  // reads either half of a row owned by its partner N warp.
  __syncthreads();
  const unsigned int full_warp = 0xffffffff;
  for (int row = warp; row < kRows; row += blockDim.x / 32) {
    const int global_row = row_start + row;
    if (FullRows || global_row < n_rows) {
      double low = static_cast<double>(values[row * kTile + lane]);
      double high = static_cast<double>(values[row * kTile + lane + 32]);
      float output_low;
      float output_high;
      if (has_inverse_u) {
        low *= static_cast<double>(transform_metadata[3 * kTile + lane]);
        high *= static_cast<double>(transform_metadata[3 * kTile + lane + 32]);
#pragma unroll
        for (int stride = 1; stride < 32; stride <<= 1) {
          const double low_peer = __shfl_xor_sync(full_warp, low, stride);
          const double high_peer = __shfl_xor_sync(full_warp, high, stride);
          low = (lane & stride) ? low_peer - low : low + low_peer;
          high = (lane & stride) ? high_peer - high : high + high_peer;
        }
        const double low_before_final = low;
        const double high_before_final = high;
        const double transformed_low = low_before_final + high_before_final;
        const double transformed_high = low_before_final - high_before_final;
        const int low_target = transform_metadata[kTile + lane];
        const int high_target = transform_metadata[kTile + lane + 32];
        // The inverse permutation can select either half.  Every lane
        // executes every shuffle before selecting its source target.
        const double low_from_low =
            __shfl_sync(full_warp, transformed_low, low_target & 31);
        const double low_from_high =
            __shfl_sync(full_warp, transformed_high, low_target & 31);
        const double high_from_low =
            __shfl_sync(full_warp, transformed_low, high_target & 31);
        const double high_from_high =
            __shfl_sync(full_warp, transformed_high, high_target & 31);
        const double permuted_low =
            low_target < 32 ? low_from_low : low_from_high;
        const double permuted_high =
            high_target < 32 ? high_from_low : high_from_high;
        output_low = static_cast<float>(
            permuted_low * static_cast<double>(transform_metadata[2 * kTile + lane]) * 0.125);
        output_high = static_cast<float>(
            permuted_high * static_cast<double>(transform_metadata[2 * kTile + lane + 32]) * 0.125);
      } else {
        output_low = static_cast<float>(low);
        output_high = static_cast<float>(high);
      }
      const int64_t output_base =
          static_cast<int64_t>(global_row) * n_cols + j * kTile;
      store_m64_output(output, output_base + lane, output_low);
      store_m64_output(output, output_base + lane + 32, output_high);
    }
  }
}

// Exact-contract M16 split-K feasibility path.  The precision tile remains
// 64x64 and every K64 product is scaled before FP32 accumulation.  Each CTA
// owns one (N64, M16, K-range) and writes a private partial-output plane;
// the fixed-order reducer below avoids atomicAdd and preserves reproducibility.
template <int StateMode>
__global__ void w4w8_a8_prefill_splitk_kernel(
    const int8_t* __restrict__ x,
    const float* __restrict__ activation_scales,
    const uint8_t* __restrict__ state_bits,
    const uint8_t* __restrict__ w4_payload,
    const float* __restrict__ w4_scales,
    const int8_t* __restrict__ w8_payload,
    const float* __restrict__ w8_scales,
    const int32_t* __restrict__ w4_offsets,
    const int32_t* __restrict__ w8_offsets,
    float* __restrict__ partial,
    int n_rows,
    int n_cols,
    int n_k,
    int split_k) {
  __shared__ __align__(16) int8_t activation[2 * 16 * kTile];
  __shared__ __align__(16) int8_t weight[2 * kTile * kTile];
  __shared__ __align__(16) uint8_t packed_w4[2 * kTile * 32];
  bool stage_is_w4[2] = {false, false};
  float stage_weight_scale[2] = {0.0f, 0.0f};
  const int thread = threadIdx.x;
  const int warp = thread >> 5;
  const int j = static_cast<int>(blockIdx.x);
  const int m_tile = static_cast<int>(blockIdx.y);
  const int split = static_cast<int>(blockIdx.z);
  const int row_start = m_tile * 16;
  const int k_begin = (n_k * split) / split_k;
  const int k_end = (n_k * (split + 1)) / split_k;
  __shared__ int initial_w4_cursor;
  __shared__ int initial_w8_cursor;
  // Compact W4/W8 arenas are row-major in K.  Advance both cursors to the
  // split boundary by scanning only the preceding state bits.
  if (thread == 0) {
    initial_w4_cursor = w4_offsets[j];
    initial_w8_cursor = w8_offsets[j];
    if constexpr (StateMode == 0) {
      initial_w4_cursor += k_begin;
    } else if constexpr (StateMode == 1) {
      initial_w8_cursor += k_begin;
    } else {
      const int64_t first_bit = static_cast<int64_t>(j) * n_k;
      const int64_t end_bit = first_bit + k_begin;
      int preceding_w4 = 0;
      if (k_begin != 0) {
        const int64_t first_byte = first_bit >> 3;
        const int64_t last_byte = (end_bit - 1) >> 3;
        for (int64_t byte = first_byte; byte <= last_byte; ++byte) {
          unsigned int mask = 0xffu;
          if (byte == first_byte) {
            mask &= 0xffu << (first_bit & 7);
          }
          if (byte == last_byte) {
            mask &= (1u << (((end_bit - 1) & 7) + 1)) - 1u;
          }
          preceding_w4 += __popc(
              static_cast<unsigned int>(state_bits[byte]) & mask);
        }
      }
      initial_w4_cursor += preceding_w4;
      initial_w8_cursor += k_begin - preceding_w4;
    }
  }
  __syncthreads();
  int w4_cursor = initial_w4_cursor;
  int w8_cursor = initial_w8_cursor;
  float accumulators[8] = {0.0f, 0.0f, 0.0f, 0.0f,
                           0.0f, 0.0f, 0.0f, 0.0f};
  const PrefillLayoutA layout_a = PrefillLayoutA::packed({16, kTile});
  const PrefillLayoutB layout_b = PrefillLayoutB::packed({kTile, kTile});

  stage_is_w4[0] = tile_is_w4<StateMode>(state_bits, j * n_k + k_begin);
  int payload_tile = 0;
  if (stage_is_w4[0]) {
    payload_tile = w4_cursor++;
    stage_weight_scale[0] = w4_scales[payload_tile];
  } else {
    payload_tile = w8_cursor++;
    stage_weight_scale[0] = w8_scales[payload_tile];
  }
  stage_m16_splitk_cp_async(x, w4_payload, w8_payload, activation, weight,
      packed_w4, layout_a, row_start, n_k, stage_is_w4[0], payload_tile, k_begin, thread);
  cutlass::arch::cp_async_fence();
  cutlass::arch::cp_async_wait<0>();
  __syncthreads();
  if (stage_is_w4[0]) {
    stage_layout_b_w4(packed_w4, weight, thread);
    __syncthreads();
  }
  for (int k_tile = k_begin; k_tile < k_end; ++k_tile) {
    const int stage = (k_tile - k_begin) & 1;
    int8_t* tile_activation = activation + stage * 16 * kTile;
    int8_t* tile_weight = weight + stage * kTile * kTile;
    if (k_tile + 1 < k_end) {
      const int next_stage = stage ^ 1;
      stage_is_w4[next_stage] = tile_is_w4<StateMode>(state_bits, j * n_k + k_tile + 1);
      if (stage_is_w4[next_stage]) {
        payload_tile = w4_cursor++;
        stage_weight_scale[next_stage] = w4_scales[payload_tile];
      } else {
        payload_tile = w8_cursor++;
        stage_weight_scale[next_stage] = w8_scales[payload_tile];
      }
      stage_m16_splitk_cp_async(x, w4_payload, w8_payload,
          activation + next_stage * 16 * kTile, weight + next_stage * kTile * kTile,
          packed_w4 + next_stage * kTile * 32, layout_a, row_start, n_k,
          stage_is_w4[next_stage], payload_tile, k_tile + 1, thread);
      cutlass::arch::cp_async_fence();
    }

    typename PrefillMma::IteratorA iterator_a(
        {tile_activation, layout_a}, thread & 31);
    typename PrefillMma::IteratorB iterator_b(
        {tile_weight, layout_b}, thread & 31);
    iterator_b.add_tile_offset({0, warp});
    typename PrefillMma::FragmentA a_fragment;
    typename PrefillMma::FragmentB b_fragment;
    typename PrefillMma::FragmentC c_fragment;
    c_fragment.clear();
    PrefillMma mma;
#pragma unroll
    for (int k32 = 0; k32 < 2; ++k32) {
      iterator_a.load(a_fragment);
      iterator_b.load(b_fragment);
      ++iterator_a;
      ++iterator_b;
      mma(c_fragment, a_fragment, b_fragment, c_fragment);
    }
    const float scale = activation_scales[m_tile * n_k + k_tile] *
        stage_weight_scale[stage];
#pragma unroll
    for (int item = 0; item < 8; ++item) {
      accumulators[item] = __fadd_rn(
          accumulators[item], __fmul_rn(static_cast<float>(c_fragment[item]), scale));
    }
    if (k_tile + 1 < k_end) {
      cutlass::arch::cp_async_wait<0>();
      __syncthreads();
      const int next_stage = stage ^ 1;
      if (stage_is_w4[next_stage]) {
        stage_layout_b_w4(packed_w4 + next_stage * kTile * 32,
            weight + next_stage * kTile * kTile, thread);
        __syncthreads();
      }
    } else {
      __syncthreads();
    }
  }

#pragma unroll
  for (int item = 0; item < 8; ++item) {
    const int lane = thread & 31;
    const int row = (lane >> 2) + 8 * ((item / 2) & 1);
    const int n = 16 * warp + 2 * (lane & 3) + 8 * (item / 4) + (item & 1);
    partial[(static_cast<int64_t>(split) * n_rows + row_start + row) * n_cols +
            j * kTile + n] = accumulators[item];
  }
}

// Exact-contract M1 split-K path.  This deliberately uses the same N64
// execution tile and compact W4/W8 arenas as the regular decode kernel.  Each
// (N64, K-range) CTA writes one private FP32 partial plane; the common fixed-
// order reducer below performs the only cross-CTA reduction.
template <int StateMode>
__global__ void w4w8_a8_decode_splitk_kernel(
    const int8_t* __restrict__ x,
    const float* __restrict__ activation_scales,
    const uint8_t* __restrict__ state_bits,
    const uint8_t* __restrict__ w4_payload,
    const float* __restrict__ w4_scales,
    const int8_t* __restrict__ w8_payload,
    const float* __restrict__ w8_scales,
    const int32_t* __restrict__ w4_offsets,
    const int32_t* __restrict__ w8_offsets,
    float* __restrict__ partial,
    int n_j,
    int n_k,
    int split_k) {
  const int thread = threadIdx.x;
  const int column = thread >> 2;
  const int k_part = thread & 3;
  const int j = static_cast<int>(blockIdx.x);
  const int split = static_cast<int>(blockIdx.y);
  const PrefillLayoutB layout_b = PrefillLayoutB::packed({kTile, kTile});
  const int k_begin = (n_k * split) / split_k;
  const int k_end = (n_k * (split + 1)) / split_k;

  // Compact payloads are row-major in K.  Every thread establishes and
  // advances the same private cursors, removing the cursor hand-off barriers.
  int w4_cursor = w4_offsets[j];
  int w8_cursor = w8_offsets[j];
  if constexpr (StateMode == 0) {
    w4_cursor += k_begin;
  } else if constexpr (StateMode == 1) {
    w8_cursor += k_begin;
  } else {
    for (int k_tile = 0; k_tile < k_begin; ++k_tile) {
      if (tile_is_w4<StateMode>(state_bits, j * n_k + k_tile)) {
        ++w4_cursor;
      } else {
        ++w8_cursor;
      }
    }
  }
  float accumulator = 0.0f;

  for (int k_tile = k_begin; k_tile < k_end; ++k_tile) {
    const int8_t* activation_tile = x + k_tile * kTile;
    const bool is_w4 = tile_is_w4<StateMode>(state_bits, j * n_k + k_tile);
    const uint8_t* w4_tile = is_w4
        ? w4_payload + static_cast<int64_t>(w4_cursor) * kTile * 32
        : nullptr;
    const int8_t* w8_tile = is_w4
        ? nullptr
        : w8_payload + static_cast<int64_t>(w8_cursor) * kTile * kTile;

    int32_t integer = 0;
    if (is_w4) {
      // Consecutive K values inside one K16 fragment are consecutive in the
      // offline Tensor Core layout.  Each thread consumes one output column
      // directly from packed W4 without widening the complete tile.
      const int kk_begin = k_part * 16;
#pragma unroll
      for (int kk = kk_begin; kk < kk_begin + 16; kk += 4) {
        integer = __dp4a(
            load_packed_s8x4(activation_tile, kk),
            load_physical_int4x4(w4_tile, layout_b({kk, column})),
            integer);
      }
    } else {
#pragma unroll
      for (int kk = k_part * 16; kk < (k_part + 1) * 16; kk += 4) {
        integer = __dp4a(
            load_packed_s8x4(activation_tile, kk),
            load_packed_s8x4(w8_tile, layout_b({kk, column})),
            integer);
      }
    }
    // Adjacent lanes own the four K16 fragments of one output column.  XOR
    // reductions stay within each four-lane subgroup of the hardware warp.
    integer += __shfl_xor_sync(0xffffffff, integer, 1);
    integer += __shfl_xor_sync(0xffffffff, integer, 2);
    if (k_part == 0) {
      const float weight_scale = is_w4 ? w4_scales[w4_cursor]
                                      : w8_scales[w8_cursor];
      accumulator = __fadd_rn(
          accumulator,
          static_cast<float>(integer) *
              (activation_scales[k_tile] * weight_scale));
    }
    if (is_w4) {
      ++w4_cursor;
    } else {
      ++w8_cursor;
    }
  }
  if (k_part == 0) {
    partial[(static_cast<int64_t>(split) * n_j + j) * kTile + column] =
        accumulator;
  }
}

__global__ void w4w8_a8_splitk_reduce_kernel(
    const float* __restrict__ partial,
    const int32_t* __restrict__ u_metadata,
    float* __restrict__ output,
    int split_k,
    int n_j,
    int n_rows,
    int n_cols) {
  __shared__ int32_t transform_metadata[4 * kTile];
  __shared__ double values[16 * kTile];
  const int thread = threadIdx.x;
  const int owner = static_cast<int>(blockIdx.x);
  const int j = owner % n_j;
  const int m_tile = owner / n_j;
  const int tile_rows = n_rows == 1 ? 1 : 16;
  const int row_start = m_tile * 16;
  for (int index = thread; index < 4 * kTile; index += blockDim.x) {
    transform_metadata[index] = u_metadata[j * 4 * kTile + index];
  }
  for (int index = thread; index < tile_rows * kTile; index += blockDim.x) {
    const int row = index / kTile;
    const int col = index % kTile;
    const int64_t output_index = static_cast<int64_t>(row_start + row) * n_cols +
        j * kTile + col;
    float value = partial[output_index];
    const int64_t plane = static_cast<int64_t>(n_rows) * n_cols;
    for (int split = 1; split < split_k; ++split) {
      value = __fadd_rn(value, partial[static_cast<int64_t>(split) * plane + output_index]);
    }
    values[index] = static_cast<double>(value);
  }
  __syncthreads();
  inverse_u64_inplace(values, tile_rows, transform_metadata, thread);
  for (int index = thread; index < tile_rows * kTile; index += blockDim.x) {
    const int row = index / kTile;
    const int col = index % kTile;
    output[static_cast<int64_t>(row_start + row) * n_cols + j * kTile + col] =
        inverse_u64_value(values, transform_metadata, index);
  }
}

template <int StateMode>
void launch(
    torch::Tensor x,
    torch::Tensor activation_scales,
    torch::Tensor state_bits,
    torch::Tensor w4_payload,
    torch::Tensor w4_scales,
    torch::Tensor w8_payload,
    torch::Tensor w8_scales,
    torch::Tensor w4_offsets,
    torch::Tensor w8_offsets,
    torch::Tensor u_metadata,
    torch::Tensor output,
    int n_j,
    int n_k,
    bool force_m64,
    bool output_bf16) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int n_rows = x.dim() == 4 ? static_cast<int>(x.size(0) * 16)
                                  : static_cast<int>(x.size(0));
  if (n_rows == 1) {
    w4w8_a8_decode_kernel<StateMode><<<n_j, 256, 0, stream>>>(
        x.data_ptr<int8_t>(), activation_scales.data_ptr<float>(),
        state_bits.data_ptr<uint8_t>(), w4_payload.data_ptr<uint8_t>(),
        w4_scales.data_ptr<float>(), w8_payload.data_ptr<int8_t>(),
        w8_scales.data_ptr<float>(), w4_offsets.data_ptr<int32_t>(),
        w8_offsets.data_ptr<int32_t>(), u_metadata.data_ptr<int32_t>(),
        output.data_ptr<float>(), n_k);
  } else if (force_m64 || n_rows >= 128) {
    const unsigned int m_blocks =
        static_cast<unsigned int>((n_rows + 63) / 64);
    const unsigned int group_m = StateMode == 1
        ? (m_blocks >= 32 ? 32 : m_blocks >= 16 ? 16 : m_blocks >= 8 ? 8 : 1)
        : 1;
    const dim3 grid(
        static_cast<unsigned int>(n_j) * group_m,
        (m_blocks + group_m - 1) / group_m);
    const auto launch_m64 = [&](auto full_rows_tag, auto tile_major_tag) {
      constexpr bool FullRows = decltype(full_rows_tag)::value;
      constexpr bool TileMajorA = decltype(tile_major_tag)::value;
      if (output_bf16) {
      check_cuda(
          cudaFuncSetAttribute(
              w4w8_a8_prefill_m64_kernel<StateMode, uint16_t, FullRows, TileMajorA>,
              cudaFuncAttributePreferredSharedMemoryCarveout,
              cudaSharedmemCarveoutMaxShared),
          "w4w8_a8 BF16 M64 shared-memory carveout");
      w4w8_a8_prefill_m64_kernel<StateMode, uint16_t, FullRows, TileMajorA><<<grid, 256, 0, stream>>>(
          x.data_ptr<int8_t>(), activation_scales.data_ptr<float>(),
          state_bits.data_ptr<uint8_t>(), w4_payload.data_ptr<uint8_t>(),
          w4_scales.data_ptr<float>(), w8_payload.data_ptr<int8_t>(),
          w8_scales.data_ptr<float>(), w4_offsets.data_ptr<int32_t>(),
          w8_offsets.data_ptr<int32_t>(), u_metadata.data_ptr<int32_t>(),
          reinterpret_cast<uint16_t*>(output.data_ptr<at::BFloat16>()),
          n_rows, n_k, n_j * kTile);
    } else {
      check_cuda(
          cudaFuncSetAttribute(
              w4w8_a8_prefill_m64_kernel<StateMode, float, FullRows, TileMajorA>,
              cudaFuncAttributePreferredSharedMemoryCarveout,
              cudaSharedmemCarveoutMaxShared),
          "w4w8_a8 M64 shared-memory carveout");
      w4w8_a8_prefill_m64_kernel<StateMode, float, FullRows, TileMajorA><<<grid, 256, 0, stream>>>(
          x.data_ptr<int8_t>(), activation_scales.data_ptr<float>(),
          state_bits.data_ptr<uint8_t>(), w4_payload.data_ptr<uint8_t>(),
          w4_scales.data_ptr<float>(), w8_payload.data_ptr<int8_t>(),
          w8_scales.data_ptr<float>(), w4_offsets.data_ptr<int32_t>(),
          w8_offsets.data_ptr<int32_t>(), u_metadata.data_ptr<int32_t>(),
          output.data_ptr<float>(), n_rows, n_k,
          n_j * kTile);
      }
    };
    if (n_rows % 64 == 0) {
      if (x.dim() == 4) {
        launch_m64(std::true_type{}, std::true_type{});
      } else {
        launch_m64(std::true_type{}, std::false_type{});
      }
    } else {
      if (x.dim() == 4) {
        launch_m64(std::false_type{}, std::true_type{});
      } else {
        launch_m64(std::false_type{}, std::false_type{});
      }
    }
  } else {
    const dim3 grid(n_j, static_cast<unsigned int>(n_rows / 16));
    w4w8_a8_prefill_kernel<StateMode><<<grid, 128, 0, stream>>>(
        x.data_ptr<int8_t>(), activation_scales.data_ptr<float>(),
        state_bits.data_ptr<uint8_t>(), w4_payload.data_ptr<uint8_t>(),
        w4_scales.data_ptr<float>(), w8_payload.data_ptr<int8_t>(),
        w8_scales.data_ptr<float>(), w4_offsets.data_ptr<int32_t>(),
        w8_offsets.data_ptr<int32_t>(), u_metadata.data_ptr<int32_t>(),
        output.data_ptr<float>(), n_k, n_j * kTile);
  }
}

template <int StateMode>
void launch_splitk(
    torch::Tensor x,
    torch::Tensor activation_scales,
    torch::Tensor state_bits,
    torch::Tensor w4_payload,
    torch::Tensor w4_scales,
    torch::Tensor w8_payload,
    torch::Tensor w8_scales,
    torch::Tensor w4_offsets,
    torch::Tensor w8_offsets,
    torch::Tensor partial,
    int n_k,
    int split_k) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int n_j = static_cast<int>(w4_offsets.numel() - 1);
  const int n_rows = static_cast<int>(x.size(0));
  const dim3 grid(
      static_cast<unsigned int>(n_j),
      static_cast<unsigned int>(n_rows / 16),
      static_cast<unsigned int>(split_k));
  w4w8_a8_prefill_splitk_kernel<StateMode><<<grid, 128, 0, stream>>>(
      x.data_ptr<int8_t>(), activation_scales.data_ptr<float>(),
      state_bits.data_ptr<uint8_t>(), w4_payload.data_ptr<uint8_t>(),
      w4_scales.data_ptr<float>(), w8_payload.data_ptr<int8_t>(),
      w8_scales.data_ptr<float>(), w4_offsets.data_ptr<int32_t>(),
      w8_offsets.data_ptr<int32_t>(), partial.data_ptr<float>(), n_rows,
      n_j * kTile, n_k, split_k);
}

template <int StateMode>
void launch_decode_splitk(
    torch::Tensor x,
    torch::Tensor activation_scales,
    torch::Tensor state_bits,
    torch::Tensor w4_payload,
    torch::Tensor w4_scales,
    torch::Tensor w8_payload,
    torch::Tensor w8_scales,
    torch::Tensor w4_offsets,
    torch::Tensor w8_offsets,
    torch::Tensor partial,
    int n_k,
    int split_k) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int n_j = static_cast<int>(w4_offsets.numel() - 1);
  const dim3 grid(static_cast<unsigned int>(n_j), static_cast<unsigned int>(split_k));
  w4w8_a8_decode_splitk_kernel<StateMode><<<grid, 256, 0, stream>>>(
      x.data_ptr<int8_t>(), activation_scales.data_ptr<float>(),
      state_bits.data_ptr<uint8_t>(), w4_payload.data_ptr<uint8_t>(),
      w4_scales.data_ptr<float>(), w8_payload.data_ptr<int8_t>(),
      w8_scales.data_ptr<float>(), w4_offsets.data_ptr<int32_t>(),
      w8_offsets.data_ptr<int32_t>(), partial.data_ptr<float>(), n_j, n_k,
      split_k);
}

void require_cuda_contiguous(const torch::Tensor& value, const char* name) {
  TORCH_CHECK(value.is_cuda(), name, " must be CUDA");
  TORCH_CHECK(value.is_contiguous(), name, " must be contiguous");
}

}  // namespace

std::vector<torch::Tensor> alignquant_activation_quantize_cuda(
    torch::Tensor x,
    torch::Tensor v_metadata,
    bool tile_major) {
  require_cuda_contiguous(x, "x");
  require_cuda_contiguous(v_metadata, "v_metadata");
  TORCH_CHECK(x.scalar_type() == torch::kBFloat16 || x.scalar_type() == torch::kHalf ||
              x.scalar_type() == torch::kFloat32, "x must be bf16, fp16, or fp32");
  TORCH_CHECK(v_metadata.scalar_type() == torch::kInt32, "v_metadata must be int32");
  TORCH_CHECK(x.dim() == 2 && (x.size(0) == 1 || (x.size(0) > 0 && x.size(0) % 16 == 0)),
              "x must be [1,K] or [M,K] with positive M divisible by 16");
  TORCH_CHECK(x.size(1) % kTile == 0, "K must be a multiple of 64");
  TORCH_CHECK(!tile_major || x.size(0) >= 128,
              "tile-major A8 output requires at least 128 rows for M64 prefill");
  const int n_k = static_cast<int>(x.size(1) / kTile);
  TORCH_CHECK(v_metadata.sizes() == torch::IntArrayRef({n_k, 4, kTile}),
              "v_metadata must be [K/64,4,64]");
  TORCH_CHECK(x.device() == v_metadata.device(), "x and v_metadata must share one device");
  const c10::cuda::CUDAGuard guard(x.device());
  auto quantized = tile_major
      ? torch::empty({x.size(0) / 16, n_k, 16, kTile},
                     x.options().dtype(torch::kInt8))
      : torch::empty_like(x, x.options().dtype(torch::kInt8));
  const int n_m = x.size(0) == 1 ? 1 : static_cast<int>(x.size(0) / 16);
  auto scales = n_m == 1
      ? torch::empty({n_k}, x.options().dtype(torch::kFloat32))
      : torch::empty({n_m, n_k}, x.options().dtype(torch::kFloat32));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, x.scalar_type(), "alignquant_r5_a8_quantize", [&] {
        const dim3 grid(n_k, n_m);
        const int threads = x.size(0) == 1 ? 64 : 256;
        if (tile_major) {
          r5_a8_quantize_kernel<scalar_t, true><<<grid, threads, 0, stream>>>(
              x.data_ptr<scalar_t>(), v_metadata.data_ptr<int32_t>(),
              quantized.data_ptr<int8_t>(), scales.data_ptr<float>(),
              static_cast<int>(x.size(0)), n_k);
        } else {
          r5_a8_quantize_kernel<scalar_t, false><<<grid, threads, 0, stream>>>(
              x.data_ptr<scalar_t>(), v_metadata.data_ptr<int32_t>(),
              quantized.data_ptr<int8_t>(), scales.data_ptr<float>(),
              static_cast<int>(x.size(0)), n_k);
        }
      });
  check_cuda(cudaGetLastError(), "r5_a8_quantize_kernel");
  return {quantized, scales};
}

torch::Tensor alignquant_linear_cuda(
    torch::Tensor x_int8,
    torch::Tensor activation_scales,
    torch::Tensor state_bits,
    torch::Tensor w4_payload,
    torch::Tensor w4_scales,
    torch::Tensor w8_payload,
    torch::Tensor w8_scales,
    torch::Tensor w4_row_offsets,
    torch::Tensor w8_row_offsets,
    torch::Tensor u_metadata,
    int64_t schedule,
    int64_t decode_splits,
    bool output_bf16) {
  require_cuda_contiguous(x_int8, "x_int8");
  require_cuda_contiguous(activation_scales, "activation_scales");
  require_cuda_contiguous(state_bits, "state_bits");
  require_cuda_contiguous(w4_payload, "w4_payload");
  require_cuda_contiguous(w4_scales, "w4_scales");
  require_cuda_contiguous(w8_payload, "w8_payload");
  require_cuda_contiguous(w8_scales, "w8_scales");
  require_cuda_contiguous(w4_row_offsets, "w4_row_offsets");
  require_cuda_contiguous(w8_row_offsets, "w8_row_offsets");
  require_cuda_contiguous(u_metadata, "u_metadata");
  TORCH_CHECK(x_int8.scalar_type() == torch::kInt8, "x_int8 must be int8");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(x_int8.data_ptr()) % 4 == 0,
              "x_int8 must have a four-byte-aligned data pointer");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(w8_payload.data_ptr()) % 4 == 0,
              "w8_payload must have a four-byte-aligned data pointer");
  TORCH_CHECK(activation_scales.scalar_type() == torch::kFloat32, "activation_scales must be float32");
  TORCH_CHECK(state_bits.scalar_type() == torch::kUInt8, "state_bits must be uint8");
  TORCH_CHECK(w4_payload.scalar_type() == torch::kUInt8, "w4_payload must be uint8");
  TORCH_CHECK(w4_scales.scalar_type() == torch::kFloat32, "w4_scales must be float32");
  TORCH_CHECK(w8_payload.scalar_type() == torch::kInt8, "w8_payload must be int8");
  TORCH_CHECK(w8_scales.scalar_type() == torch::kFloat32, "w8_scales must be float32");
  TORCH_CHECK(w4_row_offsets.scalar_type() == torch::kInt32, "w4_row_offsets must be int32");
  TORCH_CHECK(w8_row_offsets.scalar_type() == torch::kInt32, "w8_row_offsets must be int32");
  TORCH_CHECK(u_metadata.scalar_type() == torch::kInt32, "u_metadata must be int32");
  const bool tile_major_a = x_int8.dim() == 4;
  TORCH_CHECK(x_int8.dim() == 2 || tile_major_a,
              "x_int8 must be [M,K] or tile-major [M/16,K/64,16,64]");
  if (tile_major_a) {
    TORCH_CHECK(x_int8.size(0) >= 8 && x_int8.size(1) > 0 &&
                    x_int8.size(2) == 16 && x_int8.size(3) == kTile,
                "tile-major x_int8 must be [M/16,K/64,16,64] with M>=128");
  } else {
    TORCH_CHECK(
        x_int8.size(0) == 1 || (x_int8.size(0) > 0 && x_int8.size(0) % 16 == 0),
        "x_int8 rows must be 1 or a positive multiple of 16");
    TORCH_CHECK(x_int8.size(1) % kTile == 0, "K must be a multiple of 64");
  }
  TORCH_CHECK(w4_payload.dim() == 3 && w4_payload.size(1) == 64 && w4_payload.size(2) == 32,
              "w4_payload must be [tiles,64,32]");
  TORCH_CHECK(w8_payload.dim() == 3 && w8_payload.size(1) == 64 && w8_payload.size(2) == 64,
              "w8_payload must be [tiles,64,64]");
  TORCH_CHECK(w4_payload.size(0) == w4_scales.numel(), "W4 payload/scale count mismatch");
  TORCH_CHECK(w8_payload.size(0) == w8_scales.numel(), "W8 payload/scale count mismatch");
  TORCH_CHECK(w4_row_offsets.dim() == 1 && w8_row_offsets.dim() == 1,
              "row offsets must be rank 1");
  TORCH_CHECK(w4_row_offsets.numel() == w8_row_offsets.numel(), "row offset sizes differ");
  const int n_j = static_cast<int>(w4_row_offsets.numel() - 1);
  const int n_k = tile_major_a ? static_cast<int>(x_int8.size(1))
                               : static_cast<int>(x_int8.size(1) / kTile);
  const int n_rows = tile_major_a ? static_cast<int>(x_int8.size(0) * 16)
                                  : static_cast<int>(x_int8.size(0));
  TORCH_CHECK(n_j > 0, "N must contain at least one 64-column tile");
  const int n_m = n_rows == 1 ? 1 : n_rows / 16;
  TORCH_CHECK(u_metadata.sizes() == torch::IntArrayRef({n_j, 4, kTile}),
              "u_metadata must be [N/64,4,64]");
  TORCH_CHECK(
      activation_scales.numel() == static_cast<int64_t>(n_m) * n_k,
      "activation_scales must contain one scalar per M16x64 tile");
  TORCH_CHECK(state_bits.numel() == (static_cast<int64_t>(n_j) * n_k + 7) / 8,
              "state_bits byte count mismatch");
  TORCH_CHECK(w4_scales.numel() + w8_scales.numel() == static_cast<int64_t>(n_j) * n_k,
              "selected payload count mismatch");
  TORCH_CHECK(x_int8.device() == activation_scales.device() &&
              x_int8.device() == state_bits.device() && x_int8.device() == w4_payload.device() &&
              x_int8.device() == w4_scales.device() && x_int8.device() == w8_payload.device() &&
              x_int8.device() == w8_scales.device() && x_int8.device() == w4_row_offsets.device() &&
              x_int8.device() == w8_row_offsets.device() && x_int8.device() == u_metadata.device(),
              "all inputs must share one CUDA device");
  TORCH_CHECK(schedule >= 0 && schedule <= 3,
              "schedule must be 0 (auto), 1 (decode), 2 (prefill), or 3 (prefill_m64)");
  TORCH_CHECK(!tile_major_a || (schedule == 0 || schedule == 3) && decode_splits == 0,
              "tile-major x_int8 requires unsplit M64 prefill");
  TORCH_CHECK(schedule != 1 || n_rows == 1,
              "forced decode schedule requires exactly one row");
  TORCH_CHECK(schedule != 2 || n_rows > 1,
              "forced prefill schedule requires more than one row");
  TORCH_CHECK(schedule != 3 || n_rows > 1,
              "forced M64 prefill schedule requires more than one row");
  TORCH_CHECK(
      decode_splits == 0 || decode_splits == 1 || decode_splits == 2 ||
          decode_splits == 4 || decode_splits == 8 || decode_splits == 16,
      "split override must be 0 (automatic), 1, 2, 4, 8, or 16");
  TORCH_CHECK(
      decode_splits == 0 || n_rows == 1 || n_rows == 16,
      "split override requires M=1 decode or exactly one M16 tile");
  TORCH_CHECK(
      decode_splits == 0 || schedule == 0,
      "split override requires the automatic schedule");
  TORCH_CHECK(
      decode_splits == 0 || decode_splits <= n_k,
      "decode_splits may not exceed K/64");
  TORCH_CHECK(
      !output_bf16 || (n_rows != 1 && decode_splits <= 1 &&
                      (n_rows >= 128 || schedule == 3)),
      "BF16 output requires the unsplit M64 prefill kernel");

  const c10::cuda::CUDAGuard guard(x_int8.device());
  auto output = torch::empty(
      {n_rows, static_cast<int64_t>(n_j * 64)},
      x_int8.options().dtype(output_bf16 ? torch::kBFloat16 : torch::kFloat32));
  // Larger output rows need fewer split-K partitions to retain work per block.
  int split_k = decode_splits != 0
      ? static_cast<int>(decode_splits)
      : schedule == 0 && n_rows < 128
      ? (n_rows == 1
             ? (n_k >= 128 ? 8 : (n_j >= 128 ? 2 : 4))
             : (n_j >= 128 ? 8 : 16))
      : 1;
  while (split_k > n_k) split_k /= 2;
  const int m_tiles = n_rows == 1 ? 1 : n_rows / 16;
  while (decode_splits == 0 && split_k > 1 &&
         m_tiles * split_k > (n_j >= 128 ? 8 : 16)) {
    split_k /= 2;
  }
  if (split_k > 1) {
    auto partial = torch::empty(
        {split_k, n_rows, static_cast<int64_t>(n_j * 64)},
        x_int8.options().dtype(torch::kFloat32));
    if (n_rows == 1) {
      if (w8_scales.numel() == 0) {
        launch_decode_splitk<0>(x_int8, activation_scales, state_bits, w4_payload,
                                w4_scales, w8_payload, w8_scales, w4_row_offsets,
                                w8_row_offsets, partial, n_k,
                                split_k);
      } else if (w4_scales.numel() == 0) {
        launch_decode_splitk<1>(x_int8, activation_scales, state_bits, w4_payload,
                                w4_scales, w8_payload, w8_scales, w4_row_offsets,
                                w8_row_offsets, partial, n_k,
                                split_k);
      } else {
        launch_decode_splitk<kMixed>(x_int8, activation_scales, state_bits, w4_payload,
                                     w4_scales, w8_payload, w8_scales,
                                     w4_row_offsets, w8_row_offsets, partial, n_k,
                                     split_k);
      }
    } else {
      TORCH_CHECK(n_rows % 16 == 0,
                  "prefill rows must be M16-aligned for split-K");
      if (w8_scales.numel() == 0) {
        launch_splitk<0>(x_int8, activation_scales, state_bits, w4_payload, w4_scales,
                         w8_payload, w8_scales, w4_row_offsets, w8_row_offsets,
                         partial, n_k, split_k);
      } else if (w4_scales.numel() == 0) {
        launch_splitk<1>(x_int8, activation_scales, state_bits, w4_payload, w4_scales,
                         w8_payload, w8_scales, w4_row_offsets, w8_row_offsets,
                         partial, n_k, split_k);
      } else {
        launch_splitk<kMixed>(x_int8, activation_scales, state_bits, w4_payload, w4_scales,
                              w8_payload, w8_scales, w4_row_offsets, w8_row_offsets,
                              partial, n_k, split_k);
      }
    }
    check_cuda(cudaGetLastError(), "w4w8_a8 split-k kernel");
    const int blocks = n_j * n_m;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    w4w8_a8_splitk_reduce_kernel<<<blocks, 256, 0, stream>>>(
        partial.data_ptr<float>(), u_metadata.data_ptr<int32_t>(), output.data_ptr<float>(),
        split_k, n_j, n_rows, n_j * 64);
    check_cuda(cudaGetLastError(), "w4w8_a8 split-k reducer");
  } else if (w8_scales.numel() == 0) {
    launch<0>(x_int8, activation_scales, state_bits, w4_payload, w4_scales,
              w8_payload, w8_scales, w4_row_offsets, w8_row_offsets, u_metadata, output, n_j, n_k,
              schedule == 3, output_bf16);
  } else if (w4_scales.numel() == 0) {
    launch<1>(x_int8, activation_scales, state_bits, w4_payload, w4_scales,
              w8_payload, w8_scales, w4_row_offsets, w8_row_offsets, u_metadata, output, n_j, n_k,
              schedule == 3, output_bf16);
  } else if (n_rows >= 1024) {
    // Long prefill pays one exact-integer expansion per projection call, then
    // uses the existing native all-W8 M64 path. Only temporary execution
    // buffers are dense; artifact storage and selected quantized values stay
    // unchanged.
    const int64_t tile_count = static_cast<int64_t>(n_j) * n_k;
    auto dense_w8_payload = torch::empty(
        {tile_count, kTile, kTile}, x_int8.options().dtype(torch::kInt8));
    auto dense_w8_scales = torch::empty(
        {tile_count}, x_int8.options().dtype(torch::kFloat32));
    auto dense_w8_offsets = torch::empty(
        {static_cast<int64_t>(n_j) + 1}, x_int8.options().dtype(torch::kInt32));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    expand_mixed_payload_to_w8_kernel<<<
        static_cast<unsigned int>(tile_count), 256, 0, stream>>>(
        state_bits.data_ptr<uint8_t>(), w4_payload.data_ptr<uint8_t>(),
        w4_scales.data_ptr<float>(), w8_payload.data_ptr<int8_t>(),
        w8_scales.data_ptr<float>(), w4_row_offsets.data_ptr<int32_t>(),
        w8_row_offsets.data_ptr<int32_t>(), dense_w8_payload.data_ptr<int8_t>(),
        dense_w8_scales.data_ptr<float>(), dense_w8_offsets.data_ptr<int32_t>(),
        n_j, n_k);
    check_cuda(cudaGetLastError(), "w4w8_a8 temporary W8 expansion");
    launch<1>(x_int8, activation_scales, state_bits, w4_payload, w4_scales,
              dense_w8_payload, dense_w8_scales, w4_row_offsets,
              dense_w8_offsets, u_metadata, output, n_j, n_k,
              schedule == 3, output_bf16);
  } else {
    launch<kMixed>(x_int8, activation_scales, state_bits, w4_payload, w4_scales,
                   w8_payload, w8_scales, w4_row_offsets, w8_row_offsets, u_metadata, output, n_j, n_k,
                   schedule == 3, output_bf16);
  }
  check_cuda(cudaGetLastError(), "w4w8_a8_linear kernel");
  return output;
}
