#include <torch/extension.h>

#include <vector>


std::vector<torch::Tensor> alignquant_activation_quantize_cuda(
    torch::Tensor activation,
    torch::Tensor v_metadata,
    bool tile_major);

torch::Tensor alignquant_linear_cuda(
    torch::Tensor activation,
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
    bool output_bf16);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def(
      "alignquant_activation_quantize_cuda",
      &alignquant_activation_quantize_cuda,
      "Frozen R5 transform and K64 absmax A8 quantization",
      pybind11::arg("activation"),
      pybind11::arg("v_metadata"),
      pybind11::arg("tile_major") = false);
  module.def(
      "alignquant_linear_cuda",
      &alignquant_linear_cuda,
      "Static packed W4A8/W8A8 projection",
      pybind11::arg("activation"),
      pybind11::arg("activation_scales"),
      pybind11::arg("state_bits"),
      pybind11::arg("w4_payload"),
      pybind11::arg("w4_scales"),
      pybind11::arg("w8_payload"),
      pybind11::arg("w8_scales"),
      pybind11::arg("w4_row_offsets"),
      pybind11::arg("w8_row_offsets"),
      pybind11::arg("u_metadata"),
      pybind11::arg("schedule") = 0,
      pybind11::arg("decode_splits") = 0,
      pybind11::arg("output_bf16") = false);
}
