#include <torch/torch.h>

using torch::Tensor;

void test_prepare_image_sort() {
  Tensor tiledepth = torch::rand({10, 10}).cuda();
}

int main() {
  test_prepare_image_sort();
  return 0;
}