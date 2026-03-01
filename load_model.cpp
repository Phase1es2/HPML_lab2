#include <iostream>
#include <torch/script.h>
#include <memory>

int main(int argc, const char* argv[]) {
    if (argc != 2) {
        std::cerr << "usage: my_torch_app <path-to-exported-script-module>\n";
        return -1;
    }

    torch::jit::script::Module module;
    try {
        module = torch::jit::load(argv[1], torch::kCPU);
    } catch (const c10::Error& e) {
        std::cerr << "Error loading the model\n" << e.what() << "\n";
        return -1;
    }
    std::cout << "Model Loaded Successfully\n";

    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(torch::ones({1, 3, 32, 32}));

    at::Tensor output = module.forward(inputs).toTensor();
    std::cout << output.slice(1, 0, 5) << "\n";

    return 0;
}