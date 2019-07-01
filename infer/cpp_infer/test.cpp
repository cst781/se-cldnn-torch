/*************************************************************************
	> File Name: test.cpp
	> Author: 
	> Mail: 
	> Created Time: 2019年06月23日 星期日 14时27分42秒
 ************************************************************************/

#include <torch/script.h> // One-stop header.
#include <vector>
#include <iostream>
#include <memory>
#include <ATen/ATen.h>
using namespace std;
int main(int argc, const char* argv[]) {
    if (argc != 2) {
              std::cerr << "usage: example-app <path-to-exported-script-module>\n";
              return -1;
    }

      //Deserialize the ScriptModule from a file using torch::jit::load().

    std::shared_ptr<torch::jit::script::Module> module = torch::jit::load(argv[1]);
    assert(module != nullptr);
    std::cout << "ok\n";
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(torch::ones({1,100,257*3}));
    at::Tensor output = module->forward(inputs).toTensor();
    //std::cout<< output.slice(1,0,5)<<'\n';
    for(int i=0; i < 100; i++)
        std::cout<< output[0][i]<<'\n';
}
