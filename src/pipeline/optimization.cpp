

#include <pixelpipes/pipeline.hpp>

namespace pixelpipes {

std::vector<OperationData> predictive_optimization(std::vector<OperationData> original) {

    std::vector<size_t> outputs;

    for (size_t i = 0; i < original.size(); i++) {
        if (original[i].first->is<Output>()) {
            outputs.push_back(i);
        }
    }




}



}

