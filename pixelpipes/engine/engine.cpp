#include <thread>
#include <functional>
#include <vector>
#include <cstdint>
#include <cstdio>
#include <queue>
#include <iostream>
#include <mutex>
#include <string>
#include <condition_variable>


#include "engine.hpp"
#include "types.hpp"

using namespace std;

namespace pixelpipes {


PipelineException::PipelineException(std::string reason, SharedPipeline pipeline, int operation):
 BaseException(std::string(reason.c_str()) + std::string(" (operation ") + std::to_string(operation) + std::string(")")), pipeline(pipeline), _operation(operation) {}

int PipelineException::operation () const throw () {
    return _operation;
}

EngineException::EngineException(std::string reason): BaseException(reason) {}

Context::Context(unsigned long index) : index(index), generator(index) {

}

unsigned int Context::random() {
    return generator();
}

unsigned long Context::sample() {
    return index;
}

class OutputList: public Variable {
public:

    OutputList(std::vector<SharedVariable> list) : list(list) {

    }

    ~OutputList() = default;

    virtual size_t size() const { return list.size(); };

    virtual std::vector<SharedVariable> get() const { return list; }; 

    virtual VariableType type() const { return VariableType::List; };

    virtual void print(std::ostream& os) const {
        os << "[Output list]";
    }

private:

    std::vector<SharedVariable> list;

};

Operation::Operation() {};
Operation::~Operation() {};

OperationType Operation::type() {
    return OperationType::Deterministic;
}


Output::Output() {};
Output::~Output() {};

OperationType Output::type() {
    return OperationType::Output;
}

Constant::Constant(SharedVariable value): value(value) {}

SharedVariable Constant::run(std::vector<SharedVariable> inputs, ContextHandle context) {
    return value;
}

SharedVariable Copy::run(std::vector<SharedVariable> inputs, ContextHandle context) {
    if (inputs.size() != 1) 
        throw OperationException("Incorrect number of parameters", shared_from_this());

    return inputs[0];
}

Jump::Jump(int offset): offset(offset) {

    if (offset < 1)
        throw VariableException("Illegal jump location");

};

OperationType Jump::type() {
    return OperationType::Control;
}

SharedVariable Jump::run(std::vector<SharedVariable> inputs, ContextHandle context) {

    VERIFY(inputs.size() == 1, "Incorrect number of parameters");

    return make_shared<Integer>(offset);

}

ConditionalJump::ConditionalJump(DNF condition, int offset): Jump(offset), condition(condition) {

    VERIFY(condition.size() >= 1, "At least one condition must be defined");

 }

SharedVariable ConditionalJump::run(std::vector<SharedVariable> inputs, ContextHandle context) {

    //VERIFY(inputs.size() == negate.size(), "Incorrect number of parameters");
/*
    DEBUG("----");

for (auto input : inputs) {
            if (input) {
                DEBUG(Integer::get_value(input));
            } else {
                DEBUG("uninit");
            }
}

    DEBUG("xx");
*/
    // Counters j and m are used to keep track of the inputs, j denotes the global position
    // while m moves within the conjunction subclause. This allows early termination of conjunction
    size_t j = 0;
    size_t m = 0;
    for (size_t i = 0; i < condition.size(); i++) {
        bool result = true;
        j += m;
        m = 0;
        for (bool negate : condition[i]) {
/*
            if (inputs[j + m]) {
                DEBUG(Integer::get_value(inputs[j + m]));
                DEBUG(negate);
            } else {
                DEBUG("uninit");
            }
*/

            result &= inputs[j + m] && ((Integer::get_value(inputs[j + m]) != 0) != negate);
            if (!result) {
                m = condition[i].size();
                break;
            } else m++;
        }
        //std::cout << "result " << i << " " << j << " " << m << " " << result << std::endl; 

        if (result) return make_shared<Integer>(0);
    }

    return make_shared<Integer>(offset);
}

Conditional::Conditional(DNF condition): condition(condition) {

    VERIFY(condition.size() >= 1, "At least one condition must be defined");
    
};

SharedVariable Conditional::run(std::vector<SharedVariable> inputs, ContextHandle context) {

    //VERIFY(inputs.size() == negate.size() + 2, "Incorrect number of parameters");

    size_t j = 2; // We start rading two inputs in, the first two inputs are output choices
    size_t m = 1;
    for (size_t i = 0; i < condition.size(); i++) {
        bool result = true;
        j += m - 1;
        m = 0;
        for (bool negate : condition[i]) {

            result &= inputs[j + m] && ((Integer::get_value(inputs[j + m]) != 0) != negate);
            m++;
            if (!result) {
                m = condition[i].size();
                break;
            }
        }
        if (result) return inputs[0];
    }

    return inputs[1];

}
 
SharedVariable Output::run(std::vector<SharedVariable> inputs, ContextHandle context) {

    VERIFY(inputs.size() >= 1, "At least one input required");

    return make_shared<OutputList>(inputs);

}

ContextQuery::ContextQuery(ContextData query): query(query) {}

SharedVariable ContextQuery::run(std::vector<SharedVariable> inputs, ContextHandle context) {

    VERIFY(inputs.size() == 0, "No inputs expected");

    switch (query)
    {
    case ContextData::SampleIndex:
        return make_shared<Integer>(context->sample());
    default:
        throw OperationException("Illegal query", shared_from_this());
    }

}


DebugOutput::DebugOutput(std::string prefix): prefix(prefix) {}

SharedVariable DebugOutput::run(std::vector<SharedVariable> inputs, ContextHandle context) {

    VERIFY(inputs.size() == 1, "Only one input supported for debug");

    std::cout << prefix << *inputs[0] << std::endl;

    return inputs[0];

}

Pipeline::Pipeline() : finalized(false) {};

Pipeline::~Pipeline() {};

void Pipeline::finalize() {

    finalized = true;

    cache.resize(operations.size());

}

int Pipeline::append(SharedOperation operation, std::vector<int> inputs) {

    if (finalized) return -1;

    if (!operation) return -1;

    for (int i : inputs) {
        if (i >= (int) operations.size() || i < 0)
            return -1;
    }

    operations.push_back(pair<SharedOperation, std::vector<int> >(operation, inputs));

    return operations.size() - 1;
}

std::vector<SharedVariable> Pipeline::run(unsigned long index) {

    vector<SharedVariable> context;
    vector<SharedVariable> result;

    context.resize(operations.size());

    size_t i = 0;

    ContextHandle local_context = std::make_unique<Context>(index);

    while(i < operations.size()) {

        if (cache[i]) {
            context[i] = cache[i];
            if (operations[i].first->type() == OperationType::Output) {
                result.push_back(cache[i]);
            }
            i++;
            continue;
        }

        vector<SharedVariable> local;
        local.reserve(operations[i].second.size());

        for (int j : operations[i].second) {
            local.push_back(context[j]);
        }

        try {
            auto output = operations[i].first->run(local, local_context);
            context[i] = output;

            if (!output) {
                throw OperationException("Operation output undefined", operations[i].first);
            }

            if (operations[i].first->type() == OperationType::Output) {
                result = std::static_pointer_cast<OutputList>(output)->get();
                break;
            } 

            if (operations[i].first->type() == OperationType::Control) {
                size_t jump = (size_t) Integer::get_value(context[i]);

                if (jump < 0 || (i + jump) >= operations.size())
                    throw PipelineException("Unable to execute jump", shared_from_this(), i);

                i += 1 + jump;
                continue;
            }

        } catch (BaseException &e) {
#ifdef DEBUG_MODE
            std::cout << "ERROR at operation " << i << ": " << e.what() << ", inputs:" << std::endl;
            int k = 0;
            for (int j : operations[i].second) {
                std::cout << " * " << k << " (operation " << j << "): " << ((local[k]) ? "OK" : "undefined") << std::endl;
                k++;
            }
#endif
            throw PipelineException(e.what(), shared_from_this(), i);
        }

        i++;
    }

    return result;

}

Engine::Engine(int workers) : workers_count(workers) {

}

Engine::~Engine() {
    stop();
}

void Engine::start() {

    if (running())
        return;

    workers = make_shared<dispatch_queue>(workers_count);

}

void Engine::stop() {

    if (!running())
        return;

    workers.reset();

}

bool Engine::running() {

    return (bool) workers;

}

bool Engine::add(std::string name, SharedPipeline pipeline) {

    std::lock_guard<std::recursive_mutex> lock (mutex_);

    pipeline->finalize();

    pipelines[name] = pipeline;

    return true;

}

bool Engine::remove(std::string name) {

    std::lock_guard<std::recursive_mutex> lock (mutex_);

    if (pipelines.find(name) == pipelines.end()) {
        return false;
    }

    pipelines.erase(pipelines.find(name));

    return true;
}

void Engine::run(std::string name, unsigned long index, std::shared_ptr<PipelineCallback> callback) {

    std::lock_guard<std::recursive_mutex> lock (mutex_);

    if (!running())
        throw EngineException("Not running");

    if (pipelines.find(name) == pipelines.end()) {
        throw EngineException("Pipeline not found");
    }

    auto pipeline = pipelines[name];

    workers->dispatch([pipeline, index, callback] () {
        try {
            auto result = pipeline->run(index); 
            callback->done(result);
        } catch (PipelineException &pe) {
            callback->error(pe);
        }
    } );

}

}