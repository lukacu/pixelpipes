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
#include "numbers.hpp"

using namespace std;

namespace pixelpipes {

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

    virtual size_t size() { return list.size(); };

    virtual std::vector<SharedVariable> get() { return list; }; 

    virtual VariableType type() { return VariableType::List; };

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

SharedVariable Copy::run(std::vector<SharedVariable> inputs, ContextHandle context) {
    if (inputs.size() != 1) 
        throw OperationException("Incorrect number of parameters", shared_from_this());

    return inputs[0];
}


Jump::Jump() {};

OperationType Jump::type() {
    return OperationType::Control;
}

SharedVariable Jump::run(std::vector<SharedVariable> inputs, ContextHandle context) {

}


SharedVariable Output::run(std::vector<SharedVariable> inputs, ContextHandle context) {

    if (inputs.size() < 1)
        throw OperationException("At least one output", shared_from_this());

    return make_shared<OutputList>(inputs);

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

        for (int j : operations[i].second) {
            local.push_back(context[j]);
        }

        try {
            auto output = operations[i].first->run(local, local_context);
            context[i] = output;
            if (operations[i].first->type() == OperationType::Output) {
                result = std::static_pointer_cast<OutputList>(output)->get();
                break;
            }

            if (operations[i].first->type() == OperationType::Control) {
                size_t jump = (size_t) Integer::get_value(context[i]);

                if (jump <= 0 || (i + jump) >= operations.size())
                    throw PipelineException("Unable to execute jump", shared_from_this(), i);

                i += jump;
                continue;
            }

        } catch (OperationException oe) {
            throw PipelineException(oe.what(), shared_from_this(), i);
        }
        catch (VariableException ve) {
            throw PipelineException(ve.what(), shared_from_this(), i);
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
        } catch (PipelineException pe) {
            callback->error(pe);
        }
    } );

}

}