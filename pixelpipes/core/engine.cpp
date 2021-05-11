
#include <chrono>
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

#include <pixelpipes/engine.hpp>
 
using namespace std;
using namespace std::chrono;

namespace pixelpipes {

PipelineException::PipelineException(std::string reason, SharedPipeline pipeline, int operation):
 BaseException(std::string(reason.c_str()) + std::string(" (operation ") + std::to_string(operation) + std::string(")")), pipeline(pipeline), _operation(operation) {}

int PipelineException::operation () const throw () {
    return _operation;
}

EngineException::EngineException(std::string reason): BaseException(reason) {}

class OutputList: public Variable {
public:

    OutputList(std::vector<SharedVariable> list) : list(list) {

    }

    ~OutputList() = default;

    virtual size_t size() const { return list.size(); };

    virtual bool is_scalar() const { return true; }

    virtual std::vector<SharedVariable> get() const { return list; }; 

    virtual TypeIdentifier type() const { return ListType; };

    virtual void describe(std::ostream& os) const {
        os << "[Output list]";
    }

private:

    std::vector<SharedVariable> list;

};

REGISTER_OPERATION("_output", Output);

Output::Output() {};
Output::~Output() {};

OperationType Output::type() {
    return OperationType::Output;
}

Constant::Constant(SharedVariable value): value(value) {}

REGISTER_OPERATION("_constant", Constant, SharedVariable);

SharedVariable Constant::run(std::vector<SharedVariable> inputs, ContextHandle context) {
    return value;
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

REGISTER_OPERATION("_jump", Jump, int);

ConditionalJump::ConditionalJump(DNF condition, int offset): Jump(offset), condition(condition) {

    VERIFY(condition.clauses.size() >= 1, "At least one condition must be defined");

}    

SharedVariable ConditionalJump::run(std::vector<SharedVariable> inputs, ContextHandle context) {



    //VERIFY(inputs.size() == negate.size(), "Incorrect number of parameters");

    // Counters j and m are used to keep track of the inputs, j denotes the global position
    // while m moves within the conjunction subclause. This allows early termination of conjunction
    size_t j = 0;
    size_t m = 0;
    for (size_t i = 0; i < condition.clauses.size(); i++) {
        bool result = true;
        j += m;
        m = 0;
        for (bool negate : condition.clauses[i]) {

            result &= inputs[j + m] && ((Boolean::get_value(inputs[j + m])) != negate);
            if (!result) {
                m = condition.clauses[i].size();
                break;
            } else m++;
        }

        if (result) return make_shared<Integer>(0);
    }

    return make_shared<Integer>(offset);
} 

REGISTER_OPERATION("_cjump", ConditionalJump, DNF, int);

Conditional::Conditional(DNF condition): condition(condition) {

    VERIFY(condition.clauses.size() >= 1, "At least one condition must be defined");
    
};
 
REGISTER_OPERATION("_condition", Conditional, DNF);

SharedVariable Conditional::run(std::vector<SharedVariable> inputs, ContextHandle context) {

    //VERIFY(inputs.size() == negate.size() + 2, "Incorrect number of parameters");

    size_t j = 2; // We start rading two inputs in, the first two inputs are output choices
    size_t m = 1;
    for (size_t i = 0; i < condition.clauses.size(); i++) {
        bool result = true;
        j += m - 1;
        m = 0;
        for (bool negate : condition.clauses[i]) {

            result &= inputs[j + m] && ((Boolean::get_value(inputs[j + m])) != negate);
            m++;
            if (!result) {
                m = condition.clauses[i].size();
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

REGISTER_OPERATION("_context", ContextQuery, ContextData);


DebugOutput::DebugOutput(std::string prefix): prefix(prefix) {}

SharedVariable DebugOutput::run(std::vector<SharedVariable> inputs, ContextHandle context) {

    VERIFY(inputs.size() == 1, "Only one input supported for debug");

    std::cout << prefix << *inputs[0] << std::endl;

    return inputs[0];

}

REGISTER_OPERATION("_debug", DebugOutput, std::string);


Pipeline::Pipeline() : finalized(false) {};

void Pipeline::finalize() {

    finalized = true;

    cache.resize(operations.size());
    stats.resize(operations.size());

    for (size_t i = 0; i < stats.size(); i++) {
        stats[i].count = 0;
        stats[i].elapsed = 0;
    }

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

    //auto start = high_resolution_clock::now();

    context.resize(operations.size());

    size_t i = 0;

    ContextHandle local_context = std::make_unique<Context>(index);

    while(i < operations.size()) {

        if (cache[i]) {
            context[i] = cache[i];
            if (getType(operations[i].first) == OperationType::Output) {
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
            auto operation_start = high_resolution_clock::now();

            auto output = operations[i].first->run(local, local_context);
            context[i] = output;
            auto operation_end = high_resolution_clock::now();

            stats[i].count++;
            stats[i].elapsed += duration_cast<microseconds>(operation_end - operation_start).count();


            if (!output) {
                throw OperationException("Operation output undefined", operations[i].first);
            }

            if (getType(operations[i].first) == OperationType::Output) {
                result = std::static_pointer_cast<OutputList>(output)->get();
                break;
            } 

            if (getType(operations[i].first) == OperationType::Control) {
                size_t jump = (size_t) Integer::get_value(context[i]);

                if (jump < 0 || (i + jump) >= operations.size())
                    throw PipelineException("Unable to execute jump", shared_from_this(), i);

                i += 1 + jump;
                continue;
            }

        } catch (BaseException &e) {
            std::cout << "ERROR at operation " << i << ": " << e.what() << ", inputs:" << std::endl;
            int k = 0;
            for (int j : operations[i].second) {
                std::cout << " * " << k << " (operation " << j << "): ";
                if ((bool)(local[k])) {
                    std::cout << local[k]->describe() << std::endl;
                } else {
                    std::cout << "undefined" << std::endl;
                }
                k++;
            }
            throw PipelineException(e.what(), shared_from_this(), i);
        }

        i++;
    }

    return result;

} 

std::vector<float> Pipeline::operation_time() {

    std::vector<float> extracted;

    for (auto op : stats) {
        if (!op.count) {
            extracted.push_back(0);
        } else {
            extracted.push_back(((float)op.elapsed / (float)op.count) / 1000);
        }
    }

    return extracted;
}

/*
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
*/
}