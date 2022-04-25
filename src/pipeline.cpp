
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

#include <pixelpipes/pipeline.hpp>

using namespace std;
using namespace std::chrono;

namespace pixelpipes
{

    PIXELPIPES_REGISTER_TYPE(DNFType, "dnf", DEFAULT_TYPE_CONSTRUCTOR(DNFType), default_type_resolve);

    PipelineException::PipelineException(std::string reason, SharedPipeline pipeline, int operation) : BaseException(std::string(reason.c_str()) + std::string(" (operation ") + std::to_string(operation) + std::string(")")), pipeline(pipeline), _operation(operation) {}

    int PipelineException::operation() const throw()
    {
        return _operation;
    }

    DNF::DNF(std::vector<std::vector<bool>> clauses)
    {

        for (auto x : clauses)
        {
            if (x.size() > 0)
            {
                push_back(x);
            }
        }
    }

    class OutputList : public Token
    {
    public:
        OutputList(std::vector<SharedToken> list) : list(list)
        {
        }

        ~OutputList() = default;

        virtual size_t size() const { return list.size(); };

        virtual std::vector<SharedToken> get() const { return list; };

        virtual TypeIdentifier type_id() const { return 0; };

        virtual void describe(std::ostream &os) const
        {
            os << "[Output list]";
        }

    private:
        std::vector<SharedToken> list;
    };

    class Output : public Operation
    {
    public:
        Output(std::string name) : name(name) {}
        virtual ~Output() = default;

        virtual SharedToken run(std::vector<SharedToken> inputs)
        {

            VERIFY(inputs.size() >= 1, "At least one input required");

            return make_shared<OutputList>(inputs);
        }

        std::string get_label() const {
            return name;
        }

        virtual TypeIdentifier type()
        {
            return GetTypeIdentifier<Output>();
        }

    private:
        std::string name;
    };

    REGISTER_OPERATION("_output", Output, std::string);

    class Jump : public Operation
    {
    public:
        Jump(int offset) : offset(offset)
        {

            if (offset < 1)
                throw TypeException("Illegal jump location");
        };

        ~Jump() = default;

        virtual SharedToken run(std::vector<SharedToken> inputs)
        {

            VERIFY(inputs.size() == 1, "Incorrect number of parameters");

            return make_shared<Integer>(offset);
        }

        virtual TypeIdentifier type()
        {
            return GetTypeIdentifier<Jump>();
        };

    protected:
        int offset;

    };

    REGISTER_OPERATION("_jump", Jump, int);

    class Constant : public Operation
    {
    public:
        Constant(SharedToken value) : value(value) {}

        virtual SharedToken run(std::vector<SharedToken> inputs)
        {
            UNUSED(inputs);
            return value;
        }

        virtual TypeIdentifier type()
        {
            return GetTypeIdentifier<Constant>();
        }

    private:
        SharedToken value;
    };

    REGISTER_OPERATION("_constant", Constant, SharedToken);

    class ConditionalJump : public Jump
    {
    public:
        ConditionalJump(DNF condition, int offset) : Jump(offset), condition(condition)
        {

            VERIFY(condition.size() >= 1, "At least one condition must be defined");
        }
        ~ConditionalJump() = default;

        virtual SharedToken run(std::vector<SharedToken> inputs)
        {

            // VERIFY(inputs.size() == negate.size(), "Incorrect number of parameters");

            // Counters j and m are used to keep track of the inputs, j denotes the global position
            // while m moves within the conjunction subclause. This allows early termination of conjunction
            size_t j = 0;
            size_t m = 0;
            for (size_t i = 0; i < condition.size(); i++)
            {
                bool result = true;
                j += m;
                m = 0;
                for (bool negate : condition[i])
                {

                    result &= inputs[j + m] && ((Boolean::get_value(inputs[j + m])) != negate);
                    if (!result)
                    {
                        m = condition[i].size();
                        break;
                    }
                    else
                        m++;
                }

                if (result)
                    return make_shared<Integer>(0);
            }

            return make_shared<Integer>(offset);
        }

        virtual TypeIdentifier type()
        {
            return GetTypeIdentifier<ConditionalJump>();
        }

    private:
        DNF condition;
    };

    class Conditional : public Operation
    {
    public:
        Conditional(DNF condition) : condition(condition)
        {

            VERIFY(condition.size() >= 1, "At least one condition must be defined");
        };

        ~Conditional() = default;

        virtual SharedToken run(std::vector<SharedToken> inputs)
        {

            // VERIFY(inputs.size() == negate.size() + 2, "Incorrect number of parameters");

            size_t j = 2; // We start rading two inputs in, the first two inputs are output choices
            size_t m = 1;
            for (size_t i = 0; i < condition.size(); i++)
            {
                bool result = true;
                j += m - 1;
                m = 0;
                for (bool negate : condition[i])
                {

                    result &= inputs[j + m] && ((Boolean::get_value(inputs[j + m])) != negate);
                    m++;
                    if (!result)
                    {
                        m = condition[i].size();
                        break;
                    }
                }
                if (result)
                    return inputs[0];
            }

            return inputs[1];
        }

        virtual TypeIdentifier type()
        {
            return GetTypeIdentifier<Conditional>();
        }


    private:
        DNF condition;
    };

    class ContextQuery : public Operation
    {
    public:
        ContextQuery(ContextData query) : query(query) {}
        ~ContextQuery() = default;

        virtual SharedToken run(std::vector<SharedToken> inputs)
        {
            UNUSED(inputs);
            return empty<Integer>();
        }

        ContextData get_query()
        {
            return query;
        }

        virtual TypeIdentifier type()
        {
            return GetTypeIdentifier<ContextQuery>();
        }

    protected:
        ContextData query;

    };

    REGISTER_OPERATION("_context", ContextQuery, ContextData);

    class DebugOutput : public Operation
    {
    public:
        DebugOutput(std::string prefix) : prefix(prefix)
        {
        }
        ~DebugOutput() = default;

        virtual SharedToken run(std::vector<SharedToken> inputs)
        {

            VERIFY(inputs.size() == 1, "Only one input supported for debug");

            std::cout << prefix << *inputs[0] << std::endl;

            return inputs[0];
        }

    protected:
        std::string prefix;
    };

    REGISTER_OPERATION("_cjump", ConditionalJump, DNF, int);

    REGISTER_OPERATION("_condition", Conditional, DNF);

    REGISTER_OPERATION("_debug", DebugOutput, std::string);

    Pipeline::Pipeline() : finalized(false) {}

    void Pipeline::finalize()
    {

        finalized = true;

        cache.resize(operations.size());
        stats.resize(operations.size());

        for (size_t i = 0; i < stats.size(); i++)
        {
            stats[i].count = 0;
            stats[i].elapsed = 0;
        }
    }

    int Pipeline::append(std::string name, std::vector<SharedToken> args, std::vector<int> inputs)
    {

        if (finalized)
            throw PipelineException("Pipeline is finalized", shared_from_this(), -1);

        SharedOperation operation = create_operation(name, args);

        for (int i : inputs)
        {
            if (i >= (int)operations.size() || i < 0)
                throw PipelineException("Operation index out of bounds", shared_from_this(), -1);

            if ((operations[i].first->type()) == GetTypeIdentifier<Output>()) {
                throw PipelineException("Cannot refer to output operation", shared_from_this(), -1);
            }
        }

        operations.push_back(pair<SharedOperation, std::vector<int>>(operation, inputs));

        if ((operations.back().first->type()) == GetTypeIdentifier<Output>()) {
            auto output = std::static_pointer_cast<Output>(operations.back().first);
            for (size_t i = 0; i < inputs.size(); i++)
                labels.push_back(output->get_label());
        }

        return operations.size() - 1;
    }

    std::vector<std::string> Pipeline::get_labels()
    {

        if (!finalized)
            throw PipelineException("Pipeline not finalized", shared_from_this(), -1);

        return std::vector<std::string>(labels.begin(), labels.end());
    }

    std::vector<SharedToken> Pipeline::run(unsigned long index)
    {

        vector<SharedToken> context;
        vector<SharedToken> result;

        if (!finalized)
            throw PipelineException("Pipeline not finalized", shared_from_this(), -1);

        // auto start = high_resolution_clock::now();

        context.resize(operations.size());

        size_t i = 0;

        std::default_random_engine generator(index);

        while (i < operations.size())
        {

            if (cache[i])
            {
                context[i] = cache[i];
                if ((operations[i].first->type()) == GetTypeIdentifier<Output>())
                {
                    result.push_back(cache[i]);
                }
                i++;
                continue;
            }

            vector<SharedToken> local;
            local.reserve(operations[i].second.size());

            for (int j : operations[i].second)
            {
                local.push_back(context[j]);
            }

            try
            {
                auto operation_start = high_resolution_clock::now();

                auto output = operations[i].first->run(local);
                auto operation_end = high_resolution_clock::now();

                stats[i].count++;
                stats[i].elapsed += duration_cast<microseconds>(operation_end - operation_start).count();

                if ((operations[i].first->type()) == GetTypeIdentifier<ContextQuery>())
                {
                    switch (std::static_pointer_cast<ContextQuery>(operations[i].first)->get_query())
                    {
                    case ContextData::SampleIndex:
                    {
                        output = make_shared<Integer>(index);
                        break;
                    }
                    case ContextData::OperationIndex:
                    {
                        output = make_shared<Integer>(i);
                        break;
                    }
                    case ContextData::RandomSeed:
                    {
                        output = make_shared<Integer>(generator());
                        break;
                    }
                    default:
                        throw PipelineException("Illegal query", shared_from_this(), i);
                    }
                }

                context[i] = output;

                if (!output)
                {
                    throw OperationException("Operation output undefined", operations[i].first);
                }

                if ((operations[i].first)->type() == GetTypeIdentifier<Output>())
                {
                    for (auto x : std::static_pointer_cast<OutputList>(output)->get())
                    {
                        result.push_back(x);
                    }
                }

                if ((operations[i].first)->type() == GetTypeIdentifier<Jump>())
                {
                    size_t jump = (size_t)Integer::get_value(context[i]);

                    if ((i + jump) >= operations.size())
                        throw PipelineException("Unable to execute jump", shared_from_this(), i);

                    i += 1 + jump;
                    continue;
                }
            }
            catch (BaseException &e)
            {
                std::cout << "ERROR at operation " << i << ": " << e.what() << ", inputs:" << std::endl;
                int k = 0;
                for (int j : operations[i].second)
                {
                    std::cout << " * " << k << " (operation " << j << "): ";
                    if ((bool)(local[k]))
                    {
                        std::cout << local[k]->describe() << std::endl;
                    }
                    else
                    {
                        std::cout << "undefined" << std::endl;
                    }
                    k++;
                }
                throw PipelineException(e.what(), shared_from_this(), i);
            }

            i++;
        }

        VERIFY(result.size() == labels.size(), "Output mismatch");

        return result;
    }

    std::vector<float> Pipeline::operation_time()
    {

        std::vector<float> extracted;

        for (auto op : stats)
        {
            if (!op.count)
            {
                extracted.push_back(0);
            }
            else
            {
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