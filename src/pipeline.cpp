
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
#include <pixelpipes/serialization.hpp>

using namespace std;
using namespace std::chrono;

namespace pixelpipes
{

    PIXELPIPES_REGISTER_TYPE(DNFType, "dnf", DEFAULT_TYPE_CONSTRUCTOR(DNFType), default_type_resolve);

    PIXELPIPES_REGISTER_WRITER(DNFType, [](SharedToken v, std::ostream &target)
                               { ContainerToken<DNF>::get_value(v).write(target); });

    PIXELPIPES_REGISTER_READER(DNFType, [](std::istream &source)
                               { return wrap(DNF(source)); });


    PipelineException::PipelineException(std::string reason, SharedPipeline pipeline, int operation) : BaseException(std::string(reason.c_str()) + std::string(" (operation ") + std::to_string(operation) + std::string(")")), pipeline(pipeline), _operation(operation) {}

    int PipelineException::operation() const throw()
    {
        return _operation;
    }

    DNF::DNF(Span<Span<bool>> clauses)
    {

        for (auto x : clauses)
        {
            if (x.size() > 0)
            {
                this->clauses.push_back(std::vector<bool>(x.begin(), x.end()));
            }
        }
    }

    void DNF::write(std::ostream &target) const
    {
        write_t(target, clauses.size());
        for (auto x : clauses)
        {
            write_t(target, x.size());
            for (auto d : x)
                write_t<bool>(target, d); // Explicit type needed
        }
    }

    DNF::DNF(std::istream &source)
    {
        size_t len = read_t<size_t>(source);
        clauses.reserve(len);
        for (size_t i = 0; i < len; i++)
        {
            std::vector<bool> clause;
            size_t n = read_t<size_t>(source);
            for (size_t j = 0; j < n; j++)
                clause.push_back(read_t<bool>(source));
            clauses.push_back(clause);
        }
    }

    size_t DNF::size() const
    {

        size_t count = 0;

        for (auto x : clauses)
        {
            count += x.size();
        }

        return count;
    }

    bool DNF::compute(TokenList values) const
    {

        auto v = values.begin();

        for (size_t i = 0; i < clauses.size(); i++)
        {
            auto offset = v;
            bool result = true;
            for (bool negate : clauses[i])
            {
                result &= (*v) && (Boolean::get_value(*v) != negate);
                if (!result)
                {
                    v = offset + clauses[i].size();
                    break;
                }
                else
                    v++;
            }

            if (result)
                return true;
        }

        return false;
    }


    class Output : public Operation
    {
    public:
        Output(std::string label) : label(label) {}
        virtual ~Output() = default;

        virtual SharedToken run(TokenList inputs)
        {

            VERIFY(inputs.size() == 1, "One input required");

            return empty();
        }

        std::string get_label() const
        {
            return label;
        }

        virtual TypeIdentifier type()
        {
            return GetTypeIdentifier<Output>();
        }

    private:
        std::string label;
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

        virtual SharedToken run(TokenList inputs)
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

        virtual SharedToken run(TokenList inputs)
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

        virtual SharedToken run(TokenList inputs)
        {

            if (condition.compute(inputs.slice(2, inputs.size() - 2))) {
                return make_shared<Integer>(0);
            } else {
                return make_shared<Integer>(offset);
            }

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

        virtual SharedToken run(TokenList inputs)
        {

            if (condition.compute(inputs.slice(2, inputs.size() - 2))) {
                return inputs[0];
            } else {
                return inputs[1];
            }

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

        virtual SharedToken run(TokenList inputs)
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

        virtual SharedToken run(TokenList inputs)
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

    int Pipeline::append(std::string name, TokenList args, Span<int> inputs)
    {

        if (finalized)
            throw PipelineException("Pipeline is finalized", shared_from_this(), -1);

        SharedOperation operation = create_operation(name, args);

        for (int i : inputs)
        {
            if (i >= (int)operations.size() || i < 0)
                throw PipelineException("Operation index out of bounds", shared_from_this(), -1);

            if ((operations[i].first->type()) == GetTypeIdentifier<Output>())
            {
                throw PipelineException("Cannot refer to output operation", shared_from_this(), -1);
            }
        }

        operations.push_back(pair<SharedOperation, std::vector<int>>(operation, std::vector<int>(inputs.begin(), inputs.end())));

        if ((operations.back().first->type()) == GetTypeIdentifier<Output>())
        {
            auto output = std::static_pointer_cast<Output>(operations.back().first);
            for (size_t i = 0; i < inputs.size(); i++)
                labels.push_back(output->get_label());
        }

        return (int) (operations.size() - 1);
    }

    std::vector<std::string> Pipeline::get_labels()
    {

        if (!finalized)
            throw PipelineException("Pipeline not finalized", shared_from_this(), -1);

        return std::vector<std::string>(labels.begin(), labels.end());
    }

    Sequence<SharedToken> Pipeline::run(unsigned long index)
    {

        vector<SharedToken> context;
        vector<SharedToken> result;

        if (!finalized)
            throw PipelineException("Pipeline not finalized", shared_from_this(), -1);

        // auto start = high_resolution_clock::now();

        context.resize(operations.size());

        size_t i = 0;

        RandomGenerator generator = make_generator((int)index);

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

                auto output = operations[i].first->run(make_span(local));
                auto operation_end = high_resolution_clock::now();

                stats[i].count++;
                stats[i].elapsed += (unsigned long) duration_cast<microseconds>(operation_end - operation_start).count();

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
                        output = make_shared<Integer>((int)i);
                        break;
                    }
                    case ContextData::RandomSeed:
                    {
                        output = make_shared<Integer>(generator());
                        break;
                    }
                    default:
                        throw PipelineException("Illegal query", shared_from_this(), (int) i);
                    }
                }

                context[i] = output;

                if ((operations[i].first)->type() == GetTypeIdentifier<Output>())
                {
                    for (auto x : local)
                    {
                        result.push_back(x);
                    }
                    i++;
                    continue;
                }

                if (!output)
                {
                    throw OperationException("Operation output undefined", operations[i].first);
                }

                if ((operations[i].first)->type() == GetTypeIdentifier<Jump>())
                {
                    size_t jump = (size_t)Integer::get_value(context[i]);

                    if ((i + jump) >= operations.size())
                        throw PipelineException("Unable to execute jump", shared_from_this(), (int) i);

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
                throw PipelineException(e.what(), shared_from_this(), (int) i);
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