
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
#include <filesystem>

#include <pixelpipes/pipeline.hpp>
#include <pixelpipes/buffer.hpp>
#include <pixelpipes/serialization.hpp>
#include <pixelpipes/operation.hpp>

#include "pipeline/optimization.hpp"

using namespace std;
using namespace std::chrono;

namespace pixelpipes
{

    template <>
    std::filesystem::path extract(const TokenReference &t)
    {
        return std::filesystem::path(extract<std::string>(t));
    }

    template <>
    TokenReference wrap(std::filesystem::path value)
    {
        return wrap(value.string());
    }

    PipelineException::PipelineException(std::string reason) : BaseException(std::string(reason.c_str())) {}

    OperationException::OperationException(std::string reason, const OperationReference &reference, int position) : PipelineException(std::string(reason.c_str()) + std::string(" (operation ") + operation_name(reference) + ", " + std::to_string(position) + std::string(")")), _position(position) {}

    class Output : public Operation
    {
    public:
        Output(std::string label) : label(label) {}
        virtual ~Output() = default;

        virtual TokenReference run(const TokenList &inputs)
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

        virtual Sequence<TokenReference> serialize() { return Sequence<TokenReference>({wrap(label)}); }

    private:
        std::string label;
    };

    bool is_output(OperationReference &op)
    {
        return (bool)op && op->is<Output>();
    }

    PIXELPIPES_OPERATION_CLASS("output", Output, std::string);

    class Constant : public Operation
    {
    public:
        Constant(const TokenReference &value) : value(value.reborrow())
        {
            VERIFY((bool)value, "Constant value undefined");
        }

        virtual TokenReference run(const TokenList &inputs)
        {
            UNUSED(inputs);
            return value.reborrow();
        }

        virtual TypeIdentifier type()
        {
            return GetTypeIdentifier<Constant>();
        }

        virtual Sequence<TokenReference> serialize() { return Sequence<TokenReference>({value.reborrow()}); }

    private:
        TokenReference value;
    };

    PIXELPIPES_OPERATION_CLASS("constant", Constant, TokenReference);
    class Conditional : public Operation
    {
    public:
        Conditional() = default;
        ~Conditional() = default;

        virtual TokenReference run(const TokenList &inputs)
        {
            VERIFY(inputs.size() == 3, "Illegal number of inputs");

            if (extract<bool>(inputs[2]))
                return inputs[0].reborrow();
            else
                return inputs[1].reborrow();
        }

        virtual TypeIdentifier type()
        {
            return GetTypeIdentifier<Conditional>();
        }

        virtual Sequence<TokenReference> serialize() { return Sequence<TokenReference>(); }
    };

    bool is_conditional(OperationReference &op)
    {
        return (bool)op && op->is<Conditional>();
    }

    bool is_constant(OperationReference &op)
    {
        return (bool)op && op->is<Constant>();
    }

    class ContextQuery : public Operation
    {
    public:
        ContextQuery(ContextData query) : query(query) {}
        ~ContextQuery() = default;

        virtual TokenReference run(const TokenList &inputs)
        {
            UNUSED(inputs);
            return empty<IntegerScalar>();
        }

        ContextData get_query()
        {
            return query;
        }

        virtual TypeIdentifier type()
        {
            return GetTypeIdentifier<ContextQuery>();
        }

        virtual Sequence<TokenReference> serialize() { return Sequence<TokenReference>({wrap(query)}); }

    protected:
        ContextData query;
    };

    bool is_context(OperationReference &op)
    {
        return (bool)op && op->is<ContextQuery>();
    }

    class DebugOutput : public Operation
    {
    public:
        DebugOutput(std::string prefix) : prefix(prefix)
        {
        }
        ~DebugOutput() = default;

        virtual TokenReference run(const TokenList &inputs)
        {

            VERIFY(inputs.size() == 1, "Only one input supported for debug");

            std::cout << prefix << inputs[0]->describe() << std::endl;

            return inputs[0].reborrow();
        }

        virtual Sequence<TokenReference> serialize() { return Sequence<TokenReference>({wrap(prefix)}); }

    protected:
        std::string prefix;
    };

    PIXELPIPES_OPERATION_CLASS("context", ContextQuery, ContextData);

    PIXELPIPES_OPERATION_CLASS("condition", Conditional);

    PIXELPIPES_OPERATION_CLASS("debug", DebugOutput, std::string);

    struct Pipeline::State
    {
        bool finalized;

        std::vector<TokenReference> cache;

        std::vector<OperationData> operations;

        std::vector<OperationStats> stats;

        std::vector<std::string> labels;
    };

    void Pipeline::StateDeleter::operator()(State *p) const
    {
        // std::cout << "Delete " << std::endl;
        delete p;
    }

    Pipeline::Pipeline() : state(new Pipeline::State())
    {
    }

    // Pipeline::Pipeline(const Pipeline &) = default;
    Pipeline::Pipeline(Pipeline &&other) = default;

    // Pipeline& Pipeline::operator=(const Pipeline &) = default;
    // Pipeline& Pipeline::operator=(Pipeline &&) = default;

    Pipeline &Pipeline::operator=(Pipeline &&other) = default;

    Pipeline::~Pipeline() = default;

    void Pipeline::finalize(bool optimize)
    {

        state->finalized = true;

        state->operations = optimize_pipeline(state->operations, optimize);

        state->cache.resize(state->operations.size());
        state->stats.resize(state->operations.size());

        for (size_t i = 0; i < state->stats.size(); i++)
        {
            state->stats[i].count = 0;
            state->stats[i].elapsed = 0;
        }
    }

    int Pipeline::append(std::string name, const TokenList &args, const Span<int> &inputs)
    {

        if (state->finalized)
            throw PipelineException("Pipeline is finalized");

        OperationReference operation = create_operation(name, args);

        for (int i : inputs)
        {
            if (i >= (int)state->operations.size() || i < 0)
                throw OperationException("Operation index out of bounds", operation, (int)state->operations.size());

            if ((state->operations[i].first->type()) == GetTypeIdentifier<Output>())
            {
                throw OperationException("Cannot refer to output operation", operation, (int)state->operations.size());
            }
        }

        state->operations.push_back(pair<OperationReference, std::vector<int>>(std::move(operation), std::vector<int>(inputs.begin(), inputs.end())));

        if ((state->operations.back().first->type()) == GetTypeIdentifier<Output>())
        {
            auto label = (state->operations.back().first.get_as<Output>())->get_label();
            for (size_t i = 0; i < inputs.size(); i++)
                state->labels.push_back(label);
        }

        return (int)(state->operations.size() - 1);
    }

    Sequence<std::string> Pipeline::get_labels() const
    {

        if (!state->finalized)
            throw PipelineException("Pipeline not finalized");

        return Sequence<std::string>(state->labels);
    }

    size_t Pipeline::size() const
    {
        return state->operations.size();
    }

    Pipeline::OperationData Pipeline::get(size_t i) const
    {
        return {state->operations[i].first.reborrow(), state->operations[i].second};
    }

    Sequence<TokenReference> Pipeline::run(unsigned long index)
    {

        vector<TokenReference> context;
        vector<TokenReference> result;

        if (!state->finalized)
            throw PipelineException("Pipeline not finalized");

        // auto start = high_resolution_clock::now();

        context.resize(state->operations.size());

        size_t i = 0;

        RandomGenerator generator = make_generator((int)index);

        while (i < state->operations.size())
        {

            if (state->cache[i])
            {
                context[i] = state->cache[i].reborrow();
                if ((state->operations[i].first->type()) == GetTypeIdentifier<Output>())
                {
                    result.push_back(state->cache[i].reborrow());
                }
                i++;
                continue;
            }

            vector<TokenReference> local;
            local.reserve(state->operations[i].second.size());

            for (int j : state->operations[i].second)
            {
                local.push_back(context[j].reborrow());
            }

            try
            {
                auto operation_start = high_resolution_clock::now();

                auto output = state->operations[i].first->run(make_span(local));
                auto operation_end = high_resolution_clock::now();

                if ((state->operations[i].first->type()) == GetTypeIdentifier<ContextQuery>())
                {
                    switch ((state->operations[i].first.get_as<ContextQuery>())->get_query())
                    {
                    case ContextData::SampleIndex:
                    {
                        output = create<IntegerScalar>(index);
                        break;
                    }
                    case ContextData::OperationIndex:
                    {
                        output = create<IntegerScalar>((int)i);
                        break;
                    }
                    case ContextData::RandomSeed:
                    {
                        output = create<IntegerScalar>(generator());
                        break;
                    }
                    default:
                        throw OperationException("Illegal query", state->operations[i].first, (int)i);
                    }
                }

                state->stats[i].count++;
                state->stats[i].elapsed += (unsigned long)duration_cast<microseconds>(operation_end - operation_start).count();

                if ((state->operations[i].first)->type() == GetTypeIdentifier<Output>())
                {
                    for (auto x = local.begin(); x != local.end(); x++)
                    {
                        result.push_back(std::move(*x));
                    }
                    i++;
                    continue;
                }

                if (!output)
                {
                    throw OperationException("Operation output undefined", state->operations[i].first, (int)i);
                }

                context[i] = std::move(output);

                if ((state->operations[i].first)->type() == GetTypeIdentifier<Jump>())
                {
                    size_t jump = (size_t)extract<int>(context[i]);

                    if ((i + jump) >= state->operations.size())
                        throw PipelineException("Unable to execute jump");

                    i += 1 + jump;
                    continue;
                }
            }
            catch (BaseException &e)
            {
                std::cout << "ERROR at operation " << i << ": " << e.what() << ", inputs:" << std::endl;
                int k = 0;
                for (int j : state->operations[i].second)
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
                throw OperationException(e.what(), state->operations[i].first, (int)i);
            }

            i++;
        }

        VERIFY(result.size() == state->labels.size(), "Output mismatch");

        auto r = Sequence<TokenReference>(result);

        return r;
    }

    std::string PIXELPIPES_API visualize_pipeline(const Pipeline &pipeline)
    {
        size_t n = 0;

        std::stringstream stream;

        stream << "digraph graphname {" << std::endl;

        std::vector<size_t> outputs;

        for (size_t o = 0; o < pipeline.size(); o++)
        {
            Pipeline::OperationData operation = pipeline.get(o);
            auto name = operation_name(operation.first);

            if (operation.first->is<ConditionalJump>())
            {
                name = "cjump";
            }

            stream << "OP" << n << " [ordering=in label=\"" << n << " - " << name << "\" ";
            if (operation.first->is<Output>())
            {
                stream << " shape=box";
                outputs.push_back(n);
            }
            else if (operation.first->is<ContextQuery>())
            {
                stream << " shape=hexagon";
            }

            stream << "];" << std::endl;

            for (auto i : operation.second)
            {
                stream << "OP" << i << " -> "
                       << "OP" << n << ";" << std::endl;
            }

            n++;
        }

        stream << "{ rank=same; ";

        for (auto o : outputs)
        {
            stream << " OP" << o;
        }

        stream << "}}" << std::endl;

        return stream.str();
    }

    std::vector<float> Pipeline::operation_time()
    {

        std::vector<float> extracted;

        for (auto op : state->stats)
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

    PIXELPIPES_REGISTER_ENUM("context", ContextData);
}