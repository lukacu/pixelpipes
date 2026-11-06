
#include <chrono>
#include <thread>
#include <functional>
#include <vector>
#include <cstdint>
#include <cstdio>
#include <queue>
#include <iostream>
#include <mutex>
#include <atomic>
#include <string>
#include <condition_variable>
#include <filesystem>
#include <map>
#include <memory>
#include <list>

#include <pixelpipes/pipeline.hpp>
#include <pixelpipes/buffer.hpp>
#include <pixelpipes/serialization.hpp>
#include <pixelpipes/operation.hpp>
#include <pixelpipes/tensor.hpp>

#include "optimization.hpp"
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

        virtual TokenReference evaluate(const TokenList &inputs)
        {
            VERIFY(inputs.size() == 1, "One input required");
            return inputs[0].reborrow();
        }

        std::string get_label() const
        {
            return label;
        }

        virtual OperationTrait trait() const override
        {
            return OperationTrait::Default;
        }

        virtual Type type() const override
        {
            return GetType<Output>();
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
        Constant(const TokenReference &value) : _value(value.reborrow())
        {
            VERIFY((bool)value, "Constant value undefined");
        }

        virtual TokenReference run(const TokenList &inputs)
        {
            UNUSED(inputs);

            VERIFY(inputs.size() == 0, "Illegal number of inputs");

            // Constants are shared
            std::scoped_lock lock(_mutex);

            return _value.reborrow();
        }

        virtual TokenReference evaluate(const TokenList &inputs)
        {
            return run(inputs);
        }

        virtual OperationTrait trait() const override
        {
            return OperationTrait::Stateful;
        }

        virtual Type type() const override
        {
            return GetType<Constant>();
        }

        virtual Sequence<TokenReference> serialize() { return Sequence<TokenReference>({_value.reborrow()}); }

    private:
        TokenReference _value;

        std::mutex _mutex;
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

        virtual TokenReference evaluate(const TokenList &inputs)
        {
            VERIFY(inputs.size() == 3, "Illegal number of inputs");

            // If all inputs are defined, evaluate the condition
            if (any_placeholder(inputs))
            {
                return create<Placeholder>(inputs[0]->shape() & inputs[1]->shape());
            }
            else
            {
                return run(inputs);
            }
        }

        virtual OperationTrait trait() const override
        {
            return OperationTrait::Default;
        }

        virtual Type type() const override
        {
            return GetType<Conditional>();
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

        virtual TokenReference evaluate(const TokenList &inputs)
        {
            UNUSED(inputs);
            return create<Placeholder>(ScalarType<int>());
        }

        virtual OperationTrait trait() const override
        {
            return OperationTrait::Stateful;
        }

        ContextData get_query()
        {
            return query;
        }

        virtual Type type() const override
        {
            return GetType<ContextQuery>();
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

            if (prefix.size() > 0)
                std::cout << prefix << ": " << inputs[0]->describe() << std::endl;
            else
                std::cout << inputs[0]->describe() << std::endl;

            return inputs[0].reborrow();
        }

        virtual TokenReference evaluate(const TokenList &inputs)
        {
            // TODO: make evaluation output toggleable via some sort of context flag
            return inputs[0].reborrow();
        }

        virtual OperationTrait trait() const override
        {
            return OperationTrait::Default;
        }

        virtual Type type() const override
        {
            return GetType<DebugOutput>();
        }

        virtual Sequence<TokenReference> serialize() { return Sequence<TokenReference>({wrap(prefix)}); }

    protected:
        std::string prefix;
    };

    PIXELPIPES_OPERATION_CLASS("context", ContextQuery, ContextData);

    PIXELPIPES_OPERATION_CLASS("condition", Conditional);

    PIXELPIPES_OPERATION_CLASS("debug", DebugOutput, std::string);

    struct Metadata::State
    {
        std::map<std::string, std::string> data;
    };

    Metadata::Metadata() = default;
    Metadata::Metadata(const Metadata &) = default;
    Metadata::~Metadata() = default;

    Metadata::Metadata(Metadata &&) = default;

    Metadata &Metadata::operator=(const Metadata &) = default;
    Metadata &Metadata::operator=(Metadata &&) = default;

    std::string Metadata::get(std::string key) const
    {
        auto i = _state->data.find(key);
        if (i == _state->data.end())
            return "";
        return i->second;
    }

    bool Metadata::has(std::string key) const
    {
        auto i = _state->data.find(key);
        return (i != _state->data.end());
    }

    void Metadata::set(std::string key, std::string value)
    {
        _state->data[key] = value;
    }

    size_t Metadata::size() const
    {
        return _state->data.size();
    }

    Sequence<std::string> Metadata::keys() const
    {
        Sequence<std::string> list(_state->data.size());

        size_t i = 0;
        for (auto item : _state->data)
        {
            list[i++] = item.first;
        }

        return list;
    }

    struct Pipeline::State
    {
        std::atomic<bool> finalized{false};

        std::vector<TokenReference> cache;
        std::vector<std::unique_ptr<std::once_flag>> cache_initializers;
        std::vector<std::unique_ptr<std::mutex>> operation_mutexes;

        std::vector<OperationData> operations;

        std::vector<bool> constants;

        std::vector<OperationStats> stats;
        std::mutex stats_mutex;

        std::vector<OutputDescription> outputs;

        Metadata metadata;
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
        if (state->finalized.load(std::memory_order_acquire))
            throw PipelineException("Pipeline is finalized");

        // First determine stateless outputs
        // prune if needed (replace with constants)
        // optimize if needed (add jumps)
/*
        TODO: Split optimization into two passes, the first one only simplifies the graph,
        the second one adds jumps and other optimizations and caching. This pipeline is not
        serializable as some operations are internal to the pipeline and not part of the operation list.

        auto tokens = stateless_pass(state->operations);

        // Replace all nodes with constant values with constant operations.
        // Operations that are not needed will be pruned during graph optimization.
        for (size_t i = 0; i < state->operations.size(); i++)
        {
            if (!_IS_PLACEHOLDER(tokens[i]))
            {
                state->operations[i].operation = create<Constant>(tokens[i]);
            }
        }
*/
        state->operations = optimize_pipeline(state->operations, optimize);

        auto tokens = stateless_pass(state->operations);

        state->cache.resize(state->operations.size());
        state->cache_initializers.clear();
        state->cache_initializers.reserve(state->operations.size());
        state->operation_mutexes.clear();
        state->operation_mutexes.resize(state->operations.size());
        for (size_t i = 0; i < state->operations.size(); i++)
        {
            state->cache_initializers.push_back(std::make_unique<std::once_flag>());
            if (state->operations[i].operation->trait() & static_cast<int>(OperationTrait::Stateful))
                state->operation_mutexes[i] = std::make_unique<std::mutex>();
        }
        state->stats.resize(state->operations.size());
        state->constants.resize(state->operations.size());

        int output_position = 0;

        for (size_t i = 0; i < state->stats.size(); i++)
        {
            state->stats[i].count = 0;
            state->stats[i].elapsed = 0;
        }

        for (size_t i = 0; i < state->operations.size(); i++)
        {
            bool constant = true;

            for (int j : state->operations[i].inputs)
            {
                if (!state->constants[j])
                {
                    constant = false;
                    break;
                }
            }

            if (is_context(state->operations[i].operation))
                constant = false;

            state->constants[i] = constant;

            // If the operation is an output, we need to store the inferred stateless shape
            if (is_output(state->operations[i].operation))
            {
                auto label = (state->operations[i].operation.get_as<Output>())->get_label();
                state->outputs[output_position].shape = tokens[i]->shape();
                output_position++;
            }
        }

        // Publish all finalized state at once. A concurrent run either sees
        // the complete immutable pipeline or fails the finalized check.
        state->finalized.store(true, std::memory_order_release);
    }

    int Pipeline::append(std::string name, const TokenList &args, const Span<int> &inputs, const Metadata &metadata)
    {

        if (state->finalized.load(std::memory_order_acquire))
            throw PipelineException("Pipeline is finalized");

        OperationReference operation = create_operation(name, args);

        for (int i : inputs)
        {
            if (i >= (int)state->operations.size() || i < 0)
                throw OperationException("Operation index out of bounds", operation, (int)state->operations.size());

            if (is_output(state->operations[i].operation))
            {
                throw OperationException("Cannot refer to output operation", operation, (int)state->operations.size());
            }
        }

        state->operations.push_back({std::move(operation), std::vector<int>(inputs.begin(), inputs.end()), metadata});

        if (is_output(state->operations.back().operation))
        {
            auto label = (state->operations.back().operation.get_as<Output>())->get_label();
            for (size_t i = 0; i < inputs.size(); i++)
                state->outputs.push_back({label, AnythingShape()});
        }

        return (int)(state->operations.size() - 1);
    }

    Sequence<OutputDescription> Pipeline::outputs() const
    {

        if (!state->finalized.load(std::memory_order_acquire))
            throw PipelineException("Pipeline not finalized");

        return Sequence<OutputDescription>(state->outputs);
    }

    size_t Pipeline::size() const
    {
        return state->operations.size();
    }

    Metadata &Pipeline::metadata()
    {
        return state->metadata;
    }

    const Metadata &Pipeline::metadata() const
    {
        return state->metadata;
    }

    Pipeline::OperationData Pipeline::get(size_t i) const
    {
        return {state->operations[i].operation.reborrow(), state->operations[i].inputs, state->operations[i].metadata};
    }

    Sequence<TokenReference> Pipeline::run(unsigned long index)
    {

        vector<TokenReference> context;
        vector<TokenReference> result;

        if (!state->finalized.load(std::memory_order_acquire))
            throw PipelineException("Pipeline not finalized");

        context.resize(state->operations.size());

        size_t i = 0;

        RandomGenerator generator = make_generator((int)index);

        auto record_stats = [this](size_t operation, unsigned long elapsed)
        {
            std::lock_guard<std::mutex> lock(state->stats_mutex);
            state->stats[operation].count++;
            state->stats[operation].elapsed += elapsed;
        };

        while (i < state->operations.size())
        {

            const auto operation_type = state->operations[i].operation->type();
            auto run_operation = [this, i](const TokenList &inputs)
            {
                if (state->operation_mutexes[i])
                {
                    std::lock_guard<std::mutex> lock(*state->operation_mutexes[i]);
                    return state->operations[i].operation->run(inputs);
                }
                return state->operations[i].operation->run(inputs);
            };
            const bool cacheable = state->constants[i] &&
                                   operation_type != GetType<Output>() &&
                                   operation_type != GetType<Jump>() &&
                                   operation_type != GetType<ConditionalJump>();

            if (cacheable)
            {
                // Constant results are shared by all runs; publish each cache slot once.
                std::call_once(*state->cache_initializers[i], [&]()
                {
                    vector<TokenReference> inputs;
                    inputs.reserve(state->operations[i].inputs.size());
                    for (int j : state->operations[i].inputs)
                        inputs.push_back(state->cache[j].reborrow());

                    try
                    {
                        auto operation_start = high_resolution_clock::now();
                        auto output = run_operation(make_span(inputs));
                        auto operation_end = high_resolution_clock::now();

                        if (!output)
                            throw OperationException("Operation output undefined", state->operations[i].operation, (int)i);

                        state->cache[i] = std::move(output);
                        record_stats(i, (unsigned long)duration_cast<microseconds>(operation_end - operation_start).count());
                    }
                    catch (BaseException &e)
                    {
                        std::cout << "ERROR at operation " << i << " " << operation_name(state->operations[i].operation) << ": " << e.what() << ", inputs:" << std::endl;
                        int k = 0;
                        for (int j : state->operations[i].inputs)
                        {
                            std::cout << " * " << k << " (operation " << j << " " << operation_name(state->operations[j].operation) << "): ";
                            if ((bool)(inputs[k]))
                                std::cout << inputs[k]->describe() << std::endl;
                            else
                                std::cout << "undefined" << std::endl;
                            k++;
                        }
                        throw OperationException(e.what(), state->operations[i].operation, (int)i);
                    }
                });

                context[i] = state->cache[i].reborrow();
                i++;
                continue;
            }

            vector<TokenReference> local;
            local.reserve(state->operations[i].inputs.size());

            for (int j : state->operations[i].inputs)
            {
                local.push_back(context[j].reborrow());
            }

            try
            {
                auto operation_start = high_resolution_clock::now();

                auto output = run_operation(make_span(local));
                auto operation_end = high_resolution_clock::now();

                if ((state->operations[i].operation->type()) == GetType<ContextQuery>())
                {
                    switch ((state->operations[i].operation.get_as<ContextQuery>())->get_query())
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
                        throw OperationException("Illegal query", state->operations[i].operation, (int)i);
                    }
                }

                record_stats(i, (unsigned long)duration_cast<microseconds>(operation_end - operation_start).count());

                if ((state->operations[i].operation)->type() == GetType<Output>())
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
                    throw OperationException("Operation output undefined", state->operations[i].operation, (int)i);
                }

                context[i] = std::move(output);

                if (operation_type == GetType<Jump>() || operation_type == GetType<ConditionalJump>())
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
                std::cout << "ERROR at operation " << i << " " << operation_name(state->operations[i].operation) << ": " << e.what() << ", inputs:" << std::endl;
                int k = 0;
                for (int j : state->operations[i].inputs)
                {
                    std::cout << " * " << k << " (operation " << j << " " << operation_name(state->operations[j].operation) << "): ";
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
                throw OperationException(e.what(), state->operations[i].operation, (int)i);
            }

            i++;
        }

        VERIFY(result.size() == state->outputs.size(), "Output mismatch");

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
            auto name = operation_name(operation.operation);

            if (operation.operation->is<ConditionalJump>())
            {
                name = "cjump";
            }

            stream << "OP" << n << " [ordering=in label=\"" << n << " - " << name << "\" ";
            if (operation.operation->is<Output>())
            {
                stream << " shape=box";
                outputs.push_back(n);
            }
            else if (operation.operation->is<ContextQuery>())
            {
                stream << " shape=hexagon";
            }

            stream << "];" << std::endl;

            for (auto i : operation.inputs)
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
        std::lock_guard<std::mutex> lock(state->stats_mutex);

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

    namespace
    {
        class AutoscalingQueue
        {
        public:
            using Task = std::function<void()>;

            AutoscalingQueue(size_t minimum, size_t maximum) : minimum_(minimum), maximum_(std::max(minimum, maximum))
            {
                std::lock_guard<std::mutex> guard(lock_);
                ensure_workers_locked(minimum_);
            }

            ~AutoscalingQueue()
            {
                {
                    std::lock_guard<std::mutex> guard(lock_);
                    stopping_ = true;
                }
                condition_.notify_all();
                for (auto &worker : workers_)
                {
                    if (worker.joinable())
                        worker.join();
                }
            }

            void dispatch(Task task)
            {
                {
                    std::lock_guard<std::mutex> guard(lock_);
                    if (stopping_)
                        throw IllegalStateException("Engine is stopping");

                    queue_.push(std::move(task));

                    // A task which arrives while all current workers are
                    // occupied is a demand signal.  Retired workers are
                    // ignored here; only live workers are counted.
                    if (live_workers_ < maximum_ && queue_.size() > 0)
                        ensure_workers_locked(std::min(maximum_, live_workers_ + 1));
                }
                condition_.notify_one();
            }

        private:
            void ensure_workers_locked(size_t target)
            {
                while (live_workers_ < target)
                {
                    ++live_workers_;
                    workers_.emplace_back([this]() { worker_loop(); });
                }
            }

            void worker_loop()
            {
                for (;;)
                {
                    Task task;
                    {
                        std::unique_lock<std::mutex> guard(lock_);
                        if (!condition_.wait_for(guard, milliseconds(100), [this]() { return stopping_ || !queue_.empty(); }))
                        {
                            if (live_workers_ > minimum_)
                            {
                                --live_workers_;
                                return;
                            }
                            continue;
                        }

                        if (queue_.empty())
                        {
                            if (stopping_)
                            {
                                --live_workers_;
                                return;
                            }
                            continue;
                        }

                        task = std::move(queue_.front());
                        queue_.pop();
                    }

                    try
                    {
                        task();
                    }
                    catch (...)
                    {
                        // Tasks own their error reporting.  A task escaping
                        // here must never terminate the process.
                    }
                }
            }

            size_t minimum_;
            size_t maximum_;
            size_t live_workers_ = 0;
            bool stopping_ = false;
            std::mutex lock_;
            std::condition_variable condition_;
            std::queue<Task> queue_;
            std::list<std::thread> workers_;
        };

        template <typename T>
        TokenReference stack_scalar(const std::vector<TokenReference> &samples)
        {
            auto result = create<Matrix<T>>(samples.size(), 1);
            auto values = result->data().template reinterpret<T>();
            for (size_t i = 0; i < samples.size(); ++i)
                values[i] = extract<T>(samples[i]);
            return result;
        }

        TokenReference stack_output(std::vector<TokenReference> samples)
        {
            if (samples.empty() || !samples[0])
                throw PipelineException("Unable to batch an empty pipeline output");

            if (samples[0]->is<Tensor>())
            {
                auto first = extract<TensorReference>(samples[0]);
                auto shape = first->shape().push(samples.size());
                auto result = create_tensor(shape);

                for (size_t i = 0; i < samples.size(); ++i)
                {
                    if (!samples[i] || !samples[i]->is<Tensor>() || !(extract<TensorReference>(samples[i])->shape() == first->shape()))
                        throw PipelineException("Pipeline outputs cannot be batched because their tensor shapes differ");
                    auto destination = extract<TensorReference>(result->get(i));
                    copy_buffer(extract<TensorReference>(samples[i]), destination);
                }
                return result;
            }

            if (samples[0]->is<IntegerScalar>())
                return stack_scalar<int>(samples);
            if (samples[0]->is<FloatScalar>())
                return stack_scalar<float>(samples);
            // Python's scalar conversion historically exposes booleans as
            // integer arrays, so retain that parity for native batches.
            if (samples[0]->is<BooleanScalar>())
                return stack_scalar<int>(samples);
            if (samples[0]->is<CharScalar>())
                return stack_scalar<char>(samples);
            if (samples[0]->is<ShortScalar>())
                return stack_scalar<short>(samples);
            if (samples[0]->is<UShortScalar>())
                return stack_scalar<ushort>(samples);

            throw PipelineException("Pipeline output type is not supported by native batching");
        }

        Sequence<TokenReference> stack_batch(std::vector<Sequence<TokenReference>> &samples)
        {
            if (samples.empty())
                throw PipelineException("Unable to batch zero samples");

            const size_t output_count = samples[0].size();
            for (const auto &sample : samples)
            {
                if (sample.size() != output_count)
                    throw PipelineException("Pipeline outputs have different lengths");
            }

            Sequence<TokenReference> result(output_count);
            for (size_t output = 0; output < output_count; ++output)
            {
                std::vector<TokenReference> values;
                values.reserve(samples.size());
                for (auto &sample : samples)
                    values.push_back(std::move(sample[output]));
                result[output] = stack_output(std::move(values));
            }
            return result;
        }
    }

    struct Engine::State
    {
        struct BatchWork
        {
            State *owner;
            uint64_t order;
            Pipeline *pipeline;
            std::vector<Sequence<TokenReference>> samples;

            BatchWork(State *owner, uint64_t order, Pipeline *pipeline, size_t size)
                : owner(owner), order(order), pipeline(pipeline), samples(size) {}
        };

        std::mutex batch_lock;
        std::condition_variable batch_ready;
        std::map<uint64_t, Sequence<TokenReference>> ready;
        Pipeline *pipeline;
        size_t batch_size;
        size_t queue_capacity;
        size_t outstanding = 0;
        unsigned long next_index = 1;
        uint64_t next_order = 0;
        uint64_t consume_order = 0;
        bool stream_failed = false;
        std::string stream_error;
        // Keep the worker pool last so it is destroyed first and cannot
        // access state that has already been torn down.
        AutoscalingQueue workers;

        State(Pipeline &pipeline, size_t batch_size, uint16_t minimum, uint16_t maximum,
              size_t queue_capacity, unsigned long first_index)
            : pipeline(&pipeline), batch_size(batch_size),
              queue_capacity(std::max<size_t>(1, queue_capacity)),
              next_index(first_index),
              workers(minimum, maximum == 0 ? std::max<size_t>(minimum, std::thread::hardware_concurrency()) : maximum)
        {
            if (batch_size == 0)
                throw IllegalStateException("Batch size should be one or more");

            std::lock_guard<std::mutex> guard(batch_lock);
            for (size_t i = 0; i < queue_capacity; ++i)
                submit_batch_locked();
        }

        void fail_batch(const std::string &error)
        {
            std::lock_guard<std::mutex> guard(batch_lock);
            if (!stream_failed)
            {
                stream_failed = true;
                stream_error = error;
            }
            batch_ready.notify_all();
        }

        void complete_batch(const std::shared_ptr<BatchWork> &work)
        {
            try
            {
                auto result = stack_batch(work->samples);
                std::lock_guard<std::mutex> guard(batch_lock);
                ready.emplace(work->order, std::move(result));
                batch_ready.notify_all();
            }
            catch (const BaseException &error)
            {
                fail_batch(error.what());
            }
            catch (const std::exception &error)
            {
                fail_batch(error.what());
            }
            catch (...)
            {
                fail_batch("Unknown exception while batching pipeline output");
            }
        }

        void submit_batch_locked()
        {
            const uint64_t order = next_order++;
            auto work = std::make_shared<BatchWork>(this, order, pipeline, batch_size);
            outstanding++;

            std::vector<unsigned long> indices(batch_size);
            for (size_t sample = 0; sample < batch_size; ++sample)
                indices[sample] = next_index++;

            // A worker owns a complete batch. This keeps batching native and
            // makes scaling respond to the number of batches the consumer
            // can drain, without creating a per-sample task fan-out.
            workers.dispatch([work, indices = std::move(indices)]()
            {
                try
                {
                    for (size_t sample = 0; sample < indices.size(); ++sample)
                        work->samples[sample] = work->pipeline->run(indices[sample]);
                    work->owner->complete_batch(work);
                }
                catch (const BaseException &error)
                {
                    work->owner->fail_batch(error.what());
                }
                catch (const std::exception &error)
                {
                    work->owner->fail_batch(error.what());
                }
                catch (...)
                {
                    work->owner->fail_batch("Unknown exception while running pipeline");
                }
            });
        }

        Sequence<TokenReference> next()
        {
            std::unique_lock<std::mutex> guard(batch_lock);
            batch_ready.wait(guard, [this]() { return stream_failed || ready.find(consume_order) != ready.end(); });
            if (stream_failed)
                throw PipelineException(stream_error);

            auto result = std::move(ready.find(consume_order)->second);
            ready.erase(consume_order++);
            --outstanding;
            submit_batch_locked();
            return result;
        }
    };

    void Engine::StateDeleter::operator()(State *p) const
    {
        delete p;
    }

    Engine::Engine(Pipeline &pipeline, size_t batch_size, uint16_t workers, uint16_t max_workers,
                   size_t queue_size, unsigned long first_index) : state(nullptr)
    {
        if (workers == 0)
            throw IllegalStateException("Engine requires at least one worker");

        state.reset(new Engine::State(pipeline, batch_size, workers, max_workers, queue_size, first_index));
    }

    Engine::~Engine() = default;

    Sequence<TokenReference> Engine::next_batch() noexcept(false)
    {
        return state->next();
    }
    PIXELPIPES_REGISTER_ENUM("context", ContextData);
}
