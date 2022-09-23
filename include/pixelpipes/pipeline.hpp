#pragma once

#include <type_traits>
#include <mutex>
#include <thread>
#include <exception>
#include <random>
#include <iostream>

#include <pixelpipes/base.hpp>
#include <pixelpipes/token.hpp>
#include <pixelpipes/operation.hpp>

namespace pixelpipes
{

    class Pipeline;
    class PipelineException;

    bool PIXELPIPES_API is_output(OperationReference &op);

    bool PIXELPIPES_API is_conditional(OperationReference &op);

    bool PIXELPIPES_API is_constant(OperationReference &op);

    bool PIXELPIPES_API is_context(OperationReference &op);

    class PIXELPIPES_API Pipeline
    {
        struct State;

        struct StateDeleter {

            void operator()(State* p) const;
        };

        // TODO: figure out what kind of magic is done here and replicate it.
        std::unique_ptr<State, StateDeleter> state;

    public:
        typedef std::pair<OperationReference, Sequence<int>> OperationData;

        Pipeline();

        virtual ~Pipeline();

        Pipeline(Pipeline &&);

        Pipeline &operator=(Pipeline &&);

        virtual void finalize(bool optimize = true);

        virtual int append(std::string name, const TokenList& args, const Span<int>& inputs);

        virtual Sequence<TokenReference> run(unsigned long index) noexcept(false);

        size_t size() const;

        OperationData get(size_t i) const;

        virtual Sequence<std::string> get_labels() const;

        std::vector<float> operation_time();

    protected:
        typedef struct
        {
            int count;
            unsigned long elapsed;
        } OperationStats;
    };

    std::string PIXELPIPES_API visualize_pipeline(const Pipeline& pipeline);

    class PipelineCallback
    {
    public:
        virtual void done(TokenList result) = 0;

        virtual void error(const PipelineException &error) = 0;
    };

    class PIXELPIPES_API PipelineException : public BaseException
    {
    public:
        PipelineException(std::string reason);
        PipelineException(const PipelineException& e) = default;
    };

    class PIXELPIPES_API OperationException : public PipelineException
    {
    public:
        OperationException(std::string reason, const OperationReference& reference, int position);
        OperationException(const OperationException& e) = default;

        inline int position() const
        {
            return _position;
        }
        
    private:
        int _position;
        std::string name;
    };

    /*
    class Engine: public std::enable_shared_from_this<Engine> {
    public:
        Engine(int workers);

        ~Engine();

        void start();

        void stop();

        bool running();

        bool add(std::string, SharedPipeline pipeline);

        bool remove(std::string);

        void run(std::string, unsigned long index, std::shared_ptr<PipelineCallback> callback) noexcept(false);

    private:

        std::map<std::string, SharedPipeline> pipelines;

        std::shared_ptr<dispatch_queue> workers;

        std::recursive_mutex mutex_;

        int workers_count;

    };

    class EngineException: public BaseException {
    public:

        EngineException(std::string reason);

    };*/

}
