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

    class PIXELPIPES_API DNF
    {
    public:
        DNF();

        DNF(Span<Span<bool>> clauses);
        DNF(std::istream &source);

        DNF(const DNF &);
        DNF(DNF &&);

        DNF &operator=(const DNF &);
        DNF &operator=(DNF &&);

        ~DNF();

        size_t size() const;

        bool compute(TokenList values) const;

        void write(std::ostream &target) const;

    private:
        struct State;

        Implementation<State> state;
    };

#define DNFType GetTypeIdentifier<DNF>()

    template <>
    DNF PIXELPIPES_API extract(const SharedToken v);

    template <>
    SharedToken PIXELPIPES_API wrap(const DNF v);

    typedef std::pair<SharedOperation, std::vector<int>> OperationData;


    class PIXELPIPES_API Pipeline
    {
        struct State;

        struct StateDeleter {

            void operator()(State* p) const;
        };

        // TODO: figure out what kind of magic is done here and replicate it.
        std::unique_ptr<State, StateDeleter> state;

    public:
        Pipeline();

        virtual ~Pipeline();

        //Pipeline(const Pipeline &);
        Pipeline(Pipeline &&);

        //Pipeline &operator=(const Pipeline &);
        Pipeline &operator=(Pipeline &&);

        virtual void finalize();

        virtual int append(std::string name, TokenList args, Span<int> inputs);

        virtual Sequence<SharedToken> run(unsigned long index) noexcept(false);

        virtual std::vector<std::string> get_labels();

        std::vector<float> operation_time();

    protected:
        typedef struct
        {
            int count;
            unsigned long elapsed;
        } OperationStats;
    };

    class PipelineCallback
    {
    public:
        virtual void done(TokenList result) = 0;

        virtual void error(const PipelineException &error) = 0;
    };

    class PIXELPIPES_API PipelineException : public BaseException
    {
    public:
        PipelineException(std::string reason, int operation);
        int operation() const throw();

    private:
        int _operation;
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
