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

    // Output description struct
    typedef struct OutputDescription
    {
        std::string label;
        Shape shape;
    } OutputDescription;

    class PIXELPIPES_API Metadata
    {
    public:
        Metadata();
        Metadata(const Metadata &);
        Metadata(Metadata &&);
        ~Metadata();

        Metadata &operator=(const Metadata &);
        Metadata &operator=(Metadata &&);

        std::string get(std::string key) const;
        bool has(std::string key) const;
        void set(std::string key, std::string value);
        size_t size() const;

        Sequence<std::string> keys() const;

    private:
        struct State;
        Implementation<State> _state;
    };

    class PIXELPIPES_API Pipeline
    {
        struct State;

        struct StateDeleter
        {

            void operator()(State *p) const;
        };

        // TODO: figure out what kind of magic is done here and replicate it.
        std::unique_ptr<State, StateDeleter> state;

    public:
        struct OperationData
        {
            OperationReference operation;
            Sequence<int> inputs;
            Metadata metadata;
        };

        Pipeline();

        virtual ~Pipeline();

        Pipeline(Pipeline &&);

        Pipeline &operator=(Pipeline &&);

        virtual void finalize(bool optimize = true);

        virtual int append(std::string name, const TokenList &args, const Span<int> &inputs, const Metadata &metadata = Metadata());

        virtual Sequence<TokenReference> run(unsigned long index) noexcept(false);

        size_t size() const;

        OperationData get(size_t i) const;

        virtual Sequence<OutputDescription> outputs() const;

        std::vector<float> operation_time();

        Metadata &metadata();

        const Metadata &metadata() const;

    protected:
        typedef struct
        {
            int count;
            unsigned long elapsed;
        } OperationStats;
    };

    std::string PIXELPIPES_API visualize_pipeline(const Pipeline &pipeline);

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
        PipelineException(const PipelineException &e) = default;
    };

    class PIXELPIPES_API OperationException : public PipelineException
    {
    public:
        OperationException(std::string reason, const OperationReference &reference, int position);
        OperationException(const OperationException &e) = default;

        inline int position() const
        {
            return _position;
        }

    private:
        int _position;
        std::string name;
    };
/*
    class PIXELPIPES_API Engine
    {
        struct State;

    public:
    
        Engine(uint16_t workers);

        ~Engine();

        void run(Pipeline& pipeline, unsigned long index, std::shared_ptr<PipelineCallback> callback) noexcept(false);

        void batch(Pipeline& pipeline, const Sequence<unsigned long>& indices, std::shared_ptr<PipelineCallback> callback) noexcept(false);

    private:

        struct StateDeleter
        {
            void operator()(State *p) const;
        };

        // TODO: figure out what kind of magic is done here and replicate it.
        std::unique_ptr<State, StateDeleter> state;

    };
*/
}
