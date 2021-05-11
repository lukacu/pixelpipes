#pragma once

#include <vector>
#include <type_traits>
#include <map>
#include <mutex>
#include <thread>
#include <exception>
#include <random>
#include <iostream>

#include <pixelpipes/base.hpp>
#include <pixelpipes/types.hpp>
#include <pixelpipes/operation.hpp>

namespace pixelpipes { 
 
class Pipeline;
class PipelineException;

class Output: public Operation {
public:

    Output();
    ~Output();

    virtual SharedVariable run(std::vector<SharedVariable> inputs, ContextHandle context);

protected:

    virtual OperationType type();

};

class Jump: public Operation {
public:

    Jump(int offset);
    ~Jump() = default;

    virtual SharedVariable run(std::vector<SharedVariable> inputs, ContextHandle context);

protected:

    int offset;

    virtual OperationType type();

};

class Constant : public Operation {
public:

    Constant(SharedVariable value);

    virtual SharedVariable run(std::vector<SharedVariable> inputs, ContextHandle context);

private:

    SharedVariable value;

};

typedef struct { std::vector<std::vector<bool> > clauses; } DNF;

constexpr static TypeIdentifier DNFType = Type<DNF>::identifier;

template<>
struct Conversion<DNF> {

    static DNF extract(const SharedVariable v) {
        VERIFY((bool) v, "Uninitialized variable");

        VERIFY(v->type() == Type<DNF>::identifier, "Illegal type");

        auto container = std::static_pointer_cast<ContainerVariable<DNF>>(v);
        return container->get();
        

    }

    static SharedVariable wrap(const DNF v) {
        return std::make_shared<ContainerVariable<DNF>>(v);
    }

};

class ConditionalJump: public Jump {
public:

    ConditionalJump(DNF condition, int offset);
    ~ConditionalJump() = default;

    virtual SharedVariable run(std::vector<SharedVariable> inputs, ContextHandle context);

private:

    DNF condition;

};

class Conditional: public Operation {
public:

    Conditional(DNF condition);
    ~Conditional() = default;

    virtual SharedVariable run(std::vector<SharedVariable> inputs, ContextHandle context);

private:

    DNF condition;

};

class ContextQuery: public Operation {
public:

    ContextQuery(ContextData query);
    ~ContextQuery() = default;

    virtual SharedVariable run(std::vector<SharedVariable> inputs, ContextHandle context);

protected:

    ContextData query;

};

class DebugOutput: public Operation {
public:

    DebugOutput(std::string prefix);
    ~DebugOutput() = default;

    virtual SharedVariable run(std::vector<SharedVariable> inputs, ContextHandle context);

protected:

    std::string prefix;

};



class Pipeline: public std::enable_shared_from_this<Pipeline>, public OperationObserver {
public:

    Pipeline();

    ~Pipeline() = default;

    virtual void finalize();

    virtual int append(SharedOperation operation, std::vector<int> inputs);

    virtual std::vector<SharedVariable> run(unsigned long index) noexcept(false);

    std::vector<float> operation_time();

protected:

    typedef struct {
        int count;
        unsigned long elapsed;
    } OperationStats;

    bool finalized;

    std::vector<SharedVariable> cache;

    std::vector<std::pair<SharedOperation, std::vector<int> > > operations;

    std::vector<OperationStats> stats;


};

typedef std::shared_ptr<Pipeline> SharedPipeline;

class PipelineCallback {
public:
    virtual void done(std::vector<SharedVariable> result) = 0;

    virtual void error(const PipelineException &error) = 0;

};

class PipelineException : public BaseException {
public:

    PipelineException(std::string reason, SharedPipeline pipeline, int operation);
	int operation () const throw ();

private:
    SharedPipeline pipeline; 
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

};*/

class EngineException: public BaseException {
public:

    EngineException(std::string reason);

};

}
