#pragma once

#include <memory>
#include <vector>
#include <type_traits>
#include <map>
#include <mutex>
#include <thread>
#include <exception>
#include <random>
#include <iostream>

#include "queue.hpp"
#include "types.hpp"

namespace pixelpipes {

enum class OperationType {Deterministic, Stohastic, Output, Control};

enum class ContextData {SampleIndex};

class Pipeline;
class OperationException;
class PipelineException;

class Context {
public:
    Context(unsigned long index);
    ~Context() = default;

    unsigned int random();
    unsigned long sample();

private:

    unsigned long index;
    std::default_random_engine generator;

};

typedef std::shared_ptr<Context> ContextHandle;

class Operation: public std::enable_shared_from_this<Operation> {
friend Pipeline;
public:
    
    ~Operation();

    virtual SharedVariable run(std::vector<SharedVariable> inputs, ContextHandle context) = 0;

protected:

    Operation();

    virtual OperationType type();

};

typedef std::shared_ptr<Operation> SharedOperation;

class Output: public Operation {
public:

    Output();
    ~Output();

    virtual SharedVariable run(std::vector<SharedVariable> inputs, ContextHandle context);

protected:

    virtual OperationType type();

};

class Copy: public Operation {
public:

    virtual SharedVariable run(std::vector<SharedVariable> inputs, ContextHandle context);

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

typedef std::vector<std::vector<bool> > DNF;

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

class OperationException : public BaseException {
public:

    OperationException(std::string reason, SharedOperation operation): BaseException(reason), operation(operation) {}

private:

    SharedOperation operation;
};


class Pipeline: public std::enable_shared_from_this<Pipeline> {
public:

    Pipeline();

    ~Pipeline();

    void finalize();

    int append(SharedOperation operation, std::vector<int> inputs);

    std::vector<SharedVariable> run(unsigned long index) noexcept(false);

private:

    bool finalized;

    std::vector<SharedVariable> cache;

    std::vector<std::pair<SharedOperation, std::vector<int> > > operations;

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

class EngineException : public BaseException {
public:

    EngineException(std::string reason);

};

}
