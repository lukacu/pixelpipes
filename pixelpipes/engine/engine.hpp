#ifndef __PP_ENGINE_H__
#define __PP_ENGINE_H__

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

#define DEBUG(X) {std::cout << X << std::endl;}

namespace pixelpipes {

enum class VariableType {Integer, Float, Image, Points, View, List};

enum class OperationType {Deterministic, Stohastic, Output, Control};

class Variable;
typedef std::shared_ptr<Variable> SharedVariable;

class Variable {
public:
    virtual VariableType type() = 0;

};

template <VariableType T> 
class TypedVariable {
public:

    virtual VariableType type() { return T; };

};

class Pipeline;
class OperationException;
class PipelineException;

class VariableException : public std::exception {
public:

    VariableException(std::string reason):
        reason(reason) {}

	const char * what () const throw () {
    	return reason.c_str();
    }
private:
    std::string reason;
};

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

    Jump();
    ~Jump() = default;

    virtual SharedVariable run(std::vector<SharedVariable> inputs, ContextHandle context);

protected:

    virtual OperationType type();

};


class OperationException : public std::exception {
public:

    OperationException(std::string reason, SharedOperation operation):
        reason(reason), operation(operation) {}

	const char * what () const throw () {
    	return reason.c_str();
    }
private:
    std::string reason;
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


class PipelineException : public std::exception {
public:

    PipelineException(std::string reason, SharedPipeline pipeline, int operation):
        reason(reason), pipeline(pipeline), _operation(operation) {}

    const char * what () const throw () {
        return reason.c_str();
    }

	int operation () const throw () {
    	return _operation;
    }

private:
    std::string reason;
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

class EngineException : public std::exception {
public:

    EngineException(std::string reason): reason(reason) {}

	const char * what () const throw () {
    	return reason.c_str();
    }
private:
    std::string reason;
};

}

#endif