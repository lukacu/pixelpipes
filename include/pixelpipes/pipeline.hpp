#pragma once

#include <vector>
#include <type_traits>
#include <map>
#include <set>
#include <mutex>
#include <thread>
#include <exception>
#include <random>
#include <iostream>

#include <pixelpipes/base.hpp>
#include <pixelpipes/token.hpp>
#include <pixelpipes/operation.hpp>

namespace pixelpipes { 
 
class Pipeline;
class PipelineException;

class DNF: public std::vector<std::vector<bool> > {
public:

    DNF(std::vector<std::vector<bool> > clauses);
    ~DNF() = default;

};

constexpr static TypeIdentifier DNFType = GetTypeIdentifier<DNF>();

template<>
inline DNF extract(const SharedToken v) {
    VERIFY((bool) v, "Uninitialized variable");

    VERIFY(v->type() == DNFType, "Illegal type");

    auto container = std::static_pointer_cast<ContainerToken<DNF>>(v);
    return container->get();
    

}

template<>
inline SharedToken wrap(const DNF v) {
    return std::make_shared<ContainerToken<DNF>>(v);
}

class Pipeline: public std::enable_shared_from_this<Pipeline>, public OperationObserver {
public:

    Pipeline();

    ~Pipeline() = default;

    virtual void finalize();

    virtual int append(std::string name, std::vector<SharedToken> args, std::vector<int> inputs);

    virtual std::vector<SharedToken> run(unsigned long index) noexcept(false);

    virtual std::vector<std::string> get_labels();

    std::vector<float> operation_time();

protected:

    typedef struct {
        int count;
        unsigned long elapsed;
    } OperationStats;

    bool finalized;

    std::vector<SharedToken> cache;

    std::vector<std::pair<SharedOperation, std::vector<int> > > operations;

    std::vector<OperationStats> stats;

    std::vector<std::string> labels;

};



typedef std::shared_ptr<Pipeline> SharedPipeline;

class PipelineCallback {
public:
    virtual void done(std::vector<SharedToken> result) = 0;

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

};

class EngineException: public BaseException {
public:

    EngineException(std::string reason);

};*/

}
