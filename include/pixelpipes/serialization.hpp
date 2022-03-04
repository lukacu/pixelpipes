#pragma once

#include <iostream>
#include <functional>
#include <set>

#include <pixelpipes/base.hpp>
#include <pixelpipes/types.hpp>
#include <pixelpipes/operation.hpp>
#include <pixelpipes/pipeline.hpp>

namespace pixelpipes {

typedef std::function<void(SharedVariable, std::ostream&)> VariableWriter;

typedef std::function<SharedVariable(std::istream&)> VariableReader;

class PIXELPIPES_API SerializationException : public BaseException {
public:
    SerializationException(std::string reason);
};

class PipelineWriter: public std::enable_shared_from_this<PipelineWriter> {
public:

    PipelineWriter();

    ~PipelineWriter() = default;

    void write(std::ostream& target);

    void write(std::string& target);

    int append(std::string name, std::vector<SharedVariable> args, std::vector<int> inputs);

    template<typename T>
    static void register_writer(VariableWriter writer) {

        Type<T> variable_type;

        register_writer(variable_type.identifier, variable_type.name, writer);

    }

private:

    typedef std::tuple<std::string, std::vector<int>, std::vector<int> > OperationData;
    typedef std::tuple<VariableWriter, TypeName, SharedModule> WriterData;
    typedef std::map<TypeIdentifier, WriterData> WriterMap;

    static WriterMap& writers();

    static void register_writer(TypeIdentifier identifier, std::string_view name, VariableWriter writer);

    std::set<SharedModule> used_modules;
    std::set<TypeIdentifier> used_types;

    std::vector<SharedVariable> variables;
    std::vector<OperationData> operations;

};

class PipelineReader: public std::enable_shared_from_this<PipelineReader> {
public:

    PipelineReader();

    ~PipelineReader() = default;

    SharedPipeline read(std::istream& target);

    SharedPipeline read(std::string& target);

    template<typename T>
    static void register_reader(VariableReader reader) {

        Type<T> variable_type;

        register_reader(variable_type.identifier, variable_type.name, reader);


    }

private:

    typedef std::tuple<VariableReader, TypeName, SharedModule> ReaderData;
    typedef std::map<TypeIdentifier, ReaderData> ReaderMap;

    static ReaderMap& readers();

    static void register_reader(TypeIdentifier identifier, std::string_view name, VariableReader reader);

};

#define PIXELPIPES_REGISTER_READER(T, F) static AddModuleInitializer CONCAT(__reader_init_, __COUNTER__) ([]() { PipelineReader::register_reader<T> ( F ); } )
#define PIXELPIPES_REGISTER_WRITER(T, F) static AddModuleInitializer CONCAT(__writer_init_, __COUNTER__) ([]() { PipelineWriter::register_writer<T> ( F ); } )

}
