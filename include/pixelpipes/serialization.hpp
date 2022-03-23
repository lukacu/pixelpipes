#pragma once

#include <iostream>
#include <functional>
#include <set>

#include <pixelpipes/base.hpp>
#include <pixelpipes/token.hpp>
#include <pixelpipes/operation.hpp>
#include <pixelpipes/pipeline.hpp>

namespace pixelpipes
{

    typedef std::function<void(SharedToken, std::ostream &)> TokenWriter;

    typedef std::function<SharedToken(std::istream &)> TokenReader;

    class PIXELPIPES_API SerializationException : public BaseException
    {
    public:
        SerializationException(std::string reason);
    };

    class PipelineWriter : public std::enable_shared_from_this<PipelineWriter>
    {
    public:
        PipelineWriter();

        ~PipelineWriter() = default;

        void write(std::ostream &target);

        void write(std::string &target);

        int append(std::string name, std::vector<SharedToken> args, std::vector<int> inputs);

        template <typename T>
        static void register_writer(TokenWriter writer)
        {

            Type<T> token_type;

            register_writer(token_type.identifier, token_type.name, writer);
        }

    private:
        typedef std::tuple<std::string, std::vector<int>, std::vector<int>> OperationData;
        typedef std::tuple<TokenWriter, TypeName, SharedModule> WriterData;
        typedef std::map<TypeIdentifier, WriterData> WriterMap;

        static WriterMap &writers();

        static void register_writer(TypeIdentifier identifier, std::string_view name, TokenWriter writer);

        std::set<SharedModule> used_modules;
        std::set<TypeIdentifier> used_types;

        std::vector<SharedToken> tokens;
        std::vector<OperationData> operations;
    };

    class PipelineReader : public std::enable_shared_from_this<PipelineReader>
    {
    public:
        PipelineReader();

        ~PipelineReader() = default;

        SharedPipeline read(std::istream &target);

        SharedPipeline read(std::string &target);

        template <typename T>
        static void register_reader(TokenReader reader)
        {

            Type<T> token_type;

            register_reader(token_type.identifier, token_type.name, reader);
        }

    private:
        typedef std::tuple<TokenReader, TypeName, SharedModule> ReaderData;
        typedef std::map<TypeIdentifier, ReaderData> ReaderMap;

        static ReaderMap &readers();

        static void register_reader(TypeIdentifier identifier, std::string_view name, TokenReader reader);
    };

#define PIXELPIPES_REGISTER_READER(T, F) static AddModuleInitializer CONCAT(__reader_init_, __COUNTER__)([]() { PipelineReader::register_reader<T>(F); })
#define PIXELPIPES_REGISTER_WRITER(T, F) static AddModuleInitializer CONCAT(__writer_init_, __COUNTER__)([]() { PipelineWriter::register_writer<T>(F); })

    template <typename T>
    T read_t(std::istream &source)
    {
        T v;
        source.read((char *)&v, sizeof(T));
        return v;
    };

    template <typename T>
    void write_t(std::ostream &target, T i)
    {
        target.write((char *)&i, sizeof(T));
    };

    template <>
    inline void write_t(std::ostream &target, bool b)
    {
        unsigned char c = b ? 0xFF : 0;
        target.write((const char *)&c, sizeof(unsigned char));
    };

    template <>
    inline bool read_t(std::istream &source)
    {
        char c;
        source.read(&c, sizeof(unsigned char));
        return c > 0;
    };

    template <>
    inline void write_t(std::ostream &target, std::string s)
    {
        size_t len = s.size();
        write_t(target, len);
        target.write(&s[0], len);
    };

    template <>
    inline std::string read_t(std::istream &source)
    {
        try
        {
            size_t len = read_t<size_t>(source);
            std::string res(len, ' ');
            source.read(&res[0], len);
            return res;
        }
        catch (std::bad_alloc &exception)
        {
            throw SerializationException("Unable to allocate a string");
        }
    };

}
