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

    class PipelineWriter;

    class PipelineReader;

    typedef void (* TokenWriter) (SharedToken, std::ostream &);

    typedef SharedToken (* TokenReader)(std::istream &);

    class PIXELPIPES_API SerializationException : public BaseException
    {
    public:
        SerializationException(std::string reason);
    };

    class PIXELPIPES_API PipelineWriter
    {
    public:
        PipelineWriter(bool compress = true, bool relocatable = false);

        ~PipelineWriter() = default;

        void write(std::ostream &target);

        void write(std::string &target);

        int append(std::string name, TokenList args, Span<int> inputs);

        static void register_writer(TypeIdentifier identifier, TokenWriter writer);

    private:
        typedef std::tuple<std::string, std::vector<int>, std::vector<int>> OperationData;
        typedef std::tuple<TokenWriter, SharedModule> WriterData;
        typedef std::map<TypeIdentifier, WriterData> WriterMap;
        typedef std::tuple<SharedToken, bool> TokenData;

        static WriterMap &writers();

        std::set<SharedModule> used_modules;
        std::set<TypeIdentifier> used_types;

        std::vector<TokenData> tokens;
        std::vector<OperationData> operations;

        std::string origin;
        bool compress;
        bool relocatable;

        void write_pipeline(std::ostream &target);

        void write_data(std::ostream &target);
    };

    class PIXELPIPES_API PipelineReader
    {
    public:
        PipelineReader();

        ~PipelineReader() = default;

        Pipeline read(std::istream &target);

        Pipeline read(std::string &target);

        static void register_reader(TypeIdentifier identifier, TokenReader reader);

    private:
        typedef std::tuple<TokenReader, SharedModule> ReaderData;
        typedef std::map<TypeIdentifier, ReaderData> ReaderMap;

        static ReaderMap &readers();

        void read_stream(std::istream &source, Pipeline& pipeline);

        void read_data(std::istream &source, Pipeline& pipeline);

        std::string origin;
    };

#define PIXELPIPES_REGISTER_READER(T, F) static AddModuleInitializer CONCAT(__reader_init_, __COUNTER__)([]() { PipelineReader::register_reader(T, F); })
#define PIXELPIPES_REGISTER_WRITER(T, F) static AddModuleInitializer CONCAT(__writer_init_, __COUNTER__)([]() { PipelineWriter::register_writer(T, F); })

    template <typename T>
    T read_t(std::istream &source)
    {
        T v;
        source.read((char *)&v, sizeof(T));
        return v;
    }

    template <typename T>
    void write_t(std::ostream &target, T i)
    {
        target.write((char *)&i, sizeof(T));
    }

    template <>
    inline void write_t(std::ostream &target, bool b)
    {
        unsigned char c = b ? 0xFF : 0;
        target.write((const char *)&c, sizeof(unsigned char));
    }

    template <>
    inline bool read_t(std::istream &source)
    {
        char c;
        source.read(&c, sizeof(unsigned char));
        return c > 0;
    }

    template <>
    inline void write_t(std::ostream &target, std::string s)
    {
        size_t len = s.size();
        write_t(target, len);
        target.write(&s[0], len);
    }

    template <>
    inline std::string read_t(std::istream &source)
    {
        size_t len;
        try
        {
            len = read_t<size_t>(source);
            std::string res(len, ' ');
            source.read(&res[0], len);
            return res;
        }
        catch (std::bad_alloc const&)
        {
            throw SerializationException(Formatter() << "Unable to allocate a string of length " << len);
        }
    }

}
