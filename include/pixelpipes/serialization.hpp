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

    class PIXELPIPES_API SerializationException : public BaseException
    {
    public:
        SerializationException(std::string reason);
        SerializationException(const SerializationException &e) = default;
    };

    class PIXELPIPES_API Serializer
    {
    public:
        virtual void write(const TokenReference &, std::ostream &) = 0;

        virtual TokenReference read(std::istream &) = 0;
    };

    typedef void (*TokenWriter)(const TokenReference &, std::ostream &);

    typedef TokenReference (*TokenReader)(std::istream &);

    void PIXELPIPES_API type_register_serializer(TypeIdentifier i, std::string_view name, TokenReader reader, TokenWriter writer);

#define PIXELPIPES_REGISTER_SERIALIZER(T, UID, READER, WRITER) static AddModuleInitializer CONCAT(__type_io_init_, __COUNTER__)([]() { type_register_serializer(T, UID, READER, WRITER); })

    void PIXELPIPES_API write_pipeline(const Pipeline &pipeline, std::ostream &drain, bool compress = true, bool relocatable = true);
    void PIXELPIPES_API write_pipeline(const Pipeline &pipeline, const std::string &drain, bool compress = true, bool relocatable = true);

    Pipeline PIXELPIPES_API read_pipeline(std::istream &source);
    Pipeline PIXELPIPES_API read_pipeline(const std::string &source);

    inline void check_error(std::ios &stream)
    {
        if (stream.eof())
        {
            throw SerializationException("End of file");
        }
        if (stream.fail() || stream.bad())
        {
            throw SerializationException("IO failure");
        }
    }

    template <typename T>
    T read_t(std::istream &source)
    {

        T v;
        source.read((char *)&v, sizeof(T));
        check_error(source);
        return v;
    }

    template <typename T>
    void write_t(std::ostream &drain, T i)
    {
        drain.write((char *)&i, sizeof(T));
        check_error(drain);
    }

    template <>
    inline void write_t(std::ostream &drain, bool b)
    {  
        unsigned char c = b ? 0xFF : 0;
        drain.write((const char *)&c, sizeof(unsigned char));
        check_error(drain);
    }

    template <>
    inline bool read_t(std::istream &source)
    {
        char c;
        source.read(&c, sizeof(unsigned char));
        check_error(source);
        return c > 0;
    }

    template <>
    inline void write_t(std::ostream &drain, std::string s)
    {
        size_t len = s.size();
        write_t(drain, len);
        drain.write(&s[0], len);
        check_error(drain);
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
            check_error(source);
            return res;
        }
        catch (std::bad_alloc const &)
        {
            throw SerializationException(Formatter() << "Unable to allocate a string of length " << len);
        }
    }

    template <typename T>
    inline void write_sequence(std::ostream &drain, const Span<T> &list)
    {
        write_t(drain, list.size());
        for (size_t i = 0; i < list.size(); i++)
        {
            write_t(drain, list[i]);
        }
    }

    template <typename T>
    inline Sequence<T> read_sequence(std::istream &source)
    {
        try
        {
            size_t len = read_t<size_t>(source);
            std::vector<T> list;
            if (len)
                list.resize(len);
            for (size_t i = 0; i < len; i++)
            {
                list[i] = read_t<T>(source);
            }
            return list;
        }
        catch (std::bad_alloc &exception)
        {
            throw SerializationException("Unable to allocate an array");
        }
    }

}
