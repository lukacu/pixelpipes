#include <fstream>
#include <algorithm>

#include <pixelpipes/serialization.hpp>
#include <pixelpipes/geometry.hpp>

#include "compression.hpp"

#include "../debug.h"

namespace pixelpipes
{

    constexpr static std::string_view __stream_header_raw = "@@PPRAW";
    constexpr static std::string_view __stream_header_compressed = "@@PPCMP";

    SerializationException::SerializationException(std::string reason) : BaseException(reason) {}

    template <>
    void write_t(std::ostream &target, DNF s)
    {
        write_t(target, s.size());
        for (auto x : s)
        {
            write_t(target, x.size());
            for (auto d : x)
                write_t<bool>(target, d); // Explicit type needed
        }
    };

    template <>
    DNF read_t(std::istream &source)
    {
        size_t len = read_t<size_t>(source);
        std::vector<std::vector<bool>> data;
        data.reserve(len);

        for (size_t i = 0; i < len; i++)
        {
            std::vector<bool> clause;
            size_t n = read_t<size_t>(source);
            for (size_t j = 0; j < n; j++)
                clause.push_back(read_t<bool>(source));
            data.push_back(clause);
        }
        return DNF(data);
    };

    template <typename T>
    void write_v(std::ostream &source, const std::vector<T> &list)
    {
        write_t(source, list.size());
        for (size_t i = 0; i < list.size(); i++)
        {
            write_t(source, list[i]);
        }
    };

    template <typename T>
    std::vector<T> read_v(std::istream &source)
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
    };

    PIXELPIPES_REGISTER_WRITER(IntegerType, [](SharedToken v, std::ostream &target)
                               {
    int i = Integer::get_value(v);
    write_t<int>(target, i); });

    PIXELPIPES_REGISTER_WRITER(FloatType, [](SharedToken v, std::ostream &target)
                               {
    float f = Float::get_value(v);
    target.write((char*)&f, sizeof(float)); });

    PIXELPIPES_REGISTER_WRITER(BooleanType, [](SharedToken v, std::ostream &target)
                               {
    bool b = Boolean::get_value(v);
    unsigned char c = b ? 0xFF : 0;
    target.write((const char *) &c, 1); });

    PIXELPIPES_REGISTER_WRITER(StringType, [](SharedToken v, std::ostream &target)
                               {
    std::string s = String::get_value(v);
    write_t(target, s); });

    PIXELPIPES_REGISTER_WRITER(IntegerListType, [](SharedToken v, std::ostream &target)
                               { write_v(target, List::get_list(v, IntegerType)->elements<int>()); });
    PIXELPIPES_REGISTER_WRITER(FloatListType, [](SharedToken v, std::ostream &target)
                               { write_v(target, List::get_list(v, FloatType)->elements<float>()); });
    PIXELPIPES_REGISTER_WRITER(BooleanListType, [](SharedToken v, std::ostream &target)
                               { write_v(target, List::get_list(v, BooleanType)->elements<bool>()); });
    PIXELPIPES_REGISTER_WRITER(StringListType, [](SharedToken v, std::ostream &target)
                               { write_v(target, List::get_list(v, StringType)->elements<std::string>()); });

    PIXELPIPES_REGISTER_WRITER(DNFType, [](SharedToken v, std::ostream &target)
                               { write_t(target, ContainerToken<DNF>::get_value(v)); });

    PIXELPIPES_REGISTER_WRITER(Point2DType, [](SharedToken v, std::ostream &target)
                               {
    Point2D p = Point2DVariable::get_value(v);
    write_t(target, p); });

    PIXELPIPES_REGISTER_WRITER(Point3DType, [](SharedToken v, std::ostream &target)
                               {
    Point3D p = Point3DVariable::get_value(v);
    write_t(target, p); });

    PIXELPIPES_REGISTER_WRITER(View2DType, [](SharedToken v, std::ostream &target)
                               {
    View2D w = View2DVariable::get_value(v);
    write_t(target, w); });

    PIXELPIPES_REGISTER_WRITER(View3DType, [](SharedToken v, std::ostream &target)
                               {
    View3D w = View3DVariable::get_value(v);
    write_t(target, w); });

    PIXELPIPES_REGISTER_READER(IntegerType, [](std::istream &source)
                               { return std::make_shared<Integer>(read_t<int>(source)); });
    PIXELPIPES_REGISTER_READER(FloatType, [](std::istream &source)
                               { return std::make_shared<Float>(read_t<float>(source)); });
    PIXELPIPES_REGISTER_READER(BooleanType, [](std::istream &source)
                               { return std::make_shared<Boolean>(read_t<unsigned char>(source) != 0); });
    PIXELPIPES_REGISTER_READER(StringType, [](std::istream &source)
                               { return std::make_shared<String>(read_t<std::string>(source)); });

    PIXELPIPES_REGISTER_READER(IntegerListType, [](std::istream &source)
                               { return std::make_shared<IntegerList>(read_v<int>(source)); });
    PIXELPIPES_REGISTER_READER(FloatListType, [](std::istream &source)
                               { return std::make_shared<FloatList>(read_v<float>(source)); });
    PIXELPIPES_REGISTER_READER(BooleanListType, [](std::istream &source)
                               { return std::make_shared<BooleanList>(read_v<bool>(source)); });
    PIXELPIPES_REGISTER_READER(StringListType, [](std::istream &source)
                               { return std::make_shared<StringList>(read_v<std::string>(source)); });

    PIXELPIPES_REGISTER_READER(DNFType, [](std::istream &source)
                               { return wrap(read_t<DNF>(source)); });

    PIXELPIPES_REGISTER_READER(Point2DType, [](std::istream &source)
                               { return std::make_shared<Point2DVariable>(read_t<Point2D>(source)); });

    PIXELPIPES_REGISTER_READER(Point3DType, [](std::istream &source)
                               { return std::make_shared<Point3DVariable>(read_t<Point3D>(source)); });

    PIXELPIPES_REGISTER_READER(View2DType, [](std::istream &source)
                               { return std::make_shared<View2DVariable>(read_t<View2D>(source)); });

    PIXELPIPES_REGISTER_READER(View3DType, [](std::istream &source)
                               { return std::make_shared<View3DVariable>(read_t<View3D>(source)); });

    PipelineWriter::PipelineWriter()
    {
    }

    void PipelineWriter::write(std::ostream &target, bool compress)
    {

        if (compress)
        {

            target << __stream_header_compressed;

            OutputCompressionStream cs(target);
            write_data(cs);
        }
        else
        {

            target << __stream_header_raw;

            write_data(target);
        }
    }

    void PipelineWriter::write_data(std::ostream &target)
    {

        // First, write the modules that have to be loaded

        write_t(target, used_modules.size());

        for (auto m : used_modules)
        {
            write_t(target, m->name());
        }

        // Map variable types used to names

        std::map<std::string_view, int> type_mapping;
        int counter = 1;

        for (auto t : used_types)
        {
            type_mapping.insert({type_name(t), counter++});
        }

        write_t(target, type_mapping.size());

        for (auto t : type_mapping)
        {
            write_t(target, std::string(t.first));
            write_t(target, t.second);
        }

        // Write the operations with their arguments

        write_t(target, tokens.size());

        for (auto t : tokens)
        {

            TypeName tn = type_name(t->type_id());

            write_t(target, type_mapping.find(tn)->second);

            TokenWriter writer = std::get<0>(writers().find(t->type_id())->second);
            writer(t, target);
        }

        // Write the input dependencies for operations (the pipeline stuff)

        write_t(target, operations.size());

        for (auto op : operations)
        {

            // Write operation name
            write_t(target, std::get<0>(op));

            // Write token indices
            write_v(target, std::get<1>(op));

            // Write inputs
            write_v(target, std::get<2>(op));

        }
    }

    void PipelineWriter::write(std::string &target, bool compress)
    {

        std::fstream stream(target, std::fstream::out);

        write(stream, compress);

        stream.close();
    }

    int PipelineWriter::append(std::string name, std::vector<SharedToken> args, std::vector<int> inputs)
    {

        create_operation(name, args);

        OperationDescription meta = describe_operation(name);

        if (meta.source)
            used_modules.insert(meta.source);

        for (int i : inputs)
        {
            if (i >= (int)operations.size() || i < 0)
                return -1;
        }

        std::vector<int> token_indices;

        for (auto arg : args)
        {

            auto argtype = arg->type_id();
            {

                SharedModule source = type_source(argtype);
                if (source)
                    used_modules.insert(source);

                used_types.insert(argtype);

                // Also look for inner types of lists
                if (List::is(arg))
                {

                    SharedModule source = type_source(List::cast(arg)->element_type_id());

                    if (source)
                        used_modules.insert(source);

                    used_types.insert(List::cast(arg)->element_type_id());
                }
            }

            if (writers().find(argtype) != writers().end())
            {

                SharedModule source = std::get<1>((writers().find(argtype))->second);

                if (source)
                    used_modules.insert(source);

                // TODO: this could be potentially optimized, remove redundancy
                tokens.push_back(arg);

                token_indices.push_back((int)tokens.size() - 1);
            }
            else
            {
                throw SerializationException(Formatter() << "Writer " << name << " not found");
            }
        }

        operations.push_back(OperationData(name, token_indices, inputs));

        return operations.size() - 1;
    }

    PipelineWriter::WriterMap &PipelineWriter::writers()
    {
        static PipelineWriter::WriterMap data;
        return data;
    }

    void PipelineWriter::register_writer(TypeIdentifier identifier, TokenWriter writer)
    {

        if (writers().find(identifier) != writers().end())
        {

            throw SerializationException(Formatter() << "Writer already registered for type " << type_name(identifier));
        }

        DEBUGMSG("Registering writer for type %s (%ld)\n", type_name(identifier).data(), identifier);

        SharedModule source = Module::context();

        writers().insert(WriterMap::value_type{identifier, WriterData{writer, source}});
    }

    PipelineReader::PipelineReader()
    {
    }

    bool check_header(std::istream &source, std::string_view header)
    {

        std::vector<char> buffer(header.size());

        auto pos = source.tellg();

        source.read(&buffer[0], header.size());

        for (size_t i = 0; i < header.size(); i++)
        {
            if (buffer[i] != header[i])
            {
                source.seekg(pos);
                return false;
            }
        }

        return true;
    }

    SharedPipeline PipelineReader::read(std::istream &source)
    {

        if (check_header(source, __stream_header_compressed))
        {
            DEBUGMSG("Compressed stream\n");
            InputCompressionStream cs(source);
            return read_data(cs);
        }
        else if (check_header(source, __stream_header_raw))
        {
            DEBUGMSG("Raw stream\n");
            return read_data(source);
        }
        else
        {
            throw SerializationException("Illegal stream");
        }
    }

    SharedPipeline PipelineReader::read_data(std::istream &source)
    {

        // First, read the modules that have to be loaded and load them

        {
            auto module_count = read_t<size_t>(source);

            for (size_t i = 0; i < module_count; i++)
            {
                std::string module_name = read_t<std::string>(source);
                DEBUGMSG("Required module %s \n", module_name.c_str());
                Module::load(module_name);
            }
        }

        // Read token type mapping

        std::map<int, TypeIdentifier> type_mapping;

        {

            auto count = read_t<size_t>(source);

            for (size_t i = 0; i < count; i++)
            {
                std::string type_name = read_t<std::string>(source);
                int code = read_t<int>(source);

                try
                {

                    auto d = readers().find(type_find(type_name));

                    if (d != readers().end())
                    {
                        type_mapping.insert({code, d->first});
                        DEBUGMSG("Type mapping: %d -> %ld (%s) \n", code, d->first, type_name.c_str())
                    }
                    else
                    {
                        throw SerializationException(Formatter() << "Reader " << type_name << " not found");
                    }
                }
                catch (TypeException &e)
                {
                    throw SerializationException(Formatter() << "Type " << type_name << " not found");
                }
            }
        }

        SharedPipeline pipeline = std::make_shared<Pipeline>();

        std::vector<SharedToken> tokens;

        // Read the variables used in operations

        {

            auto count = read_t<size_t>(source);

            DEBUGMSG("Reading %ld tokens\n", count);

            for (size_t i = 0; i < count; i++)
            {

                auto code = read_t<int>(source);

                if (type_mapping.find(code) != type_mapping.end())
                {
                    TypeIdentifier type_id = type_mapping.find(code)->second;

                    auto reader = std::get<0>(readers().find(type_id)->second);

                    tokens.push_back(reader(source));
                }
                else
                {
                    throw SerializationException("Token reading error");
                }
            }
        }

        // Read the operations with their arguments, populate pipeline

        {

            auto count = read_t<size_t>(source);

            DEBUGMSG("Reconstructing pipeline with %ld operations\n", count);

            for (size_t i = 0; i < count; i++)
            {

                std::string name = read_t<std::string>(source);

                std::vector<SharedToken> arguments;

                std::vector<int> token_indices = read_v<int>(source);

                std::vector<int> inputs = read_v<int>(source);

                for (auto t : token_indices)
                {
                    if (t >= tokens.size())
                        throw SerializationException("Illegal token index");
                    arguments.push_back(tokens[t]);
                }

                pipeline->append(name, arguments, inputs);
            }
        }

        pipeline->finalize();

        return pipeline;
    }

    SharedPipeline PipelineReader::read(std::string &target)
    {

        std::fstream stream(target, std::fstream::in);

        SharedPipeline pipeline = read(stream);

        stream.close();

        return pipeline;
    }

    PipelineReader::ReaderMap &PipelineReader::readers()
    {
        static ReaderMap data;
        return data;
    }

    void PipelineReader::register_reader(TypeIdentifier identifier, TokenReader reader)
    {

        if (readers().find(identifier) != readers().end())
        {

            throw SerializationException("Reader already registered for this type");
        }

        DEBUGMSG("Registering reader for type %s (%ld) \n", type_name(identifier).data(), identifier);

        SharedModule source = Module::context();

        readers().insert(ReaderMap::value_type{identifier, ReaderData{reader, source}});
    }

}
