#include <fstream>
#include <algorithm>
#include <filesystem>

#include <pixelpipes/serialization.hpp>
#include <pixelpipes/geometry.hpp>

#include "compression.hpp"

#include "../debug.h"

namespace pixelpipes
{

    constexpr static std::string_view __stream_header_raw = "@@PPRAW";
    constexpr static std::string_view __stream_header_compressed = "@@PPCMP";

    SerializationException::SerializationException(std::string reason) : BaseException(reason) {}

    template <typename T>
    void write_v(std::ostream &source, const std::vector<T> &list)
    {
        write_t(source, list.size());
        for (size_t i = 0; i < list.size(); i++)
        {
            write_t(source, list[i]);
        }
    }

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
    }

    PIXELPIPES_REGISTER_WRITER(IntegerIdentifier, [](SharedToken v, std::ostream &target)
                               {
    int i = Integer::get_value(v);
    write_t<int>(target, i); });

    PIXELPIPES_REGISTER_WRITER(FloatIdentifier, [](SharedToken v, std::ostream &target)
                               {
    float f = Float::get_value(v);
    target.write((char*)&f, sizeof(float)); });

    PIXELPIPES_REGISTER_WRITER(BooleanIdentifier, [](SharedToken v, std::ostream &target)
                               {
    bool b = Boolean::get_value(v);
    unsigned char c = b ? 0xFF : 0;
    target.write((const char *) &c, 1); });

    PIXELPIPES_REGISTER_WRITER(StringIdentifier, [](SharedToken v, std::ostream &target)
                               {
    std::string s = String::get_value(v);
    write_t(target, s); });

    PIXELPIPES_REGISTER_WRITER(IntegerListIdentifier, [](SharedToken v, std::ostream &target)
                               { write_v(target, extract<std::vector<int>>(v)); });
    PIXELPIPES_REGISTER_WRITER(FloatListIdentifier, [](SharedToken v, std::ostream &target)
                               { write_v(target, extract<std::vector<float>>(v)); });
    PIXELPIPES_REGISTER_WRITER(BooleanListIdentifier, [](SharedToken v, std::ostream &target)
                               { write_v(target, extract<std::vector<bool>>(v)); });
    PIXELPIPES_REGISTER_WRITER(StringListIdentifier, [](SharedToken v, std::ostream &target)
                               { write_v(target, extract<std::vector<std::string>>(v)); });

    PIXELPIPES_REGISTER_WRITER(Point2DIdentifier, [](SharedToken v, std::ostream &target)
                               {
    Point2D p = Point2DVariable::get_value(v);
    write_t(target, p); });

    PIXELPIPES_REGISTER_WRITER(Point3DIdentifier, [](SharedToken v, std::ostream &target)
                               {
    Point3D p = Point3DVariable::get_value(v);
    write_t(target, p); });

    PIXELPIPES_REGISTER_WRITER(View2DIdentifier, [](SharedToken v, std::ostream &target)
                               {
    View2D w = View2DVariable::get_value(v);
    write_t(target, w); });

    PIXELPIPES_REGISTER_WRITER(View3DIdentifier, [](SharedToken v, std::ostream &target)
                               {
    View3D w = View3DVariable::get_value(v);
    write_t(target, w); });

    PIXELPIPES_REGISTER_READER(IntegerIdentifier, [](std::istream &source) -> SharedToken
                               { return std::make_shared<Integer>(read_t<int>(source)); });
    PIXELPIPES_REGISTER_READER(FloatIdentifier, [](std::istream &source) -> SharedToken
                               { return std::make_shared<Float>(read_t<float>(source)); });
    PIXELPIPES_REGISTER_READER(BooleanIdentifier, [](std::istream &source) -> SharedToken
                               { return std::make_shared<Boolean>(read_t<unsigned char>(source) != 0); });
    PIXELPIPES_REGISTER_READER(StringIdentifier, [](std::istream &source) -> SharedToken
                               { return std::make_shared<String>(read_t<std::string>(source)); });

    PIXELPIPES_REGISTER_READER(IntegerListIdentifier, [](std::istream &source) -> SharedToken
                               { return wrap(read_v<int>(source)); });
    PIXELPIPES_REGISTER_READER(FloatListIdentifier, [](std::istream &source) -> SharedToken
                               { return wrap(read_v<float>(source)); });
    PIXELPIPES_REGISTER_READER(BooleanListIdentifier, [](std::istream &source) -> SharedToken
                               { return wrap(read_v<bool>(source)); });
    PIXELPIPES_REGISTER_READER(StringListIdentifier, [](std::istream &source) -> SharedToken
                               { return wrap(read_v<std::string>(source)); });

    PIXELPIPES_REGISTER_READER(Point2DIdentifier, [](std::istream &source) -> SharedToken
                               { return std::make_shared<Point2DVariable>(read_t<Point2D>(source)); });

    PIXELPIPES_REGISTER_READER(Point3DIdentifier, [](std::istream &source) -> SharedToken
                               { return std::make_shared<Point3DVariable>(read_t<Point3D>(source)); });

    PIXELPIPES_REGISTER_READER(View2DIdentifier, [](std::istream &source) -> SharedToken
                               { return std::make_shared<View2DVariable>(read_t<View2D>(source)); });

    PIXELPIPES_REGISTER_READER(View3DIdentifier, [](std::istream &source) -> SharedToken
                               { return std::make_shared<View3DVariable>(read_t<View3D>(source)); });

    class PrefixList : public List
    {
    public:
        PrefixList(Span<std::string> list, std::string prefix = std::string()) : list(list.begin(), list.end()), prefix(prefix)
        {
            if (list.empty())
                throw TypeException("List is empty");
        }

        ~PrefixList() = default;

        virtual size_t size() const
        {
            return list.size();
        }

        virtual TypeIdentifier element_type_id() const
        {
            return StringIdentifier;
        }

        virtual TypeIdentifier type_id() const
        {
            return GetTypeIdentifier<PrefixList>();
        }

        virtual SharedToken get(size_t index) const
        {

            if (index >= list.size())
            {
                throw TypeException("Index out of range");
            }

            return wrap(prefix + list[index]);
        }

    private:
        std::vector<std::string> list;

        std::string prefix;
    };

    class FileList : public Operation
    {
    public:
        FileList(Sequence<std::string> list) : value(std::make_shared<StringList>(list))
        {
        }

        virtual SharedToken run(TokenList inputs)
        {
            UNUSED(inputs);
            return value;
        }

        virtual TypeIdentifier type()
        {
            return GetTypeIdentifier<FileList>();
        }

    private:
        SharedToken value;
    };

    REGISTER_OPERATION("file_list", FileList, Sequence<std::string>);

    SharedList make_relative(SharedList files, std::filesystem::path origin)
    {

        std::vector<std::string> relative;

        for (size_t i = 0; i < files->size(); i++)
        {
            auto path = std::filesystem::absolute(extract<std::string>(files->get(i)));

            if (!origin.empty())
            {
                relative.push_back(std::filesystem::relative(path, origin));
            }
            else
            {
                relative.push_back(path);
            }
        }

        return std::make_shared<StringList>(make_span(relative));
    }

    SharedList make_absolute(SharedList files, std::filesystem::path origin)
    {

        std::vector<std::string> absolute;

        for (size_t i = 0; i < files->size(); i++)
        {

            auto path = std::filesystem::path(extract<std::string>(files->get(i)));

            if (path.is_relative())
            {
                absolute.push_back((origin / path).lexically_normal());
            }
            else
            {
                absolute.push_back(path.lexically_normal());
            }
        }

        return std::make_shared<StringList>(make_span(absolute));
    }

    PipelineWriter::PipelineWriter(bool compress, bool relocatable) : compress(compress), relocatable(relocatable)
    {
    }

    void PipelineWriter::write(std::ostream &target)
    {

        if (relocatable)
        {
            origin = std::filesystem::current_path();
        }
        else
        {
            origin = "";
        }

        write_pipeline(target);
    }

    void PipelineWriter::write(std::string &target)
    {

        std::fstream stream(target, std::fstream::out);

        if (relocatable)
        {
            origin = std::filesystem::absolute(target).parent_path();
        }
        else
        {
            origin = "";
        }

        write_pipeline(stream);

        stream.close();
    }

    void PipelineWriter::write_pipeline(std::ostream &target)
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

        std::map<std::string, int> type_mapping;
        int counter = 1;

        for (auto t : used_types)
        {
            type_mapping.insert({type_name(t), counter++});
        }

        write_t(target, type_mapping.size());

        for (auto t : type_mapping)
        {
            DEBUGMSG("Type mapping: %d -> %s \n", t.second, t.first.c_str())
            write_t(target, t.first);
            write_t(target, t.second);
        }

        // Write the operations with their arguments

        write_t(target, tokens.size());

        for (auto t : tokens)
        {

            auto token = std::get<0>(t);

            TypeName tn = type_name(token->type_id());

            write_t(target, type_mapping.find(tn)->second);

            TokenWriter writer = std::get<0>(writers().find(token->type_id())->second);

            // We have marked filename lists during appending
            if (std::get<1>(t))
            {
                token = make_relative(List::get_list(token, StringIdentifier), origin);
            }

            writer(token, target);
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

    int PipelineWriter::append(std::string name, TokenList args, Span<int> inputs)
    {

        SharedOperation op = create_operation(name, args);

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
            bool filename = false;

            if (op->type() == GetTypeIdentifier<FileList>() && List::is_list(arg, StringIdentifier))
            {
                filename = true;
            }

            auto argtype = arg->type_id();
            {

                SharedModule source = type_source(argtype);
                if (source)
                    used_modules.insert(source);

                used_types.insert(argtype);

                // Also look for inner types of lists
                if (List::is(arg))
                {

                    SharedModule source2 = type_source(List::cast(arg)->element_type_id());

                    if (source2)
                        used_modules.insert(source2);

                    used_types.insert(List::cast(arg)->element_type_id());
                }
            }

            if (writers().find(argtype) != writers().end())
            {

                SharedModule source = std::get<1>((writers().find(argtype))->second);

                if (source)
                    used_modules.insert(source);

                // TODO: this could be potentially optimized, remove redundancy
                tokens.push_back({arg, filename});

                token_indices.push_back((int)tokens.size() - 1);
            }
            else
            {
                throw SerializationException(Formatter() << "Writer " << name << " not found");
            }
        }

        operations.push_back(OperationData(name, token_indices, std::vector<int>(inputs.begin(), inputs.end())));

        return (int)(operations.size() - 1);
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

    Pipeline PipelineReader::read(std::string &target)
    {

        std::fstream stream(target, std::fstream::in);

        origin = std::filesystem::absolute(target).parent_path();

        Pipeline pipeline;

        read_stream(stream, pipeline);

        stream.close();

        return pipeline;
    }

    Pipeline PipelineReader::read(std::istream &source)
    {

        origin = std::filesystem::current_path().string();

        Pipeline pipeline;

        read_stream(source, pipeline);

        return pipeline;
    }

    void PipelineReader::read_stream(std::istream &source, Pipeline &pipeline)
    {

        if (check_header(source, __stream_header_compressed))
        {
            DEBUGMSG("Compressed stream\n");
            InputCompressionStream cs(source);
            read_data(cs, pipeline);
        }
        else if (check_header(source, __stream_header_raw))
        {
            DEBUGMSG("Raw stream\n");
            read_data(source, pipeline);
        }
        else
        {
            throw SerializationException("Illegal stream");
        }

        pipeline.finalize();
    }

    void PipelineReader::read_data(std::istream &source, Pipeline &pipeline)
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
                catch (TypeException)
                {
                    throw SerializationException(Formatter() << "Type " << type_name << " not found");
                }
            }
        }

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
                    if (t >= (int)tokens.size())
                        throw SerializationException("Illegal token index");

                    if (name == "file_list" && List::is_list(tokens[t], StringIdentifier))
                    { // TODO: this should be done without strings

                        tokens[t] = make_absolute(List::get_list(tokens[t]), origin);
                    }

                    arguments.push_back(tokens[t]);
                }

                pipeline.append(name, make_span(arguments), make_span(inputs));
            }
        }
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
