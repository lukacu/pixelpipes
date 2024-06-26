#include <fstream>
#include <algorithm>
#include <filesystem>

#include <pixelpipes/serialization.hpp>
#include <pixelpipes/tensor.hpp>

#include "compression.hpp"

#include "../debug.h"

namespace pixelpipes
{

    template <typename T>
    struct pointer_comparator
    {
        bool operator()(const Pointer<T> &a, const Pointer<T> &b) const noexcept
        {
            return (a.get() < b.get());
        }
    };

    constexpr static std::string_view __stream_header_raw = "@@PPRAW";
    constexpr static std::string_view __stream_header_compressed = "@@PPCMP";

    SerializationException::SerializationException(std::string reason) : BaseException(reason) {}

    typedef std::tuple<std::string, TokenReader, TokenWriter, ModuleReference> SerializatorData;
    typedef std::map<Type, SerializatorData> SerializatorMap;

    static SerializatorMap &handlers()
    {
        static SerializatorMap data;
        return data;
    }

    inline char type_to_char(Type t)
    {
        if (t == GetType<int>())
        {
            return 'i';
        }
        else if (t == GetType<float>())
        {
            return 'f';
        }
        else if (t == GetType<char>())
        {
            return 'c';
        }
        else if (t == GetType<bool>())
        {
            return 'b';
        }
        else if (t == GetType<short>())
        {
            return 's';
        }
        else if (t == GetType<ushort>())
        {
            return 'S';
        }
        else if (t == GetType<uchar>())
        {
            return 'C';
        }
        // Custom data type
        return 'x';
    }

    inline Type char_to_type(char c)
    {
        if (c == 'i')
        {
            return GetType<int>();
        }
        else if (c == 'f')
        {
            return GetType<float>();
        }
        else if (c == 'c')
        {
            return GetType<char>();
        }
        else if (c == 'C')
        {
            return GetType<uchar>();
        }
        else if (c == 'b')
        {
            return GetType<bool>();
        }
        else if (c == 's')
        {
            return GetType<short>();
        }
        else if (c == 'S')
        {
            return GetType<ushort>();
        }
        return AnyType;
    }

    void write_metadata(std::ostream &drain, const Metadata &data)
    {
        write_t(drain, data.size());
        for (auto key : data.keys())
        {
            write_t(drain, key);
            write_t(drain, data.get(key));
        }
    }

    Metadata read_metadata(std::istream &source, Metadata& data)
    {
        try
        {
            size_t len = read_t<size_t>(source);
            for (size_t i = 0; i < len; i++)
            {
                auto key = read_t<std::string>(source);
                auto value = read_t<std::string>(source);
                data.set(key, value);
            }
            return data;
        }
        catch (std::bad_alloc &exception)
        {
            throw SerializationException("Unable to allocate metadata object");
        }
    }

    void write_shape(const Shape &s, std::ostream &drain)
    {
        VERIFY(s.is_fixed(), "Unable to serialize non-fixed size");
        VERIFY(type_size(s.element()) > 0, "Unable to encode type identifier");

        write_t(drain, type_to_char(s.element()));

        write_t(drain, (unsigned char)s.rank());
        for (auto x : s)
        {
            write_t(drain, (size_t)x);
        }
    }

    void write_tensor(const TokenReference &v, std::ostream &drain)
    {
        TensorReference t = extract<TensorReference>(v);
        Shape s = t->shape();
        write_shape(t->shape(), drain);
  
        size_t total = 0;
        for (ReadonlySliceIterator it = t->read_slices(); (bool)(*it); it++)
        {
            drain.write((char *)(*it).data(), (*it).size());
            total += (*it).size();
			check_error(drain);
        }

        VERIFY(t->size() == total, "Tensor array size mismatch");
    }

    Shape read_shape(std::istream &source)
    {
        char element = read_t<char>(source);
        size_t dimensions = read_t<unsigned char>(source);

        std::vector<size_t> shape;
        shape.resize(dimensions);
        for (size_t i = 0; i < dimensions; i++)
        {
            shape[i] = read_t<size_t>(source);
        }

        return Shape(char_to_type(element), make_view(shape));
    }

    TokenReference read_tensor(std::istream &source)
    {

        Shape s = read_shape(source);

        size_t c = std::accumulate(s.begin(), s.end(), 1, std::multiplies<size_t>()) * type_size(s.element());

        TensorReference tensor = create_tensor(s);

        VERIFY(tensor->size() == c, "Tensor read error");

        source.read((char *)tensor->data().data(), c);

		check_error(source);

        return tensor;
    }

    TokenReference read_token(std::istream &source)
    {

        auto prefix = read_t<char>(source);

        if (prefix == 'X')
        {
            auto name = read_t<std::string>(source);

            for (auto x = handlers().begin(); x != handlers().end(); x++)
            {

                if (name == std::get<0>(x->second))
                {
                    auto reader = std::get<1>(x->second);
                    return reader(source);
                }
            }
        }
        if (prefix == 'T')
        {
            return read_tensor(source);
        }
        else if (prefix == 'L')
        {
            size_t len = read_t<size_t>(source);

            Sequence<TokenReference> tokens(len);

            for (size_t i = 0; i < len; i++)
            {
                tokens[i] = read_token(source);
            }

            return create<GenericList>(tokens);
        }
        else 
        {
            switch (char_to_type(prefix))
            {
            case IntegerType:
                return create<IntegerScalar>(read_t<int>(source));
            case CharType:
                return create<CharScalar>(read_t<uchar>(source));
            case ShortType:
                return create<ShortScalar>(read_t<short>(source));
            case UnsignedShortType:
                return create<ShortScalar>(read_t<ushort>(source));
            case FloatType:
                return create<FloatScalar>(read_t<float>(source));
            case BooleanType:
                return create<BooleanScalar>(read_t<bool>(source));
            }
        }

        throw SerializationException("Token reading error");
    }

    void write_token(const TokenReference &token, std::ostream &drain)
    {

        try
        {

            if (handlers().find(token->typeId()) != handlers().end())
            {
                TokenWriter writer = std::get<2>(handlers().find(token->typeId())->second);

                write_t(drain, 'X');

                write_t(drain, std::get<0>(handlers().find(token->typeId())->second));

                writer(token, drain);
                return;
            }

            if (token->is<Tensor>())
            {
                // This should capture most vectors, matrices ...
                write_t(drain, 'T');
                write_tensor(token, drain);
                return;
            }

            if (token->is<List>())
            {
                // Custom lists are handled here ...
                write_t(drain, 'L');
                write_t(drain, token->cast<List>()->length());

                for (size_t i = 0; i < token->cast<List>()->length(); i++)
                {
                    write_token(token->cast<List>()->get(i), drain);
                }
                return;
            }

            // Scalars are written here
            Shape shape = token->shape();
            if (shape.is_scalar())
            {
                write_t(drain, type_to_char(shape.element()));

                if (shape.element() == IntegerType)
                {
                    write_t(drain, extract<int>(token));
                    return;
                }
                else if (shape.element() == CharType)
                {
                    write_t(drain, extract<char>(token));
                    return;
                }
                else if (shape.element() == ShortType)
                {
                    write_t(drain, extract<short>(token));
                    return;
                }
                else if (shape.element() == UnsignedShortType)
                {
                    write_t(drain, extract<ushort>(token));
                    return;
                }
                else if (shape.element() == BooleanType)
                {
                    write_t(drain, extract<bool>(token));
                    return;
                }
                else if (shape.element() == FloatType)
                {
                    write_t(drain, extract<float>(token));
                    return;
                }
            }
        }
        catch (std::exception &e)
        {
            throw SerializationException((Formatter() << "Token encoding error: " << e.what()).str());
        }

        throw SerializationException((Formatter() << "Only tensors, lists, primitive scalars, or types with defined handlers allowed, got " << token).str());
    }

    class FileList : public Operation
    {
    public:
        FileList(Sequence<std::string> list) : value(create<StringList>(list))
        {
        }

        virtual TokenReference run(const TokenList &inputs)
        {
            UNUSED(inputs);
            return value.reborrow();
        }

        virtual Type type() const override
        {
            return GetType<FileList>();
        }

        virtual void describe(std::ostream &os) const
        {
            StringListReference list = extract<StringListReference>(value);

            os << "[FileList: " << list->length() << " files]";
        }

        virtual OperationTrait trait() const override
        {
            return OperationTrait::Stateful;
        }

        virtual TokenReference evaluate(const TokenList &inputs) override
        {
            UNUSED(inputs);
            return value.reborrow();
        }

        virtual Sequence<TokenReference> serialize() { return Sequence<TokenReference>({value.reborrow()}); }

    private:
        TokenReference value;
    };

    PIXELPIPES_OPERATION_CLASS("file_list", FileList, Sequence<std::string>);

    ListReference make_relative(ListReference files, std::filesystem::path origin)
    {

        std::vector<std::string> relative;

        for (size_t i = 0; i < files->length(); i++)
        {
            auto path = std::filesystem::absolute(extract<std::string>(files->get(i)));

            if (!origin.empty())
            {
                relative.push_back(std::filesystem::relative(path, origin).string());
            }
            else
            {
                relative.push_back(path.string());
            }
        }

        return create<StringList>(make_span(relative));
    }

    ListReference make_absolute(ListReference files, std::filesystem::path origin)
    {

        std::vector<std::string> absolute;

        for (size_t i = 0; i < files->length(); i++)
        {

            auto path = std::filesystem::path(extract<std::string>(files->get(i)));

            if (path.is_relative())
            {
                absolute.push_back((origin / path).lexically_normal().string());
            }
            else
            {
                absolute.push_back(path.lexically_normal().string());
            }
        }

        return create<StringList>(make_span(absolute));
    }

    void type_register_serializer(Type identifier, std::string_view uid, TokenReader reader, TokenWriter writer)
    {

        if (handlers().find(identifier) != handlers().end())
        {
            throw SerializationException("Serializer already registered for this type");
        }

        DEBUGMSG("Registering handler for type %s (%x) \n", uid.data(), identifier);

        handlers().insert(SerializatorMap::value_type{identifier, SerializatorData{uid, reader, writer, Module::context().reborrow()}});
    }

    struct PipelineData
    {

        typedef std::tuple<std::string, Sequence<int>, Sequence<int>, Metadata> OperationData;
        typedef std::tuple<TokenReference, bool> TokenData;

        std::vector<TokenData> tokens;
        std::vector<OperationData> operations;

        std::set<ModuleReference, pointer_comparator<Module>> used_modules;
        std::set<Type> used_types;

        PipelineData(const Pipeline &pipeline, const std::string &origin)
        {

            for (size_t o = 0; o < pipeline.size(); o++)
            {

                Pipeline::OperationData op = pipeline.get(o);

                auto name = operation_name(op.operation);

                ModuleReference source = operation_source(name);

                // Add operation source module dependency
                if (source)
                    used_modules.insert(source.reborrow());

                std::vector<int> token_indices;

                auto args = op.operation->serialize();

                for (auto arg = args.begin(); arg != args.end(); arg++)
                {
                    bool filename = false;

                    TokenReference token = arg->reborrow();

                    if (op.operation->type() == GetType<FileList>() && (*arg)->is<StringList>())
                    {
                        token = make_relative(extract<ListReference>(*arg), origin);
                    }

                    auto argtype = (*arg)->typeId();

                    if (handlers().find(argtype) != handlers().end())
                    {
                        // If custom serializer exists, we add its source module to the dependency list
                        ModuleReference source = std::get<3>((handlers().find(argtype))->second).reborrow();
                        if (source)
                            used_modules.insert(source.reborrow());
                    }

                    // TODO: this could be potentially optimized, remove redundancy
                    tokens.push_back({token.reborrow(), filename});

                    token_indices.push_back((int)tokens.size() - 1);
                }

                operations.push_back(OperationData(name, Sequence<int>(token_indices), Sequence<int>(op.inputs), op.metadata));
            }
        }
    };

    void _write_data(const Pipeline &pipeline, std::ostream &drain, std::string origin, bool relocatable)
    {

        if (origin.empty())
        {

            if (relocatable)
            {
                origin = std::filesystem::current_path().string();
            }
            else
            {
                origin = "";
            }
        }

        PipelineData data(pipeline, origin);

        // First, write the modules that have to be loaded

        write_t(drain, data.used_modules.size());

        for (auto m = data.used_modules.begin(); m != data.used_modules.end(); m++)
        {
            write_t(drain, (*m)->name());
        }

        write_metadata(drain, pipeline.metadata());

        // Write the operations with their arguments

        write_t(drain, data.tokens.size());

        for (auto t = data.tokens.begin(); t != data.tokens.end(); t++)
        {

            write_token(std::get<0>(*t), drain);
        }

        // Write the input dependencies for operations (the pipeline stuff)

        write_t(drain, data.operations.size());

        for (auto op : data.operations)
        {
            // Write operation name
            write_t(drain, std::get<0>(op));

            // Write token indices
            write_sequence(drain, std::get<1>(op));

            // Write inputs
            write_sequence(drain, std::get<2>(op));

            write_metadata(drain, std::get<3>(op));

        }
    }

    void _write_stream(const Pipeline &pipeline, std::ostream &drain, const std::string origin, bool compress, bool relocatable)
    {

        if (compress)
        {

            drain << __stream_header_compressed;

            OutputCompressionStream cs(drain);
            _write_data(pipeline, cs, origin, relocatable);
        }
        else
        {

            drain << __stream_header_raw;

            _write_data(pipeline, drain, origin, relocatable);
        }
    }

    void write_pipeline(const Pipeline &pipeline, std::ostream &drain, bool compress, bool relocatable)
    {

        std::string origin = std::filesystem::current_path().string();
        _write_stream(pipeline, drain, origin, compress, relocatable);
    }

    void write_pipeline(const Pipeline &pipeline, const std::string &drain, bool compress, bool relocatable)
    {

        std::fstream stream(drain, std::fstream::out  | std::ios_base::binary);

        std::string origin = (relocatable) ? std::filesystem::absolute(drain).parent_path().string() : "";

        _write_stream(pipeline, stream, origin, compress, relocatable);

        stream.flush();
        stream.close();
    }

    bool _check_header(std::istream &source, std::string_view header)
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

    void _read_data(std::istream &source, Pipeline &pipeline, const std::string &origin)
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

        std::vector<TokenReference> tokens;

        read_metadata(source, pipeline.metadata());

        // Read the variables used in operations
        {

            auto count = read_t<size_t>(source);

            DEBUGMSG("Reading %ld tokens\n", count);

            for (size_t i = 0; i < count; i++)
            {
                tokens.push_back(read_token(source));
            }
        }

        // Read the operations with their arguments, populate pipeline

        {

            auto count = read_t<size_t>(source);

            DEBUGMSG("Reconstructing pipeline with %ld operations\n", count);

            for (size_t i = 0; i < count; i++)
            {

                std::string name = read_t<std::string>(source);

                try {

                    std::vector<TokenReference> arguments;

                    Sequence<int> token_indices = read_sequence<int>(source);

                    Sequence<int> inputs = read_sequence<int>(source);

                    Metadata metadata;
                    read_metadata(source, metadata);

                    for (auto t : token_indices)
                    {
                        if (t >= (int)tokens.size())
                            throw SerializationException("Illegal token index");

                        if (name == "file_list" && tokens[t]->is<List>())
                        { // TODO: this should be done without string identifier?
                            tokens[t] = make_absolute(extract<ListReference>(tokens[t]), origin);
                        }

                        arguments.push_back(tokens[t].reborrow());
                    }

                    pipeline.append(name, make_span(arguments), make_span(inputs), metadata);

                } catch (TypeException &e) {
                    throw SerializationException((Formatter() << "Operation deserialization error (" << i << "/" << count << ", " << name << "): "  << e.what()).str());
                }
            }
        }
    }

    void _read_stream(std::istream &source, Pipeline &pipeline, const std::string &origin)
    {

        if (_check_header(source, __stream_header_compressed))
        {
            DEBUGMSG("Compressed stream\n");
            InputCompressionStream cs(source);
            _read_data(cs, pipeline, origin);
        }
        else if (_check_header(source, __stream_header_raw))
        {
            DEBUGMSG("Raw stream\n");
            _read_data(source, pipeline, origin);
        }
        else
        {
            throw SerializationException("Illegal stream");
        }

        pipeline.finalize();
    }

    Pipeline read_pipeline(std::istream &source)
    {
        std::string origin = std::filesystem::current_path().string();

        Pipeline pipeline;

        _read_stream(source, pipeline, origin);

        return pipeline;
    }

    Pipeline read_pipeline(const std::string &drain)
    {

        std::fstream stream(drain, std::fstream::in | std::ios_base::binary);

        std::string origin = std::filesystem::absolute(drain).parent_path().string();

        Pipeline pipeline;

        _read_stream(stream, pipeline, origin);

        stream.flush();

        stream.close();

        return pipeline;
    }

}
