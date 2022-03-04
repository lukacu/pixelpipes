#include <fstream>
#include <algorithm>

#include <pixelpipes/serialization.hpp>
#include <pixelpipes/geometry.hpp>

namespace pixelpipes
{

    constexpr static std::string_view __stream_header = "@@PPRAW";

    SerializationException::SerializationException(std::string reason) : BaseException(reason) {}

    void write_size(std::ostream &target, size_t i)
    {

        target.write((char *)&i, sizeof(size_t));
    };

    template <typename T>
    T read_t(std::istream &source)
    {
        T v;
        source.read((char *)&v, sizeof(T));
        return v;
    };

    void write_string(std::ostream &target, std::string s)
    {

        size_t len = s.size();

        write_size(target, len);
        target.write(&s[0], len);
    };

    std::string read_string(std::istream &source)
    {

        size_t len = read_t<size_t>(source);
        std::string res(len, ' ');
        source.read(&res[0], len);
        return res;
    };

    template <typename T>
    void write_t(std::ostream &target, T i)
    {

        target.write((char *)&i, sizeof(T));
    };

    PIXELPIPES_REGISTER_WRITER(int, [](SharedVariable v, std::ostream &target)
                               {
    int i = Integer::get_value(v);
    write_t<int>(target, i); });

    PIXELPIPES_REGISTER_WRITER(float, [](SharedVariable v, std::ostream &target)
                               {
    float f = Float::get_value(v);
    target.write((char*)&f, sizeof(float)); });

    PIXELPIPES_REGISTER_WRITER(bool, [](SharedVariable v, std::ostream &target)
                               {
    bool b = Boolean::get_value(v);
    unsigned char c = b ? 0xFF : 0;

    target.write((const char *) &c, 1); });

    PIXELPIPES_REGISTER_WRITER(std::string, [](SharedVariable v, std::ostream &target)
                               {
    std::string s = String::get_value(v);
    write_string(target, s); });

    PIXELPIPES_REGISTER_WRITER(Point2D, [](SharedVariable v, std::ostream &target)
                               {
    Point2D p = Point2DVariable::get_value(v);
    write_t(target, p); });

    PIXELPIPES_REGISTER_WRITER(Point3D, [](SharedVariable v, std::ostream &target)
                               {
    Point3D p = Point3DVariable::get_value(v);
    write_t(target, p); });

    PIXELPIPES_REGISTER_WRITER(View2D, [](SharedVariable v, std::ostream &target)
                               {
    View2D w = View2DVariable::get_value(v);
    write_t(target, w); });

    PIXELPIPES_REGISTER_WRITER(View3D, [](SharedVariable v, std::ostream &target)
                               {
    View3D w = View3DVariable::get_value(v);
    write_t(target, w); });

    PIXELPIPES_REGISTER_READER(int, [](std::istream &source)
                               { return std::make_shared<Integer>(read_t<int>(source)); });

    PIXELPIPES_REGISTER_READER(float, [](std::istream &source)
                               { return std::make_shared<Float>(read_t<float>(source)); });

    PIXELPIPES_REGISTER_READER(bool, [](std::istream &source)
                               {
    char c;
    source.read(&c, 1);
    return std::make_shared<Boolean>(c > 0); });

    PIXELPIPES_REGISTER_READER(std::string, [](std::istream &source)
                               { return std::make_shared<String>(read_string(source)); });

    PIXELPIPES_REGISTER_READER(Point2D, [](std::istream &source)
                               { return std::make_shared<Point2DVariable>(read_t<Point2D>(source)); });

    PIXELPIPES_REGISTER_READER(Point3D, [](std::istream &source)
                               { return std::make_shared<Point3DVariable>(read_t<Point3D>(source)); });

    PIXELPIPES_REGISTER_READER(View2D, [](std::istream &source)
                               { return std::make_shared<View2DVariable>(read_t<View2D>(source)); });

    PIXELPIPES_REGISTER_READER(View3D, [](std::istream &source)
                               { return std::make_shared<View3DVariable>(read_t<View3D>(source)); });

    PipelineWriter::PipelineWriter()
    {
    }

    void PipelineWriter::write(std::ostream &target)
    {

        target << __stream_header;

        // First, write the modules that have to be loaded

        write_size(target, used_modules.size());

        for (auto m : used_modules)
        {
            write_string(target, m->name());
        }

        // Map variable types used to names

        std::map<std::string_view, int> type_mapping;
        int counter = 1;

        for (auto t : used_types)
        {
            type_mapping.insert({std::get<1>(writers().find(t)->second), counter++});
        }

        write_size(target, type_mapping.size());

        for (auto t : type_mapping)
        {
            write_string(target, std::string(t.first));
            write_t(target, t.second);
        }

        // Write the operations with their arguments

        write_size(target, variables.size());

        for (auto var : variables)
        {

            TypeName tn = std::get<1>(writers().find(var->type())->second);

            write_t(target, type_mapping.find(tn)->second);

            VariableWriter writer = std::get<0>(writers().find(var->type())->second);
            writer(var, target);
        }

        // Write the input dependencies for operations (the pipeline stuff)

        write_size(target, operations.size());

        for (auto op : operations)
        {

            write_string(target, std::get<0>(op));

            write_size(target, std::get<1>(op).size());
            for (auto i : std::get<1>(op))
            {
                write_t(target, i);
            }

            write_size(target, std::get<2>(op).size());
            for (auto i : std::get<2>(op))
            {
                write_t(target, i);
            }
        }
    }

    void PipelineWriter::write(std::string &target)
    {

        std::fstream stream(target, std::fstream::out);

        write(stream);

        stream.close();
    }

    int PipelineWriter::append(std::string name, std::vector<SharedVariable> args, std::vector<int> inputs)
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

        std::vector<int> var_indices;

        for (auto arg : args)
        {

            auto argtype = arg->type();

            if (writers().find(argtype) != writers().end())
            {

                SharedModule source = std::get<2>((writers().find(argtype))->second);

                if (source)
                    used_modules.insert(source);

                used_types.insert(argtype);

                // TODO: this could be potentially optimized, remove redundancy
                variables.push_back(arg);

                var_indices.push_back((int)variables.size() - 1);
            }
            else
            {
                throw SerializationException("Writer not found");
            }
        }

        operations.push_back(OperationData(name, var_indices, inputs));

        return operations.size() - 1;
    }

    PipelineWriter::WriterMap &PipelineWriter::writers()
    {
        static PipelineWriter::WriterMap data;
        return data;
    }

    void PipelineWriter::register_writer(TypeIdentifier identifier, std::string_view name, VariableWriter writer)
    {

        DEBUGMSG("Registering writer for type %s (%p)\n", std::string(name).data(), identifier);

        if (writers().find(identifier) != writers().end())
        {

            throw SerializationException("Writer already registered for type");
        }

        for (auto d : writers())
        {

            if (std::get<1>(d.second) == name)
            {

                throw SerializationException("Writer already registered with this type name");
            }
        }

        SharedModule source = Module::context();

        writers().insert(WriterMap::value_type{identifier, WriterData{writer, name, source}});
    }

    PipelineReader::PipelineReader()
    {
    }

    SharedPipeline PipelineReader::read(std::istream &source)
    {

        {
            char buffer[__stream_header.size()];

            source.read(buffer, __stream_header.size());

            if (buffer != __stream_header)
            {
                throw SerializationException("Illegal stream");
            }
        }

        // First, read the modules that have to be loaded and load them

        {
            auto module_count = read_t<size_t>(source);

            for (size_t i = 0; i < module_count; i++)
            {
                auto module_name = read_string(source);
                Module::load(module_name);
            }
        }

        // Read variable type mapping

        std::map<int, TypeIdentifier> type_mapping;

        {

            auto count = read_t<size_t>(source);

            for (size_t i = 0; i < count; i++)
            {
                std::string type_name = read_string(source);
                int code = read_t<int>(source);

                bool found = false;

                for (auto d : readers())
                {
                    if (std::get<1>(d.second) == type_name)
                    {
                        type_mapping.insert({code, d.first});
                        DEBUGMSG("Type mapping: %d -> %p (%s) \n", code, d.first, type_name.c_str())
                        found = true;
                        break;
                    }
                }

                if (!found)
                {
                    throw SerializationException("Reader not found");
                }
            }
        }

        SharedPipeline pipeline = std::make_shared<Pipeline>();

        std::vector<SharedVariable> variables;

        // Read the variables used in operations

        {

            auto count = read_t<size_t>(source);

            for (size_t i = 0; i < count; i++)
            {

                auto code = read_t<int>(source);

                if (type_mapping.find(code) != type_mapping.end())
                {

                    TypeIdentifier type_id = type_mapping.find(code)->second;

                    auto reader = std::get<0>(readers().find(type_id)->second);

                    variables.push_back(reader(source));
                }
                else
                {
                    throw SerializationException("Variable reading error");
                }
            }
        }

        // Read the operations with their arguments, populate pipeline

        {

            auto count = read_t<size_t>(source);

            for (size_t i = 0; i < count; i++)
            {

                std::string name = read_string(source);

                auto argument_count = read_t<size_t>(source);

                std::vector<SharedVariable> arguments;

                for (size_t j = 0; j < argument_count; j++)
                {
                    arguments.push_back(variables[read_t<int>(source)]);
                }

                auto input_count = read_t<size_t>(source);

                std::vector<int> inputs;

                for (size_t j = 0; j < input_count; j++)
                {
                    inputs.push_back(read_t<int>(source));
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

    void PipelineReader::register_reader(TypeIdentifier identifier, std::string_view name, VariableReader reader)
    {

        DEBUGMSG("Registering reader for type %s (%p) \n", std::string(name).data(), identifier);

        if (readers().find(identifier) != readers().end())
        {

            throw SerializationException("Reader already registered for this type");
        }

        for (auto d : readers())
        {

            if (std::get<1>(d.second) == name)
            {

                throw SerializationException("Reader already registered with this type name");
            }
        }

        SharedModule source = Module::context();

        readers().insert(ReaderMap::value_type{identifier, ReaderData{reader, name, source}});
    }

}
