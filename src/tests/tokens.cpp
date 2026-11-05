#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

#include <pixelpipes/serialization.hpp>
#include <pixelpipes/tensor.hpp>

using namespace pixelpipes;

namespace {

void expect(bool condition, const char *message)
{
    if (!condition)
        throw std::runtime_error(message);
}

void test_shapes_and_scalars()
{
    const auto matrix = MatrixType(IntegerType, 3, 4);
    expect(matrix.element() == IntegerType, "matrix element type mismatch");
    expect(matrix.rank() == 2, "matrix rank mismatch");
    expect(matrix[0] == 3 && matrix[1] == 4, "matrix dimensions mismatch");
    expect(matrix.is_fixed() && !matrix.is_scalar(), "matrix shape flags mismatch");

    const auto scalar = ScalarType<float>();
    expect(scalar.element() == FloatType, "scalar element type mismatch");
    expect(scalar.is_scalar() && scalar.is_fixed(), "scalar shape flags mismatch");

    auto value = create<IntegerScalar>(42);
    expect(extract<int>(value) == 42, "scalar value mismatch");
    expect(value->shape().element() == IntegerType, "token element type mismatch");
}

void test_generic_list()
{
    auto values = create<GenericList>(std::initializer_list<TokenReference>{wrap(7), wrap(11), wrap(13)});

    expect(values->length() == 3, "list length mismatch");
    expect(extract<int>(values->get(0)) == 7, "first list value mismatch");
    expect(extract<int>(values->get(1)) == 11, "second list value mismatch");
    expect(extract<int>(values->get(2)) == 13, "third list value mismatch");
}

void test_primitive_serialization()
{
    std::stringstream stream;
    write_t(stream, 123456);
    write_t(stream, true);
    write_t(stream, std::string("pixelpipes"));

    expect(read_t<int>(stream) == 123456, "integer serialization mismatch");
    expect(read_t<bool>(stream), "boolean serialization mismatch");
    expect(read_t<std::string>(stream) == "pixelpipes", "string serialization mismatch");

    bool truncated = false;
    try
    {
        std::stringstream empty;
        (void)read_t<int>(empty);
    }
    catch (const SerializationException &)
    {
        truncated = true;
    }
    expect(truncated, "truncated input did not raise SerializationException");
}

Sequence<TokenReference> run_pipeline(Pipeline &pipeline)
{
    pipeline.finalize();
    return pipeline.run(1);
}

void test_pipeline_serialization()
{
    Pipeline pipeline;
    Sequence<TokenReference> constant_arguments({wrap(42)});
    Sequence<int> no_inputs;
    const int constant = pipeline.append("constant", constant_arguments, make_span(no_inputs));

    Sequence<TokenReference> output_arguments({wrap(std::string("answer"))});
    Sequence<int> output_inputs({constant});
    pipeline.append("output", output_arguments, make_span(output_inputs));
    pipeline.metadata().set("suite", "cpp");

    auto result = run_pipeline(pipeline);
    expect(result.size() == 1, "pipeline output count mismatch");
    expect(extract<int>(result[0]) == 42, "pipeline output value mismatch");

    std::stringstream stream;
    write_pipeline(pipeline, stream, false, false);
    stream.seekg(0);

    auto restored = read_pipeline(stream);
    expect(restored.metadata().get("suite") == "cpp", "pipeline metadata mismatch");
    auto restored_result = restored.run(1);
    expect(restored_result.size() == 1, "restored output count mismatch");
    expect(extract<int>(restored_result[0]) == 42, "restored output value mismatch");
}

} // namespace

int main()
{
    try
    {
        test_shapes_and_scalars();
        test_generic_list();
        test_primitive_serialization();
        test_pipeline_serialization();
    }
    catch (const BaseException &error)
    {
        std::cerr << "PixelPipes C++ test failed: " << error.what() << std::endl;
        return 1;
    }
    catch (const std::exception &error)
    {
        std::cerr << "PixelPipes C++ test failed: " << error.what() << std::endl;
        return 1;
    }

    return 0;
}
