
#include <pixelpipes/operation.hpp>
#include <pixelpipes/image.hpp>

#include <lodepng.h>

using namespace pixelpipes;
using namespace lodepng;

SharedToken ImageReadPngPalette(TokenList inputs) noexcept(false)
{

    VERIFY(inputs.size() == 1, "Incorrect number of parameters");

    std::string filename = extract<std::string>(inputs[0]);
    std::vector<unsigned char> png;
    unsigned width, height;

    unsigned error = lodepng::load_file(png, filename);
    VERIFY(!error, lodepng_error_text(error));

    auto image = new std::vector<unsigned char>();
    error = lodepng::decode(image[0], width, height, png, LCT_PALETTE, 8);

    if (error) delete image;

    VERIFY(!error, lodepng_error_text(error));

    return std::make_shared<BufferImage>(width, height, 1, ImageDepth::Byte, &(*image)[0], [image](){ delete image; });
}

REGISTER_OPERATION_FUNCTION("load_png_palette", ImageReadPngPalette);
