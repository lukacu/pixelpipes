
#include <pixelpipes/operation.hpp>
#include <pixelpipes/image.hpp>

#include <lodepng.h>

using namespace pixelpipes;
using namespace lodepng;

namespace pixelpipes
{

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

        if (error)
            delete image;

        VERIFY(!error, lodepng_error_text(error));

        return std::make_shared<BufferImage>(width, height, 1, ImageDepth::Byte, &(*image)[0], [image]()
                                             { delete image; });
    }

    REGISTER_OPERATION_FUNCTION("load_png_palette", ImageReadPngPalette);
    /*
    PIXELPIPES_OPERATION("load_png_palette",
            ImageReadPngPalette,
            "Load a PNG image as palette indices",
            types::Image(undefined, undefined, 1),
            input<types::Char(undefined), std::string, "PNG file name">);

    input<type, native, description>
    parameter<native, description> = def

    type<char>
    type<byte, none, none>
    type<

    */

}