
#include <fstream>

#include <pixelpipes/operation.hpp>
#include <pixelpipes/tensor.hpp>

#include <lodepng.h>

using namespace pixelpipes;
using namespace lodepng;

ByteSequence decode_png_wrapper(unsigned &w, unsigned &h, const unsigned char *in,
                                size_t insize, LodePNGColorType colortype, unsigned bitdepth)
{
    unsigned char *buffer = 0;
    unsigned error = lodepng_decode_memory(&buffer, &w, &h, in, insize, colortype, bitdepth);
    if (buffer && !error)
    {
        State state;
        state.info_raw.colortype = colortype;
        state.info_raw.bitdepth = bitdepth;
        size_t buffersize = lodepng_get_raw_size(w, h, &state.info_raw);
        return ByteSequence::claim(buffer, buffersize);
    }
    else
    {
        free(buffer);
    }
    throw TypeException(lodepng_error_text(error));
}

namespace pixelpipes
{

    TokenReference read_file(std::string filename) noexcept(false)
    {

        std::ifstream handle(filename, std::ios::binary | std::ios::ate);
        std::streamsize size = handle.tellg();

        VERIFY(size >= 0, Formatter() << "Unable to open file " << filename << " for reading");

        handle.seekg(0, std::ios::beg);

        auto buffer = create<FlatBuffer>(size);

        VERIFY( (bool) handle.read((char *) buffer->data().data(), size), "Unable to read file");

        return buffer;
    }

    PIXELPIPES_ACCESS_OPERATION_AUTO("read_file", read_file, (constant_shape<uchar, unknown>) );

    TokenReference load_png_palette(const BufferReference& buffer) noexcept(false)
    {
        unsigned width, height;

        auto data = decode_png_wrapper(width, height, buffer->data().data(), buffer->size(), LCT_PALETTE, 8);

        return create<Matrix<char>>(width, height, data.reinterpret<char>());
    }

    PIXELPIPES_COMPUTE_OPERATION_AUTO("load_png_palette", load_png_palette, (constant_shape<uchar, unknown, unknown>) );

}