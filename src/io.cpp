
#include <pixelpipes/operation.hpp>
#include <pixelpipes/tensor.hpp>

#include <lodepng.h>

using namespace pixelpipes;
using namespace lodepng;

ByteSequence decode_png_wrapper(unsigned& w, unsigned& h, const unsigned char* in,
                size_t insize, LodePNGColorType colortype, unsigned bitdepth) {
  unsigned char* buffer = 0;
  unsigned error = lodepng_decode_memory(&buffer, &w, &h, in, insize, colortype, bitdepth);
  if(buffer && !error) {
    State state;
    state.info_raw.colortype = colortype;
    state.info_raw.bitdepth = bitdepth;
    size_t buffersize = lodepng_get_raw_size(w, h, &state.info_raw);
    return ByteSequence::claim(buffer, buffersize);
    //out.insert(out.end(), &buffer[0], &buffer[buffersize]);
  } else {
      free(buffer);
  }
  throw TypeException(lodepng_error_text(error));
}

namespace pixelpipes
{
/*
    TokenReference read_file(std::string filename) noexcept(false)
    {



    }*/

    TokenReference load_png_palette(std::string filename) noexcept(false)
    {
        std::vector<unsigned char> png;
        unsigned width, height;

        unsigned error = lodepng::load_file(png, filename);
        VERIFY(!error, lodepng_error_text(error));

        auto data = decode_png_wrapper(width, height, png.data(), png.size(), LCT_PALETTE, 8);

        return create<Matrix<char>>(width, height, data.reinterpret<char>());
    }

    PIXELPIPES_OPERATION_AUTO("load_png_palette", load_png_palette);

}