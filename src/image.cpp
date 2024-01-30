

#include <cstring>
#include <cmath>

#include <pixelpipes/tensor.hpp>
#include <pixelpipes/serialization.hpp>

namespace pixelpipes
{

    /**
     * @brief Apply view linear transformation to an image.
     *
     */
    TokenReference GetImageProperties(const TensorReference& tensor)
    {
        Shape s = tensor->shape();

        VERIFY(s.rank() <= 3, "Image has rank 3 or less");

        size_t depth = 0;

        if (s.element() == IntegerType) {
                depth = sizeof(int);
        } else if (s.element() == CharType) {
                depth = sizeof(uchar);
        } else if (s.element() == ShortType) {
                depth = sizeof(short);
        } else if (s.element() == UnsignedShortType) {
                depth = sizeof(ushort);
        } else if (s.element() == FloatType) {
                depth = sizeof(float);
        } else {
            throw TypeException("Unknown element type");
        }

        return create<IntegerVector>(make_view(std::vector<int>({(int)s[1], (int)s[0], (int)s[2], (int)depth * 8})));
    }

    PIXELPIPES_UNIT_OPERATION_AUTO("image_properties", GetImageProperties, (constant_shape<int, 4>));

}
