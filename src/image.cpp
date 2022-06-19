

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

        VERIFY(s.dimensions() <= 3, "Image has rank 3 or less");

        size_t depth = 0;

        if (s.element() == IntegerIdentifier) {
                depth = sizeof(int);
        } else if (s.element() == CharIdentifier) {
                depth = sizeof(char);
        } else if (s.element() == ShortIdentifier) {
                depth = sizeof(short);
        } else if (s.element() == FloatIdentifier) {
                depth = sizeof(float);
        } else {
            throw TypeException("Unknown element type");
        }

        return create<IntegerVector>(make_view(std::vector<int>({(int)s[1], (int)s[0], (int)s[2], (int)depth * 8})));
    }

    PIXELPIPES_OPERATION_AUTO("image_properties", GetImageProperties);

}
