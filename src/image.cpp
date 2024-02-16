

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
        TokenReference image_properties(const TokenList &inputs)
        {
                VERIFY(inputs.size() == 1, "Image properties requires one argument");

                Shape s = inputs[0]->shape();

                VERIFY(s.rank() <= 3, "Image has rank 3 or less");

                TokenReference dtype = create<Placeholder>(IntegerType);

                if (s.element() == IntegerType)
                {
                        dtype = wrap(DataType::Integer);
                }
                else if (s.element() == CharType)
                {
                        dtype = wrap(DataType::Char);
                }
                else if (s.element() == ShortType)
                {
                        dtype = wrap(DataType::Short);
                }
                else if (s.element() == UnsignedShortType)
                {
                        dtype = wrap(DataType::UnsignedShort);
                }
                else if (s.element() == FloatType)
                {
                        dtype = wrap(DataType::Float);
                }
                /*else if (s.element() != AnyType)
                {
                        std::cout << "Unsupported image depth: " << s.element() << " " << BooleanType << std::endl;
                        throw TypeException("Unsupported image depth");
                }*/

                Sequence<TokenReference> data((size_t)4);
                if (s.rank() < 3)
                {
                        data[0] = wrap(s[1]);
                        data[1] = wrap(s[0]);
                        data[2] = wrap(1);
                        data[3] = dtype.reborrow();
                } else {
                        data[0] = wrap(s[2]);
                        data[1] = wrap(s[1]);
                        data[2] = wrap(s[0]);
                        data[3] = dtype.reborrow();
                }

                return create<GenericList>(data);
        }

        PIXELPIPES_UNIT_OPERATION("image_properties", image_properties, image_properties);

}
