

#include <pixelpipes/image.hpp>

namespace pixelpipes {

    size_t multiply_shape(detail::any_container<size_t> dims) {

        if (dims->size() == 0) return 0;

        size_t count = 1;

        for (std::vector<size_t>::const_iterator it = dims->cbegin(); it != dims->cend(); it++) {
            count *= *it;
        }

        return count;
    }

    ImageVariable::ImageVariable(Image value) : ContainerVariable<Image>(value) {
        
    }

    void ImageVariable::describe(std::ostream& os) const {
        if (value->ndims() == 2) {
            os << "[Image: width " << value->shape(1) << " height " << value->shape(0) << "]";
        }
        else if (value->ndims() == 3) {
            os << "[Image: width " << value->shape(1) << " height " << value->shape(0) << " channels" << value->shape(2) << "]";
        }
        else {
            os << "[Image: " << value->ndims() << " dimensions ]";
        }
    }

    BufferImage::BufferImage(detail::any_container<size_t> dims, ImageDepth depth): data_depth(depth), dimensions(dims->begin(), dims->end()) {

        size_t buffer_size = multiply_shape(dimensions) * ((size_t) data_depth >> 8);

        buffer = new unsigned char[buffer_size];

    }

    BufferImage::~BufferImage() {

        delete[] buffer;

    }

    ImageDepth BufferImage::depth() const {

        return data_depth;

    }

    size_t BufferImage::shape(size_t i) const {

        return dimensions[i];

    }

    const std::vector<size_t> BufferImage::shape() const {
        
        return dimensions;

    }

    size_t BufferImage::stride(size_t i) const {

        if (i == dimensions.size() - 1) return ((size_t) data_depth >> 8);

        return dimensions[i-1] * ((size_t) data_depth >> 8);

    }

    size_t BufferImage::ndims() const {

        return dimensions.size();

    }

    TypeIdentifier BufferImage::backend() const {

        return GetTypeIdentifier<BufferImage>();

    }

    ImageDataIterator& BufferImage::begin() {


    }

    ImageDataIterator& BufferImage::end() {

    }

}
