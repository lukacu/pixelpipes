
#include <memory>
#include <algorithm>
#include <limits>

#include <xtensor/xtensor.hpp>
#include <xtensor/xtensor_simd.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xnoalias.hpp>

#include <pixelpipes/tensor.hpp>

namespace pixelpipes
{

    PIXELPIPES_REGISTER_ENUM("datatype", DataType);

    struct TensorGuard
    {
        TensorReference guard;
    };

    TensorView::TensorView(const TensorReference &source, const Sizes &shape) : TensorView(source, 0, shape, generate_strides(shape, source->cell_size()))
    {
        VERIFY(size() == source->size(), "Tensor element count mismatch");
    }

    TensorView::TensorView(const TensorReference &source, size_t offset, const Sizes &shape, const Sizes &strides)
    {
        // TODO: can we verify shape somehow?
        // TODO: what if tensor is already a view?
        // VERIFY(!source->is<TensorView>(), "Unable to view existing views");

        VERIFY(shape.size() == strides.size(), "Size mismatch");

        _shape = SizeSequence(shape);
        _strides = SizeSequence(strides);

        _cell_size = source->cell_size();
        _cell_type = source->datatype();

        _size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>()) * _cell_size;

        _data = ByteSpan(source->data().data() + offset, source->data().size() - offset);

        _owner = new TensorGuard{source.reborrow()};
        _cleanup = ([](void *v) -> void
                    { delete static_cast<TensorGuard *>(v); });
    }

    TensorView::~TensorView()
    {
        if (_owner)
        {
            _cleanup(_owner);
        }
    }

    Shape TensorView::shape() const
    {
        return Shape(datatype(), _shape);
    }

    size_t TensorView::length() const
    {
        return _shape[0];
    }

    size_t TensorView::cell_size() const
    {
        return _cell_size;
    }

    Type TensorView::datatype() const
    {
        return _cell_type;
    }

    void TensorView::describe(std::ostream &os) const
    {
        os << "[Tensor view]";
    }

    size_t TensorView::size() const
    {
        return _size;
    }

    TokenReference TensorView::get(const Sizes &index) const
    {
        size_t o = get_offset(index);
        return get_scalar(o);
    }

    TokenReference TensorView::get(size_t i) const
    {
        if (_shape.size() == 1)
        {
            return get_scalar(i * cell_size());
        }
        else
        {

            std::vector<size_t> index(_shape.size(), 0);
            index[0] = i;
            size_t offset = get_offset(make_span(index));

            auto ref = pixelpipes::cast<Tensor>(reference());

            return create<TensorView>(ref, offset, make_view(_shape, 1), make_view(_strides, 1));
        }
        return empty();
    }

    ReadonlySliceIterator TensorView::read_slices() const
    {
        return ReadonlySliceIterator(const_data(), _shape, _strides, cell_size());
    }

    WriteableSliceIterator TensorView::write_slices()
    {
        return WriteableSliceIterator(data(), _shape, _strides, cell_size());
    }

    ByteView TensorView::const_data() const
    {
        return ByteView(_data.data(), _data.size());
    }

    SizeSequence TensorView::strides() const
    {
        return _strides;
    }

    ByteSpan TensorView::data()
    {
        return ByteSpan(_data.data(), _data.size());
    }

    template <>
    Sequence<TensorReference> extract(const TokenReference &v)
    {
        VERIFY((bool)v, "Uninitialized variable");

        if (v->is<List>())
        {
            return (extract<ListReference>(v))->elements<TensorReference>();
        }
        else if (v->is<TensorReference>())
        {
            return Sequence<TensorReference>({extract<TensorReference>(v)});
        }

        throw TypeException("Unable to convert to list of images");
    }


    /**
     * @brief Converts data type of tensor
     *
     */
    /*
    TokenReference convert(const TensorReference &tensor, ImageDepth depth) noexcept(false)
    {
        double maxin = maximum_value(image);
        int dtype = -1;
        double maxout = 1;

        switch (depth)
        {
        case ImageDepth::Char:
            dtype = CV_8U;
            maxout = std::numeric_limits<uchar>::max();
            break;
        case ImageDepth::Short:
            dtype = CV_16S;
            maxout = std::numeric_limits<short>::max();
            break;
        case ImageDepth::UShort:
            dtype = CV_16U;
            maxout = std::numeric_limits<ushort>::max();
            break;
        case ImageDepth::Integer:
            dtype = CV_32S;
            maxout = std::numeric_limits<int>::max();
            break;
        case ImageDepth::Float:
            dtype = CV_32F;
            maxout = 1;
            break;
        }

        if (image.depth() == dtype)
        {
            // No conversion required
            return image;
        }

        cv::Mat result;
        image.convertTo(result, dtype, maxout / maxin);

        return result;
    }

    PIXELPIPES_OPERATION_AUTO("convert", convert);
*/
}