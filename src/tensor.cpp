
#include <memory>
#include <algorithm>
#include <limits>

#include <pixelpipes/tensor.hpp>

namespace pixelpipes
{

    TensorView::TensorView(const TensorReference &source, size_t offset, const Sizes &shape, const Sizes &strides) : _data(source.reborrow())
    {

      //  size_t s = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>()) * source->cell_size();


//        VERIFY(source->size() == s, "Data length does not match its shape");

        _shape = SizeSequence(shape);
        _strides = SizeSequence(strides);
        _offset = offset;
    }

    Shape TensorView::shape() const
    {
        return Shape(_data->cell_type(), _shape);
    }

    size_t TensorView::length() const
    {
        return _shape[0];
    }

    size_t TensorView::cell_size() const
    {
        return _data->cell_size();
    }

    TypeIdentifier TensorView::cell_type() const
    {
        return _data->cell_type();
    }

    void TensorView::describe(std::ostream &os) const
    {
        os << "[Tensor view]";
    }

    size_t TensorView::size() const
    {
        return _data->size();
    }

    TokenReference TensorView::get(const Sizes &index) const
    {
        UNUSED(index);
        /* switch (_data->cell_type()) {
             case CharIdentifier:
                 return create<CharScalar>();
         }*/
        return empty();
    }

    TokenReference TensorView::get(size_t i) const
    {
        UNUSED(i);
        /*
                if (_shape.size() == 1)
                {
                    return create<ScalarToken<T>>(_data.at<T>(i * sizeof(T)));
                }
                else
                {

                    std::array<size_t, N> index;
                    index.fill(0);
                    index[0] = i;
                    size_t o1 = get_offset(make_span(index));
                    index[0] = i + 1;
                    size_t o2 = get_offset(make_span(index));

                    auto data = Span<T>(reinterpret_cast<const T *>(std::data(_data) + o1), (o2 - o1) / sizeof(T));

                    if (_shape.size() == 1)
                    {
                        return create<Vector<T>>(data);
                    }
                    else if (N == 3)
                    {
                        return create<Matrix<T>>(_shape[1], _shape[2], data);
                    }
                    else
                    {
                        auto shape = make_span(_shape, (size_t)1);
                        return create<ArrayTensor<T, N - 1>>(shape, data);
                    }
                }*/
        return empty();
    }

    ReadonlySliceIterator TensorView::read_slices() const
    {
        return ReadonlySliceIterator(_data->const_data(), _shape, _strides, _data->cell_size());
    }

    WriteableSliceIterator TensorView::write_slices()
    {
        return WriteableSliceIterator(_data->data(), _shape, _strides, _data->cell_size());
    }

    const uchar *TensorView::const_data() const
    {
        return _data->const_data();
    }

    SizeSequence TensorView::strides() const
    {
        return _strides;
    }

    uchar *TensorView::data()
    {
        return _data->data();
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

    template <typename A, typename B>
    struct nonsaturate_cast
    {
        inline B operator()(const A val)
        {
            return static_cast<B>(val);
        }
    };

    template <typename A, typename B>
    struct saturate_cast
    {
        inline B operator()(const A val)
        {
            return static_cast<B>(std::clamp<A>(val, std::numeric_limits<B>::min(), std::numeric_limits<B>::max()));
        }
    };

    template <typename T, typename Op, typename C>
    inline void _execute_slice(const uchar *b0, const uchar *b1, uchar *b2, size_t length)
    {
        Op op;
        C cast;
        length /= sizeof(T);

        const T *c0 = (T *)b0;
        const T *c1 = (T *)b1;
        T *c2 = (T *)b2;

        for (size_t i = 0; i < length; i++)
        {
            c2[i] = cast(op(c0[i], c1[i]));
        }
    }

    template <typename T, typename Op, typename C>
    inline void _execute_tensor_tensor(ReadonlySliceIterator &it0, ReadonlySliceIterator &it1, WriteableSliceIterator &it2)
    {
        size_t offset0 = 0;
        size_t offset1 = 0;
        size_t offset2 = 0;

        while (true)
        {
            size_t len0 = (*it0).size() - offset0;
            size_t len1 = (*it1).size() - offset1;
            size_t len2 = (*it2).size() - offset2;
            size_t length = (std::min)((std::min)(len0, len1), len2);

            if (length == 0)
                break;

            _execute_slice<T, Op, C>((*it0).data() + offset0, (*it1).data() + offset1, (uchar *)((*it2).data() + offset2), length);

            offset0 += length;
            offset1 += length;
            offset2 += length;

            if (len0 == length)
            {
                offset0 = 0;
                it0++;
            }

            if (len1 == length)
            {
                offset1 = 0;
                it1++;
            }

            if (len2 == length)
            {
                offset2 = 0;
                it2++;
            }
        }
    }

    template <typename T, typename Op, typename C>
    inline void _execute_tensor_scalar(ReadonlySliceIterator &it0, T it1, WriteableSliceIterator &it2)
    {
        size_t offset0 = 0;
        size_t offset2 = 0;

        Op op;
        C cast;

        while (true)
        {
            size_t len0 = (*it0).size() - offset0;
            size_t len2 = (*it2).size() - offset2;
            size_t length = (std::min)(len0, len2);

            if (length == 0)
                break;

            size_t blength = length / sizeof(T);

            const T *c0 = (T *)(*it0).data() + offset0;
            T *c2 = (T *)((*it2).data() + offset2);

            for (size_t i = 0; i < blength; i++)
            {
                c2[i] = cast(op(c0[i], it1));
            }

            offset0 += length;
            offset2 += length;

            if (len0 == length)
            {
                offset0 = 0;
                it0++;
            }

            if (len2 == length)
            {
                offset2 = 0;
                it2++;
            }
        }
    }

    template <typename T, typename Op, typename C>
    inline void _execute_scalar_tensor(T it0, ReadonlySliceIterator &it1, WriteableSliceIterator &it2)
    {
        size_t offset1 = 0;
        size_t offset2 = 0;

        Op op;
        C cast;

        while (true)
        {
            size_t len1 = (*it1).size() - offset1;
            size_t len2 = (*it2).size() - offset2;
            size_t length = (std::min)(len1, len2);

            if (length == 0)
                break;

            size_t blength = length / sizeof(T);

            const T *c1 = (T *)(*it1).data() + offset1;
            T *c2 = (T *)((*it2).data() + offset2);

            for (size_t i = 0; i < blength; i++)
            {
                c2[i] = cast(op(it0, c1[i]));
            }

            offset1 += length;
            offset2 += length;

            if (len1 == length)
            {
                offset1 = 0;
                it1++;
            }

            if (len2 == length)
            {
                offset2 = 0;
                it2++;
            }
        }
    }

    inline SizeSequence _broadcasting_strides(const Shape& original, const SizeSequence& desired, const SizeSequence& strides) 
    {
        auto bstrides = SizeSequence::repeat(desired.size(), (size_t) original[original.dimensions() - 1]);

        for (size_t i = 0; i < original.dimensions(); i++) 
        {
            if (desired[i] != (size_t)original[i]) {
                bstrides[i] = 0;
            } else {
                bstrides[i] = strides[i];
            }


        }

        return bstrides;

    }

    template <template <typename> class T, template <typename, typename> class C>
    TokenReference tensor_elementwise_binary(const TokenReference &t0, const TokenReference &t1)
    {
        if (t0->is<Tensor>() && t1->is<Tensor>())
        {

            TensorReference tr0 = extract<TensorReference>(t0);
            TensorReference tr1 = extract<TensorReference>(t1);

            Shape s0 = tr0->shape();
            Shape s1 = tr1->shape();

            VERIFY(s0.element() == s1.element(), "Tensor type mismatch");
            // VERIFY(s0.dimensions() == s1.dimensions(), "Tensor dimensions mismatch");

            size_t outdim = std::max(s0.dimensions(), s1.dimensions());
            SizeSequence outsize(outdim);

            for (size_t i = 0; i < outdim; i++)
            {
                if (s0[i] == 1)
                {
                    outsize[i] = s1[i];
                }
                else if (s1[i] == 1)
                {
                    outsize[i] = s0[i];
                }
                else if (s0[i] == s1[i])
                {
                    outsize[i] = s0[i];
                }
                else
                    throw TypeException("Tensor dimension mismatch");
            }

            auto strides0 = _broadcasting_strides(s0, outsize, tr0->strides());
            auto strides1 = _broadcasting_strides(s1, outsize, tr1->strides());

            ReadonlySliceIterator it0 = TensorView(tr0, 0, outsize, strides0).read_slices();
            ReadonlySliceIterator it1 = TensorView(tr1, 0, outsize, strides1).read_slices();

            if (s0.element() == CharIdentifier)
            {
                TensorReference result = create_tensor<char>(s0);
                WriteableSliceIterator it2 = result->write_slices();
                _execute_tensor_tensor<char, T<int>, C<int, char>>(it0, it1, it2);
                return result;
            }
            else if (s0.element() == ShortIdentifier)
            {
                TensorReference result = create_tensor<short>(s0);
                WriteableSliceIterator it2 = result->write_slices();
                _execute_tensor_tensor<short, T<int>, C<int, short>>(it0, it1, it2);
                return result;
            }
            else if (s0.element() == IntegerIdentifier)
            {
                TensorReference result = create_tensor<int>(s0);
                WriteableSliceIterator it2 = result->write_slices();
                _execute_tensor_tensor<int, T<int>, C<int, int>>(it0, it1, it2);
                return result;
            }
            else if (s0.element() == FloatIdentifier)
            {
                TensorReference result = create_tensor<float>(s0);
                WriteableSliceIterator it2 = result->write_slices();
                _execute_tensor_tensor<float, T<float>, nonsaturate_cast<float, float>>(it0, it1, it2);
                return result;
            }
            else
            {
                throw TypeException("Unsupported tensor type");
            }
        }

        if (t0->is<Tensor>())
        {

            TensorReference tr0 = extract<TensorReference>(t0);
            Shape s0 = tr0->shape();

            ReadonlySliceIterator it0 = tr0->read_slices();

            if (s0.element() == CharIdentifier)
            {
                TensorReference result = create_tensor<char>(s0);
                WriteableSliceIterator it2 = result->write_slices();
                _execute_tensor_scalar<char, T<int>, C<int, char>>(it0, extract<int>(t1), it2);
                return result;
            }
            else if (s0.element() == ShortIdentifier)
            {
                TensorReference result = create_tensor<short>(s0);
                WriteableSliceIterator it2 = result->write_slices();
                _execute_tensor_scalar<short, T<int>, C<int, short>>(it0, extract<int>(t1), it2);
                return result;
            }
            else if (s0.element() == IntegerIdentifier)
            {
                TensorReference result = create_tensor<int>(s0);
                WriteableSliceIterator it2 = result->write_slices();
                _execute_tensor_scalar<int, T<int>, C<int, int>>(it0, extract<int>(t1), it2);
                return result;
            }
            else if (s0.element() == FloatIdentifier)
            {
                TensorReference result = create_tensor<float>(s0);
                WriteableSliceIterator it2 = result->write_slices();
                _execute_tensor_scalar<float, T<float>, nonsaturate_cast<float, float>>(it0, extract<float>(t1), it2);
                return result;
            }
            else
            {
                throw TypeException("Unsupported tensor type");
            }
        }

        if (t1->is<Tensor>())
        {

            TensorReference tr1 = extract<TensorReference>(t1);
            Shape s1 = tr1->shape();

            ReadonlySliceIterator it1 = tr1->read_slices();

            if (s1.element() == CharIdentifier)
            {
                TensorReference result = create_tensor<char>(s1);
                WriteableSliceIterator it2 = result->write_slices();
                _execute_scalar_tensor<char, T<int>, C<int, char>>(extract<int>(t0), it1, it2);
                return result;
            }
            else if (s1.element() == ShortIdentifier)
            {
                TensorReference result = create_tensor<short>(s1);
                WriteableSliceIterator it2 = result->write_slices();
                _execute_scalar_tensor<short, T<int>, C<int, short>>(extract<int>(t0), it1, it2);
                return result;
            }
            else if (s1.element() == IntegerIdentifier)
            {
                TensorReference result = create_tensor<int>(s1);
                WriteableSliceIterator it2 = result->write_slices();
                _execute_scalar_tensor<int, T<int>, C<int, int>>(extract<int>(t0), it1, it2);
                return result;
            }
            else if (s1.element() == FloatIdentifier)
            {
                TensorReference result = create_tensor<float>(s1);
                WriteableSliceIterator it2 = result->write_slices();
                _execute_scalar_tensor<float, T<float>, nonsaturate_cast<float, float>>(extract<float>(t0), it1, it2);
                return result;
            }
            else
            {
                throw TypeException("Unsupported tensor type");
            }
        }

        throw TypeException("Not a tensor");
    }

#define tensor_add tensor_elementwise_binary<std::plus, nonsaturate_cast>
    PIXELPIPES_OPERATION_AUTO("tensor_add", tensor_add);

#define tensor_subtract tensor_elementwise_binary<std::minus, nonsaturate_cast>
    PIXELPIPES_OPERATION_AUTO("tensor_subtract", tensor_subtract);

#define tensor_multiply tensor_elementwise_binary<std::multiplies, nonsaturate_cast>
    PIXELPIPES_OPERATION_AUTO("tensor_multiply", tensor_multiply);

#define tensor_divide tensor_elementwise_binary<std::divides, nonsaturate_cast>
    PIXELPIPES_OPERATION_AUTO("tensor_divide", tensor_divide);

#define tensor_add_saturate tensor_elementwise_binary<std::plus, saturate_cast>
    PIXELPIPES_OPERATION_AUTO("tensor_add_saturate", tensor_add_saturate);

#define tensor_subtract_saturate tensor_elementwise_binary<std::minus, saturate_cast>
    PIXELPIPES_OPERATION_AUTO("tensor_subtract_saturate", tensor_subtract_saturate);

#define tensor_multiply_saturate tensor_elementwise_binary<std::multiplies, saturate_cast>
    PIXELPIPES_OPERATION_AUTO("tensor_multiply_saturate", tensor_multiply_saturate);

#define tensor_divide_saturate tensor_elementwise_binary<std::divides, saturate_cast>
    PIXELPIPES_OPERATION_AUTO("tensor_divide_saturate", tensor_divide_saturate);

}