
#include <memory>
#include <algorithm>
#include <limits>

#include <pixelpipes/tensor.hpp>

namespace pixelpipes
{

    TensorReference create_tensor(TypeIdentifier element, Sizes sizes)
    {

        if (element == IntegerIdentifier)
        {
            return create_tensor<int>(sizes);
        }
        else if (element == FloatIdentifier)
        {
            return create_tensor<float>(sizes);
        }
        else if (element == CharIdentifier)
        {
            return create_tensor<char>(sizes);
        }
        else if (element == BooleanIdentifier)
        {
            return create_tensor<bool>(sizes);
        }
        else if (element == ShortIdentifier)
        {
            return create_tensor<short>(sizes);
        }
        else if (element == UShortIdentifier)
        {
            return create_tensor<ushort>(sizes);
        }
        else
        {
            throw TypeException("Unsupported tensor format");
        }
    }

    TensorReference create_tensor(Shape s)
    {

        return create_tensor(s.element(), SizeSequence(std::vector<size_t>(s.begin(), s.end())));
    }

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
        //VERIFY(!source->is<TensorView>(), "Unable to view existing views");

        VERIFY(shape.size() == strides.size(), "Size mismatch");

        _shape = SizeSequence(shape);
        _strides = SizeSequence(strides);

        _cell_size = source->cell_size();
        _cell_type = source->cell_type();

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
        return Shape(cell_type(), _shape);
    }

    size_t TensorView::length() const
    {
        return _shape[0];
    }

    size_t TensorView::cell_size() const
    {
        return _cell_size;
    }

    TypeIdentifier TensorView::cell_type() const
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

    template <typename F, typename T, template <typename, typename> class C>
    inline void _execute_tensor_cast(ReadonlySliceIterator &it0, WriteableSliceIterator &it1)
    {
        size_t offset0 = 0;
        size_t offset1 = 0;

        C<F, T> cast;

        while (true)
        {
            size_t len0 = (*it0).size() - offset0;
            size_t len1 = (*it1).size() - offset1;
            size_t length = (std::min)(len0 / sizeof(F), len1 / sizeof(T));

            if (length == 0)
                break;

            const F *c0 = (F *)(*it0).data() + offset0;
            T *c1 = (T *)((*it1).data() + offset1);

            for (size_t i = 0; i < length; i++)
            {
                c1[i] = cast(c0[i]);
            }

            offset0 += length * sizeof(F);
            offset1 += length * sizeof(T);

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
        }
    }

    template <typename F, template <typename, typename> class C>
    void _execute_tensor_cast(ReadonlySliceIterator &it0, const TensorReference &t1)
    {

        WriteableSliceIterator it1 = t1->write_slices();
        auto c1 = t1->cell_type();
        if (c1 == CharIdentifier)
        {
            _execute_tensor_cast<F, char, C>(it0, it1);
        }
        else if (c1 == ShortIdentifier)
        {
            _execute_tensor_cast<F, short, C>(it0, it1);
        }
        else if (c1 == UShortIdentifier)
        {
            _execute_tensor_cast<F, ushort, C>(it0, it1);
        }
        else if (c1 == IntegerIdentifier)
        {
            _execute_tensor_cast<F, int, C>(it0, it1);
        }
        else if (c1 == FloatIdentifier)
        {
            _execute_tensor_cast<F, float, C>(it0, it1);
        }
        else
        {
            throw TypeException("Unsupported tensor type");
        }
    }

    template <template <typename, typename> class C>
    void _tensor_copy_cast(const TensorReference &t0, const TensorReference &t1)
    {
        ReadonlySliceIterator it0 = t0->read_slices();

        auto c0 = t0->cell_type();

        if (c0 == CharIdentifier)
        {
            _execute_tensor_cast<uchar, C>(it0, t1);
        }
        else if (c0 == ShortIdentifier)
        {
            _execute_tensor_cast<short, C>(it0, t1);
        }
        else if (c0 == UShortIdentifier)
        {
            _execute_tensor_cast<ushort, C>(it0, t1);
        }
        else if (c0 == IntegerIdentifier)
        {
            _execute_tensor_cast<int, C>(it0, t1);
        }
        else if (c0 == FloatIdentifier)
        {
            _execute_tensor_cast<float, C>(it0, t1);
        }
        else
        {
            throw TypeException("Unsupported tensor type");
        }
    }

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

    inline SizeSequence _broadcasting_strides(const Shape &original, const SizeSequence &desired, const SizeSequence &strides)
    {
        auto bstrides = SizeSequence::repeat(desired.size(), (size_t)original[original.rank() - 1]);

        for (size_t i = 0; i < original.rank(); i++)
        {
            if (desired[i] != (size_t)original[i])
            {
                bstrides[i] = 0;
            }
            else
            {
                bstrides[i] = strides[i];
            }
        }

        return bstrides;
    }

    inline TypeIdentifier _promote_type(TypeIdentifier a, TypeIdentifier b)
    {
        static TypeIdentifier ordered[] = {CharIdentifier, ShortIdentifier, UShortIdentifier, IntegerIdentifier, FloatIdentifier};

        size_t ai = 0;
        size_t bi = 0;

        for (size_t i = 0; i < 5; i++)
        {
            if (ordered[i] == a)
                ai = i;
            if (ordered[i] == b)
                bi = i;
        }

        if (ai > bi)
            return a;
        return b;
    }

    template <template <typename> class T, template <typename, typename> class C>
    TokenReference tensor_elementwise_binary(const TokenReference &t0, const TokenReference &t1)
    {
        Shape s0 = t0->shape();
        Shape s1 = t1->shape();

        if (!s0.is_scalar() && !s1.is_scalar())
        {

            TensorReference tr0 = extract<TensorReference>(t0);
            TensorReference tr1 = extract<TensorReference>(t1);

            size_t outdim = std::max(s0.rank(), s1.rank());
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

            TensorReference tv0 = create<TensorView>(tr0, 0, outsize, strides0);
            TensorReference tv1 = create<TensorView>(tr1, 0, outsize, strides1);

            TypeIdentifier return_type = 0;
            TensorReference result;

            if (s0.element() != s1.element())
            {
                return_type = _promote_type(s0.element(), s1.element());
                result = create_tensor(return_type, outsize);
                if (return_type == s0.element())
                {
                    _tensor_copy_cast<nonsaturate_cast>(tv1, result);
                    tv1 = result.reborrow();
                }
                else
                {
                    _tensor_copy_cast<nonsaturate_cast>(tv0, result);
                    tv0 = result.reborrow();
                }
            }
            else
            {
                return_type = s0.element();
                result = create_tensor(return_type, outsize);
            }

            ReadonlySliceIterator it0 = tv0->read_slices();
            ReadonlySliceIterator it1 = tv1->read_slices();
            WriteableSliceIterator it2 = result->write_slices();

            if (return_type == CharIdentifier)
            {
                _execute_tensor_tensor<uchar, T<int>, C<int, uchar>>(it0, it1, it2);
                return result;
            }
            else if (return_type == ShortIdentifier)
            {
                _execute_tensor_tensor<short, T<int>, C<int, short>>(it0, it1, it2);
                return result;
            }
            else if (return_type == UShortIdentifier)
            {
                _execute_tensor_tensor<ushort, T<int>, C<int, ushort>>(it0, it1, it2);
                return result;
            }
            else if (return_type == IntegerIdentifier)
            {
                _execute_tensor_tensor<int, T<int>, C<int, int>>(it0, it1, it2);
                return result;
            }
            else if (return_type == FloatIdentifier)
            {
                _execute_tensor_tensor<float, T<float>, nonsaturate_cast<float, float>>(it0, it1, it2);
                return result;
            }
            else
            {
                throw TypeException("Unsupported tensor type");
            }
        }

        if (!s0.is_scalar())
        {

            TensorReference tr0 = extract<TensorReference>(t0);

            /*TypeIdentifier return_type = 0;
            TensorReference result;

            if (s0.element() != s1.element()) {
                return_type = _promote_type(s0.element(), s1.element());
                result = create_tensor(return_type, outsize);
                if (return_type != s0.element()) {
                    _tensor_copy_cast<nonsaturate_cast>(tr0, result);
                    tr0 = result;
                }
            } else {
                return_type = s0.element();
                result = create_tensor(s0);
            }*/

            ReadonlySliceIterator it0 = tr0->read_slices();

            if (s0.element() == CharIdentifier)
            {
                TensorReference result = create_tensor(s0);
                WriteableSliceIterator it2 = result->write_slices();
                _execute_tensor_scalar<uchar, T<int>, C<int, uchar>>(it0, extract<int>(t1), it2);
                return result;
            }
            else if (s0.element() == ShortIdentifier)
            {
                TensorReference result = create_tensor(s0);
                WriteableSliceIterator it2 = result->write_slices();
                _execute_tensor_scalar<short, T<int>, C<int, short>>(it0, extract<int>(t1), it2);
                return result;
            }
            else if (s0.element() == UShortIdentifier)
            {
                TensorReference result = create_tensor(s0);
                WriteableSliceIterator it2 = result->write_slices();
                _execute_tensor_scalar<ushort, T<int>, C<int, ushort>>(it0, extract<int>(t1), it2);
                return result;
            }
            else if (s0.element() == IntegerIdentifier)
            {
                TensorReference result = create_tensor(s0);
                WriteableSliceIterator it2 = result->write_slices();
                _execute_tensor_scalar<int, T<int>, C<int, int>>(it0, extract<int>(t1), it2);
                return result;
            }
            else if (s0.element() == FloatIdentifier)
            {
                TensorReference result = create_tensor(s0);
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
            ReadonlySliceIterator it1 = tr1->read_slices();

            if (s1.element() == CharIdentifier)
            {
                TensorReference result = create_tensor(s1);
                WriteableSliceIterator it2 = result->write_slices();
                _execute_scalar_tensor<uchar, T<int>, C<int, uchar>>(extract<int>(t0), it1, it2);
                return result;
            }
            else if (s1.element() == ShortIdentifier)
            {
                TensorReference result = create_tensor(s1);
                WriteableSliceIterator it2 = result->write_slices();
                _execute_scalar_tensor<short, T<int>, C<int, short>>(extract<int>(t0), it1, it2);
                return result;
            }
            if (s1.element() == UShortIdentifier)
            {
                TensorReference result = create_tensor(s1);
                WriteableSliceIterator it2 = result->write_slices();
                _execute_scalar_tensor<ushort, T<int>, C<int, ushort>>(extract<int>(t0), it1, it2);
                return result;
            }
            else if (s1.element() == IntegerIdentifier)
            {
                TensorReference result = create_tensor(s1);
                WriteableSliceIterator it2 = result->write_slices();
                _execute_scalar_tensor<int, T<int>, C<int, int>>(extract<int>(t0), it1, it2);
                return result;
            }
            else if (s1.element() == FloatIdentifier)
            {
                TensorReference result = create_tensor(s1);
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

    /*
    #define tensor_add tensor_elementwise_binary<std::plus, nonsaturate_cast>
        PIXELPIPES_OPERATION_AUTO("tensor_add", tensor_add);

    #define tensor_subtract tensor_elementwise_binary<std::minus, nonsaturate_cast>
        PIXELPIPES_OPERATION_AUTO("tensor_subtract", tensor_subtract);

    #define tensor_multiply tensor_elementwise_binary<std::multiplies, nonsaturate_cast>
        PIXELPIPES_OPERATION_AUTO("tensor_multiply", tensor_multiply);

    #define tensor_divide tensor_elementwise_binary<std::divides, nonsaturate_cast>
        PIXELPIPES_OPERATION_AUTO("tensor_divide", tensor_divide);
    */

}