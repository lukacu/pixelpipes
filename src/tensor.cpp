
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

    TensorReference create_scalar(const TokenReference &in)
    {
        Shape s = in->shape();

        if (!s.is_scalar())
            throw TypeException("Input not a scalar");

        TensorReference out = create_tensor(s.element(), SizeSequence({1}));

        if (s.element() == IntegerIdentifier)
        {
            out->data().reinterpret<int>()[0] = extract<int>(in);
        }
        else if (s.element() == FloatIdentifier)
        {
            out->data().reinterpret<float>()[0] = extract<float>(in);
        }
        else if (s.element() == CharIdentifier)
        {
            out->data().reinterpret<uchar>()[0] = extract<char>(in);
        }
        else if (s.element() == BooleanIdentifier)
        {
            out->data().reinterpret<bool>()[0] = extract<bool>(in);
        }
        else if (s.element() == ShortIdentifier)
        {
            out->data().reinterpret<short>()[0] = extract<short>(in);
        }
        else if (s.element() == UShortIdentifier)
        {
            out->data().reinterpret<ushort>()[0] = extract<ushort>(in);
        }
        else
        {
            throw TypeException("Unsupported tensor format");
        }

        return out;
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
        // VERIFY(!source->is<TensorView>(), "Unable to view existing views");

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
        template <typename TIN>
        inline auto operator()(TIN &val)
        {
            return xt::cast<B>(val);
        }
    };

    template <typename A, typename B>
    struct saturate_cast
    {
        template <typename TIN>
        inline auto operator()(TIN &val)
        {
            return xt::cast<B>(xt::clip(val, std::numeric_limits<B>::min(), std::numeric_limits<B>::max()));
        }
    };

    template <typename Op, typename C, typename TA, typename TB, typename TR>
    inline void _execute_tensor_binary(TA &&a, TB &&b, TR &&res)
    {
        C cast;
        Op operation;
        auto v = operation(a, b);
        res = cast(v);
    }

    template <typename TIN>
    inline void _execute_xtensor_cast(TIN &&ain, const TensorReference &out)
    {
        auto t = out->cell_type();

        if (t == CharIdentifier)
        {
            auto aout = wrap_xtensor<uchar>(out);
            aout = xt::cast<uchar>(ain);
        }
        else if (t == ShortIdentifier)
        {
            auto aout = wrap_xtensor<short>(out);
            aout = xt::cast<short>(ain);
        }
        else if (t == UShortIdentifier)
        {
            auto aout = wrap_xtensor<ushort>(out);
            aout = xt::cast<ushort>(ain);
        }
        else if (t == IntegerIdentifier)
        {
            auto aout = wrap_xtensor<int>(out);
            aout = xt::cast<int>(ain);
        }
        else if (t == FloatIdentifier)
        {
            auto aout = wrap_xtensor<float>(out);
            aout = xt::cast<float>(ain);
        }
        else
        {
            throw TypeException("Unsupported tensor type");
        }
    }

    void copy_tensor(const TensorReference &in, const TensorReference &out)
    {

        auto tin = in->cell_type();

        if (tin == out->cell_type())
        {
            if (tin == CharIdentifier)
            {
                auto aout = wrap_xtensor<uchar>(out);
                auto ain = wrap_xtensor<uchar>(in);
                xt::assign_xexpression(aout, ain);
            }
            else if (tin == ShortIdentifier)
            {
                auto aout = wrap_xtensor<short>(out);
                auto ain = wrap_xtensor<short>(in);
                aout = ain;
            }
            else if (tin == UShortIdentifier)
            {
                auto aout = wrap_xtensor<ushort>(out);
                auto ain = wrap_xtensor<ushort>(in);
                aout = xt::eval(ain);
                
            }
            else if (tin == IntegerIdentifier)
            {
                auto aout = wrap_xtensor<int>(out);
                auto ain = wrap_xtensor<int>(in);
                aout = xt::eval(ain);
            }
            else if (tin == FloatIdentifier)
            {
                auto aout = wrap_xtensor<float>(out);
                auto ain = wrap_xtensor<float>(in);
                xt::assign_xexpression(aout, ain);
            }
            else
            {
                throw TypeException("Unsupported tensor type");
            }


        }
        else
        {

            if (tin == CharIdentifier)
            {
                _execute_xtensor_cast(wrap_xtensor<uchar>(in), out);
            }
            else if (tin == ShortIdentifier)
            {
                _execute_xtensor_cast(wrap_xtensor<short>(in), out);
            }
            else if (tin == UShortIdentifier)
            {
                _execute_xtensor_cast(wrap_xtensor<ushort>(in), out);
            }
            else if (tin == IntegerIdentifier)
            {
                _execute_xtensor_cast(wrap_xtensor<int>(in), out);
            }
            else if (tin == FloatIdentifier)
            {
                _execute_xtensor_cast(wrap_xtensor<float>(in), out);
            }
            else
            {
                throw TypeException("Unsupported tensor type");
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

    template <class T, template <typename, typename> class C>
    TokenReference tensor_elementwise_binary(const TokenReference &ta, const TokenReference &tb)
    {
        Shape sa = ta->shape();
        Shape sb = tb->shape();

        TensorReference tra;
        TensorReference trb;

        if (sa.is_scalar())
        {
            tra = create_scalar(ta);
        }
        else
        {
            tra = extract<TensorReference>(ta);
        }

        if (sb.is_scalar())
        {
            trb = create_scalar(tb);
        }
        else
        {
            trb = extract<TensorReference>(tb);
        }

        size_t outdim = std::max(sa.rank(), sb.rank());
        SizeSequence outsize(outdim);

        for (size_t i = 0; i < outdim; i++)
        {
            if (sa[i] == 1)
            {
                outsize[i] = sb[i];
            }
            else if (sb[i] == 1)
            {
                outsize[i] = sa[i];
            }
            else if (sa[i] == sb[i])
            {
                outsize[i] = sa[i];
            }
            else
                throw TypeException("Tensor dimension mismatch");
        }

        TypeIdentifier return_type = 0;
        TensorReference result;

        auto strides_a = _broadcasting_strides(sa, outsize, tra->strides());
        auto strides_b = _broadcasting_strides(sb, outsize, trb->strides());

        TensorReference tva = create<TensorView>(tra, 0, outsize, strides_a);
        TensorReference tvb = create<TensorView>(trb, 0, outsize, strides_b);

        if (sa.element() != sb.element())
        {
            return_type = _promote_type(sa.element(), sb.element());

            result = create_tensor(return_type, outsize);
            if (return_type == sa.element())
            {
                copy_tensor(tvb, result);
                tvb = result.reborrow();
                tva = tra.reborrow();
            }
            else
            {
                copy_tensor(tva, result);
                tva = result.reborrow();
                tvb = trb.reborrow();
            }
        }
        else
        {
            return_type = sa.element();
            result = create_tensor(return_type, outsize);
            tva = tra.reborrow();
            tvb = trb.reborrow();
        }

        if (return_type == CharIdentifier)
        {
            auto a = wrap_xtensor<uchar>(tva);
            auto b = wrap_xtensor<uchar>(tvb);
            auto out = wrap_xtensor<uchar>(result);

            _execute_tensor_binary<T, C<int, uchar>>(a, b, out);
            return result;
        }
        else if (return_type == ShortIdentifier)
        {
            auto a = wrap_xtensor<short>(tva);
            auto b = wrap_xtensor<short>(tvb);
            auto out = wrap_xtensor<short>(result);

            _execute_tensor_binary<T, C<int, short>>(a, b, out);
            return result;
        }
        else if (return_type == UShortIdentifier)
        {
            auto a = wrap_xtensor<ushort>(tva);
            auto b = wrap_xtensor<ushort>(tvb);
            auto out = wrap_xtensor<ushort>(result);

            _execute_tensor_binary<T, C<int, ushort>>(a, b, out);
            return result;
        }
        else if (return_type == IntegerIdentifier)
        {
            auto a = wrap_xtensor<int>(tva);
            auto b = wrap_xtensor<int>(tvb);
            auto out = wrap_xtensor<int>(result);

            _execute_tensor_binary<T, C<int, int>>(a, b, out);
            return result;
        }
        else if (return_type == FloatIdentifier)
        {
            auto a = wrap_xtensor<float>(tva);
            auto b = wrap_xtensor<float>(tvb);
            auto out = wrap_xtensor<float>(result);

            // There is no saturation in floats
            _execute_tensor_binary<T, nonsaturate_cast<float, float>>(a, b, out);
            return result;
        }
        else
        {
            throw TypeException("Unsupported tensor type");
        }
    }

#define tensor_add tensor_elementwise_binary<xt::detail::plus, nonsaturate_cast>
    PIXELPIPES_OPERATION_AUTO("tensor_add", tensor_add);

#define tensor_subtract tensor_elementwise_binary<xt::detail::minus, nonsaturate_cast>
    PIXELPIPES_OPERATION_AUTO("tensor_subtract", tensor_subtract);

#define tensor_multiply tensor_elementwise_binary<xt::detail::multiplies, nonsaturate_cast>
    PIXELPIPES_OPERATION_AUTO("tensor_multiply", tensor_multiply);

#define tensor_divide tensor_elementwise_binary<xt::detail::divides, nonsaturate_cast>
    PIXELPIPES_OPERATION_AUTO("tensor_divide", tensor_divide);

#define tensor_add_saturate tensor_elementwise_binary<xt::detail::plus, saturate_cast>
    PIXELPIPES_OPERATION_AUTO("tensor_add_saturate", tensor_add_saturate);

#define tensor_subtract_saturate tensor_elementwise_binary<xt::detail::minus, saturate_cast>
    PIXELPIPES_OPERATION_AUTO("tensor_subtract_saturate", tensor_subtract_saturate);

#define tensor_multiply_saturate tensor_elementwise_binary<xt::detail::multiplies, saturate_cast>
    PIXELPIPES_OPERATION_AUTO("tensor_multiply_saturate", tensor_multiply_saturate);

#define tensor_divide_saturate tensor_elementwise_binary<xt::detail::divides, saturate_cast>
    PIXELPIPES_OPERATION_AUTO("tensor_divide_saturate", tensor_divide_saturate);

    class Stack : public Operation
    {
    public:
        Stack() {}

        virtual TokenReference run(const TokenList &inputs)
        {
            VERIFY(inputs.size() > 1, "Two or more tensors expected");

            TensorReference t0 = extract<TensorReference>(inputs[0]);

            Shape s = t0->shape();

            for (size_t i = 1; i < inputs.size(); i++)
            {
                TensorReference ti = extract<TensorReference>(inputs[i]);

                VERIFY(s == ti->shape(), "Shape mismatch");
            }

            s = s.push(inputs.size());

            TensorReference result = create_tensor(s);

            for (size_t i = 0; i < inputs.size(); i++)
            {
                TensorReference ts = extract<TensorReference>(inputs[i]);
                TensorReference td = extract<TensorReference>(result->get(i));

                copy_buffer(ts, td);
            }

            return result;
        }

        virtual TypeIdentifier type()
        {
            return GetTypeIdentifier<Stack>();
        }

        virtual Sequence<TokenReference> serialize() { return Sequence<TokenReference>(); }

    };

    PIXELPIPES_OPERATION_CLASS("stack", Stack);

    /**
     * @brief Converts depth of an image, scaling pixel values.
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