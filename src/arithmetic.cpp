
#include <memory>
#include <algorithm>
#include <limits>

#include <xtensor/xtensor.hpp>
#include <xtensor/xtensor_simd.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xnoalias.hpp>

#include <pixelpipes/tensor.hpp>
#include <pixelpipes/operation.hpp>

namespace pixelpipes
{
    TensorReference create_tensor(Type element, Sizes sizes)
    {

        if (element == IntegerType)
        {
            return create_tensor<int>(sizes);
        }
        else if (element == FloatType)
        {
            return create_tensor<float>(sizes);
        }
        else if (element == CharType)
        {
            return create_tensor<char>(sizes);
        }
        else if (element == BooleanType)
        {
            return create_tensor<bool>(sizes);
        }
        else if (element == ShortType)
        {
            return create_tensor<short>(sizes);
        }
        else if (element == UnsignedShortType)
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
        // Not the nicest way to do this, but it works (optimization is not a priority here)
        //return create_tensor(s.element(), s.sizes());
        return create_tensor(s.element(), SizeSequence(std::vector<size_t>(s.begin(), s.end())));
    }

    TensorReference create_scalar(const TokenReference &in)
    {
        Shape s = in->shape();

        if (!s.is_scalar())
            throw TypeException("Input not a scalar");

        if (s.element() == IntegerType)
        {
            return create<IntegerScalar>(extract<int>(in));
        }
        else if (s.element() == FloatType)
        {
            return create<FloatScalar>(extract<float>(in));
        }
        else if (s.element() == CharType)
        {
            return create<CharScalar>(extract<char>(in));
        }
        else if (s.element() == BooleanType)
        {
            return create<BooleanScalar>(extract<bool>(in));
        }
        else if (s.element() == ShortType)
        {
            return create<ShortScalar>(extract<short>(in));
        }
        else if (s.element() == UnsignedShortType)
        {
            return create<UShortScalar>(extract<ushort>(in));
        }
        else
        {
            throw TypeException("Unsupported format");
        }
    }

    template <typename A, typename B>
    struct no_cast
    {
        template <typename TIN>
        inline auto operator()(TIN &val)
        {
            return val;
        }
    };

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
        auto t = out->datatype();

        if (t == CharType)
        {
            auto aout = wrap_xtensor<uchar>(out);
            aout = xt::cast<uchar>(xt::clip(ain, std::numeric_limits<uchar>::min(), std::numeric_limits<uchar>::max()));
        }
        else if (t == ShortType)
        {
            auto aout = wrap_xtensor<short>(out);
            aout = xt::cast<short>(xt::clip(ain, std::numeric_limits<short>::min(), std::numeric_limits<short>::max()));
        }
        else if (t == UnsignedShortType)
        {
            auto aout = wrap_xtensor<ushort>(out);
            aout = xt::cast<ushort>(xt::clip(ain, std::numeric_limits<ushort>::min(), std::numeric_limits<ushort>::max()));
        }
        else if (t == IntegerType)
        {
            auto aout = wrap_xtensor<int>(out);
            aout = xt::cast<int>(xt::clip(ain, std::numeric_limits<int>::min(), std::numeric_limits<int>::max()));
        }
        else if (t == FloatType)
        {
            auto aout = wrap_xtensor<float>(out);
            aout = xt::cast<float>(ain);
        }
        else if (t == BooleanType)
        {
            auto aout = wrap_xtensor<bool>(out);
            aout = xt::cast<bool>(ain);
        }
        else
        {
            throw TypeException("Unsupported tensor type");
        }
    }

    void copy_tensor(const TensorReference &in, const TensorReference &out)
    {

        auto tin = in->datatype();

        if (tin == out->datatype())
        {
            if (tin == CharType)
            {
                auto aout = wrap_xtensor<uchar>(out);
                auto ain = wrap_xtensor<uchar>(in);
                xt::assign_xexpression(aout, ain);
            }
            else if (tin == ShortType)
            {
                auto aout = wrap_xtensor<short>(out);
                auto ain = wrap_xtensor<short>(in);
                aout = ain;
            }
            else if (tin == UnsignedShortType)
            {
                auto aout = wrap_xtensor<ushort>(out);
                auto ain = wrap_xtensor<ushort>(in);
                aout = xt::eval(ain);
            }
            else if (tin == IntegerType)
            {
                auto aout = wrap_xtensor<int>(out);
                auto ain = wrap_xtensor<int>(in);
                aout = xt::eval(ain);
            }
            else if (tin == FloatType)
            {
                auto aout = wrap_xtensor<float>(out);
                auto ain = wrap_xtensor<float>(in);
                xt::assign_xexpression(aout, ain);
            }
            else if (tin == BooleanType)
            {
                auto aout = wrap_xtensor<bool>(out);
                auto ain = wrap_xtensor<bool>(in);
                xt::assign_xexpression(aout, ain);
            }
            else
            {
                throw TypeException("Unsupported tensor type");
            }
        }
        else
        {

            if (tin == CharType)
            {
                _execute_xtensor_cast(wrap_xtensor<uchar>(in), out);
            }
            else if (tin == ShortType)
            {
                _execute_xtensor_cast(wrap_xtensor<short>(in), out);
            }
            else if (tin == UnsignedShortType)
            {
                _execute_xtensor_cast(wrap_xtensor<ushort>(in), out);
            }
            else if (tin == IntegerType)
            {
                _execute_xtensor_cast(wrap_xtensor<int>(in), out);
            }
            else if (tin == FloatType)
            {
                _execute_xtensor_cast(wrap_xtensor<float>(in), out);
            }
            else if (tin == BooleanType)
            {
                _execute_xtensor_cast(wrap_xtensor<bool>(in), out);
            }
            else
            {
                throw TypeException("Unsupported tensor type");
            }
        }
    }

    inline SizeSequence _broadcasting_strides(const Shape &original, const SizeSequence &desired, const SizeSequence &strides)
    /**
     * @brief Computes strides for broadcasting a tensor
     *
     * @param original Original shape
     * @param desired Desired shape
     * @param strides Strides of the original tensor
     *
     * @return Strides for the broadcasted tensor
     */
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

    inline Type _promote_type(Type a, Type b)
    {
        static Type ordered[] = {CharType, ShortType, UnsignedShortType, IntegerType, FloatType};

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

    Shape _infer_common_shape(const Shape& sa, const Shape& sb) 
    {

        size_t outdim = std::max(sa.rank(), sb.rank());
        SizeSequence outsize(outdim);

        for (size_t i = 0; i < outdim; i++)
        {
            if (!(sa[i]) || !(sb[i]))
            {
                outsize[i] = unknown;
            }
            else if (sa[i] == 1)
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
            {
                throw TypeException(std::string("Shape dimension mismatch:") + to_string(sa) + " != " + to_string(sb));
            }
        }

        return Shape(AnyType, outsize);

    }


    TokenReference _determine_shape_binary(const TokenList &inputs)
    {

        VERIFY(inputs.size() == 2, "At least two tokens expected");

        Shape sa = inputs[0]->shape();
        Shape sb = inputs[1]->shape();

        Shape common_shape = _infer_common_shape(sa, sb);

        if (sa.element() == AnyType || sb.element() == AnyType)
            return create<Placeholder>(common_shape);

        return create<Placeholder>(common_shape.cast(_promote_type(sa.element(), sb.element())));
    }

    template <typename T>
    TokenReference _determine_shape_binary_cast(const TokenList &inputs)
    {
        VERIFY(inputs.size() == 2, "At least two tokens expected");

        Shape sa = inputs[0]->shape();
        Shape sb = inputs[1]->shape();

        Shape common_shape = _infer_common_shape(sa, sb);

        return create<Placeholder>(common_shape.cast(GetType<T>()));
    }

    TokenReference _determine_shape_unary(const TokenList &inputs)
    {

        VERIFY(inputs.size() == 1, "One token expected");

        Shape sa = inputs[0]->shape();

        return create<Placeholder>(sa);
    }

    template <typename T>
    TokenReference _determine_shape_unary_cast(const TokenList &inputs)
    {

        VERIFY(inputs.size() == 1, "One token expected");

        Shape sa = inputs[0]->shape();

        return create<Placeholder>(sa.cast(GetType<T>()));
    }

    template <class T, template <typename, typename> class C>
    TokenReference _elementwise_binary(const TokenReference &ta, const TokenReference &tb)
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

        Type return_type = 0;
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

        if (return_type == CharType)
        {
            auto a = wrap_xtensor<uchar>(tva);
            auto b = wrap_xtensor<uchar>(tvb);
            auto out = wrap_xtensor<uchar>(result);

            _execute_tensor_binary<T, C<int, uchar>>(a, b, out);
            return result;
        }
        else if (return_type == ShortType)
        {
            auto a = wrap_xtensor<short>(tva);
            auto b = wrap_xtensor<short>(tvb);
            auto out = wrap_xtensor<short>(result);

            _execute_tensor_binary<T, C<int, short>>(a, b, out);
            return result;
        }
        else if (return_type == UnsignedShortType)
        {
            auto a = wrap_xtensor<ushort>(tva);
            auto b = wrap_xtensor<ushort>(tvb);
            auto out = wrap_xtensor<ushort>(result);

            _execute_tensor_binary<T, C<int, ushort>>(a, b, out);
            return result;
        }
        else if (return_type == IntegerType)
        {
            auto a = wrap_xtensor<int>(tva);
            auto b = wrap_xtensor<int>(tvb);
            auto out = wrap_xtensor<int>(result);

            _execute_tensor_binary<T, C<int, int>>(a, b, out);
            return result;
        }
        else if (return_type == FloatType)
        {
            auto a = wrap_xtensor<float>(tva);
            auto b = wrap_xtensor<float>(tvb);
            auto out = wrap_xtensor<float>(result);

            // There is no saturation in floats
            _execute_tensor_binary<T, no_cast<float, float>>(a, b, out);
            return result;
        }

        else
        {
            throw TypeException("Unsupported tensor type");
        }
    }

    TokenReference common_shape(const TokenList &inputs)
    {

        VERIFY(inputs.size() > 0, "At least one tensor expected");

        TensorReference t0 = extract<TensorReference>(inputs[0]);

        Shape s = t0->shape();

        for (size_t i = 1; i < inputs.size(); i++)
        {
            TensorReference ti = extract<TensorReference>(inputs[i]);

            VERIFY(s == ti->shape(), "Shape mismatch");
        }

        return t0;
    }

    template <typename T, template <typename, typename> class C>
    TokenReference _elementwise_unary(const TokenReference &ta)
    {
        Shape sa = ta->shape();

        TensorReference tra;

        if (sa.is_scalar())
        {
            tra = create_scalar(ta);
        }
        else
        {
            tra = extract<TensorReference>(ta);
        }

        TensorReference result = create_tensor(sa);

        if (sa.element() == CharType)
        {
            auto a = wrap_xtensor<uchar>(tra);
            auto out = wrap_xtensor<uchar>(result);

            C<int, uchar> cast;
            T operation;

            auto v = operation(a);
            out = cast(v);
        }
        else if (sa.element() == ShortType)
        {
            auto a = wrap_xtensor<short>(tra);
            auto out = wrap_xtensor<short>(result);

            C<int, short> cast;
            T operation;

            auto v = operation(a);
            out = cast(v);
        }
        else if (sa.element() == UnsignedShortType)
        {
            auto a = wrap_xtensor<ushort>(tra);
            auto out = wrap_xtensor<ushort>(result);

            C<int, ushort> cast;
            T operation;

            auto v = operation(a);
            out = cast(v);
        }
        else if (sa.element() == IntegerType)
        {
            auto a = wrap_xtensor<int>(tra);
            auto out = wrap_xtensor<int>(result);

            C<int, int> cast;
            T operation;

            auto v = operation(a);
            out = cast(v);
        }
        else if (sa.element() == FloatType)
        {
            auto a = wrap_xtensor<float>(tra);
            auto out = wrap_xtensor<float>(result);

            C<float, float> cast;
            T operation;

            auto v = operation(a);
            out = cast(v);
        }
        else
        {
            throw TypeException("Unsupported tensor type");
        }

        return result;
    }

    // Arithmetic operations on tensors

#define _add _elementwise_binary<xt::detail::plus, nonsaturate_cast>
    PIXELPIPES_COMPUTE_OPERATION_AUTO("add", _add, _determine_shape_binary);

#define _subtract _elementwise_binary<xt::detail::minus, nonsaturate_cast>
    PIXELPIPES_COMPUTE_OPERATION_AUTO("subtract", _subtract, _determine_shape_binary);

#define _multiply _elementwise_binary<xt::detail::multiplies, nonsaturate_cast>
    PIXELPIPES_COMPUTE_OPERATION_AUTO("multiply", _multiply, _determine_shape_binary);

#define _divide _elementwise_binary<xt::detail::divides, nonsaturate_cast>
    PIXELPIPES_COMPUTE_OPERATION_AUTO("divide", _divide, _determine_shape_binary);

#define _add_saturate _elementwise_binary<xt::detail::plus, saturate_cast>
    PIXELPIPES_COMPUTE_OPERATION_AUTO("add_saturate", _add_saturate, _determine_shape_binary);

#define _subtract_saturate _elementwise_binary<xt::detail::minus, saturate_cast>
    PIXELPIPES_COMPUTE_OPERATION_AUTO("subtract_saturate", _subtract_saturate, _determine_shape_binary);

#define _multiply_saturate _elementwise_binary<xt::detail::multiplies, saturate_cast>
    PIXELPIPES_COMPUTE_OPERATION_AUTO("multiply_saturate", _multiply_saturate, _determine_shape_binary);

#define _divide_saturate _elementwise_binary<xt::detail::divides, saturate_cast>
    PIXELPIPES_COMPUTE_OPERATION_AUTO("divide_saturate", _divide_saturate, _determine_shape_binary);

    // Functor operations for power, modulo, sqrt

#define XTENSOR_FUNCTOR_UNARY(name, op) \
    struct name                         \
    {                                   \
        template <typename T1>          \
        inline auto operator()(T1 &a)   \
        {                               \
            return op(a);               \
        }                               \
    };

#define XTENSOR_FUNCTOR_BINARY(name, op)     \
    struct name                              \
    {                                        \
        template <typename T1, typename T2>  \
        inline auto operator()(T1 &a, T2 &b) \
        {                                    \
            return op(a, b);                 \
        }                                    \
    };

    struct tensor_modulo_functor
    {
        template <typename T1, typename T2>
        inline auto operator()(T1 &a, T2 &b)
        {
            // Switch between floating point and integer modulo
            if constexpr (std::is_integral<T1>::value && std::is_integral<T2>::value)
            {
                return a % b;
            }
            else
            {
                return xt::fmod(a, b);
            }
        }
    };

#define _modulo _elementwise_binary<tensor_modulo_functor, nonsaturate_cast>
    PIXELPIPES_COMPUTE_OPERATION_AUTO("modulo", _modulo, _determine_shape_binary);

    XTENSOR_FUNCTOR_BINARY(tensor_power_functor, xt::pow)

#define _power _elementwise_binary<tensor_power_functor, nonsaturate_cast>
    PIXELPIPES_COMPUTE_OPERATION_AUTO("power", _power, _determine_shape_binary_cast<float>);

    XTENSOR_FUNCTOR_UNARY(tensor_sqrt_functor, xt::sqrt)

#define _sqrt _elementwise_unary<tensor_sqrt_functor, nonsaturate_cast>
    PIXELPIPES_COMPUTE_OPERATION_AUTO("sqrt", _sqrt, _determine_shape_binary_cast<float>);

    // Trigonometric operations on tensors

    XTENSOR_FUNCTOR_UNARY(xtensor_sin_functor, xt::sin)

#define _sin _elementwise_unary<xtensor_sin_functor, nonsaturate_cast>
    PIXELPIPES_COMPUTE_OPERATION_AUTO("sin", _sin, _determine_shape_unary_cast<float>);

    XTENSOR_FUNCTOR_UNARY(xtensor_cos_functor, xt::cos)

#define _cos _elementwise_unary<xtensor_cos_functor, nonsaturate_cast>
    PIXELPIPES_COMPUTE_OPERATION_AUTO("cos", _cos, _determine_shape_unary_cast<float>);

    XTENSOR_FUNCTOR_UNARY(xtensor_tan_functor, xt::tan)

#define _tan _elementwise_unary<xtensor_tan_functor, nonsaturate_cast>
    PIXELPIPES_COMPUTE_OPERATION_AUTO("tan", _tan, _determine_shape_unary_cast<float>);

    XTENSOR_FUNCTOR_UNARY(xtensor_asin_functor, xt::asin)

#define _asin _elementwise_unary<xtensor_asin_functor, nonsaturate_cast>
    PIXELPIPES_COMPUTE_OPERATION_AUTO("asin", _asin, _determine_shape_unary_cast<float>);

    XTENSOR_FUNCTOR_UNARY(xtensor_acos_functor, xt::acos)

#define _acos _elementwise_unary<xtensor_acos_functor, nonsaturate_cast>
    PIXELPIPES_COMPUTE_OPERATION_AUTO("acos", _acos, _determine_shape_unary_cast<float>);

    XTENSOR_FUNCTOR_UNARY(xtensor_atan_functor, xt::atan)

#define _atan _elementwise_unary<xtensor_atan_functor, nonsaturate_cast>
    PIXELPIPES_COMPUTE_OPERATION_AUTO("atan", _atan, _determine_shape_unary_cast<float>);

    // Rounding operations

    XTENSOR_FUNCTOR_UNARY(xtensor_ceil_functor, xt::ceil)

#define _ceil _elementwise_unary<xtensor_ceil_functor, nonsaturate_cast>
    PIXELPIPES_COMPUTE_OPERATION_AUTO("ceil", _ceil, _determine_shape_unary);

    XTENSOR_FUNCTOR_UNARY(xtensor_floor_functor, xt::floor)

#define _floor _elementwise_unary<xtensor_floor_functor, nonsaturate_cast>
    PIXELPIPES_COMPUTE_OPERATION_AUTO("floor", _floor, _determine_shape_unary);

    XTENSOR_FUNCTOR_UNARY(xtensor_round_functor, xt::round)

#define _round _elementwise_unary<xtensor_round_functor, nonsaturate_cast>
    PIXELPIPES_COMPUTE_OPERATION_AUTO("round", _round, _determine_shape_unary);

    // Min and max operations

    XTENSOR_FUNCTOR_BINARY(xtensor_min_functor, xt::minimum)
#define _minimum _elementwise_binary<xtensor_min_functor, nonsaturate_cast>
    PIXELPIPES_COMPUTE_OPERATION_AUTO("min", _minimum, _determine_shape_binary);

    XTENSOR_FUNCTOR_BINARY(xtensor_max_functor, xt::maximum)
#define _maximum _elementwise_binary<xtensor_max_functor, nonsaturate_cast>
    PIXELPIPES_COMPUTE_OPERATION_AUTO("max", _maximum, _determine_shape_binary);

    // Comparison operations on tensors

    template <class T>
    TokenReference _elementwise_comparison(const TokenReference &ta, const TokenReference &tb)
    {
        Shape sa = ta->shape();
        Shape sb = tb->shape();

        TensorReference tra;
        TensorReference trb;

        bool scalar_a = sa.is_scalar();
        bool scalar_b = sb.is_scalar();

        if (scalar_a)
        {
            tra = create_scalar(ta);
        }
        else
        {
            tra = extract<TensorReference>(ta);
        }

        if (scalar_b)
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

        Type common_type = 0;
        TensorReference converted;

        auto strides_a = _broadcasting_strides(sa, outsize, tra->strides());
        auto strides_b = _broadcasting_strides(sb, outsize, trb->strides());

        TensorReference tva = create<TensorView>(tra, 0, outsize, strides_a);
        TensorReference tvb = create<TensorView>(trb, 0, outsize, strides_b);

        if (sa.element() != sb.element())
        {
            common_type = _promote_type(sa.element(), sb.element());

            converted = create_tensor(common_type, outsize);
            if (common_type == sa.element())
            {
                copy_tensor(tvb, converted);
                tvb = converted.reborrow();
                tva = tra.reborrow();
            }
            else
            {
                copy_tensor(tva, converted);
                tva = converted.reborrow();
                tvb = trb.reborrow();
            }
        }
        else
        {
            common_type = sa.element();
            tva = tra.reborrow();
            tvb = trb.reborrow();
        }
       
        TensorReference result = create_tensor(BooleanType, outsize);
        auto out = wrap_xtensor<bool>(result);

        if (common_type == CharType)
        {
            auto a = wrap_xtensor<uchar>(tva);
            auto b = wrap_xtensor<uchar>(tvb);
 
            _execute_tensor_binary<T, no_cast<bool, bool>>(a, b, out);
            return result;
        }
        else if (common_type == ShortType)
        {
            auto a = wrap_xtensor<short>(tva);
            auto b = wrap_xtensor<short>(tvb);

            _execute_tensor_binary<T, no_cast<bool, bool>>(a, b, out);
            return result;
        }
        else if (common_type == UnsignedShortType)
        {
            auto a = wrap_xtensor<ushort>(tva);
            auto b = wrap_xtensor<ushort>(tvb);

            _execute_tensor_binary<T, no_cast<bool, bool>>(a, b, out);
            return result;
        }
        else if (common_type == IntegerType)
        {
            auto a = wrap_xtensor<int>(tva);
            auto b = wrap_xtensor<int>(tvb);

            _execute_tensor_binary<T, no_cast<bool, bool>>(a, b, out);
            return result;
        }
        else if (common_type == FloatType)
        {
            auto a = wrap_xtensor<float>(tva);
            auto b = wrap_xtensor<float>(tvb);

            _execute_tensor_binary<T, no_cast<bool, bool>>(a, b, out);
            return result;
        }
        else
        {
            throw TypeException("Unsupported tensor type");
        }

        return result;
    }

    XTENSOR_FUNCTOR_BINARY(xtensor_equal_functor, xt::equal)

#define _equal _elementwise_comparison<xtensor_equal_functor>
    PIXELPIPES_COMPUTE_OPERATION_AUTO("equal", _equal, _determine_shape_binary_cast<bool>);

    XTENSOR_FUNCTOR_BINARY(xtensor_not_equal_functor, xt::not_equal)

#define _not_equal _elementwise_comparison<xtensor_not_equal_functor>
    PIXELPIPES_COMPUTE_OPERATION_AUTO("not_equal", _not_equal, _determine_shape_binary_cast<bool>);

#define _less _elementwise_comparison<xt::detail::less>
    PIXELPIPES_COMPUTE_OPERATION_AUTO("less", _less, _determine_shape_binary_cast<bool>);

#define _less_equal _elementwise_comparison<xt::detail::less_equal>
    PIXELPIPES_COMPUTE_OPERATION_AUTO("less_equal", _less_equal, _determine_shape_binary_cast<bool>);

#define _greater _elementwise_comparison<xt::detail::greater>
    PIXELPIPES_COMPUTE_OPERATION_AUTO("greater", _greater, _determine_shape_binary_cast<bool>);

#define _greater_equal _elementwise_comparison<xt::detail::greater_equal>
    PIXELPIPES_COMPUTE_OPERATION_AUTO("greater_equal", _greater_equal, _determine_shape_binary_cast<bool>);

    // Logical operations on tensors

    template <class T>
    TokenReference _logical_binary(const TokenReference &ta, const TokenReference &tb)
    {
        Shape sa = ta->shape();
        Shape sb = tb->shape();

        TensorReference tra;
        TensorReference trb;

        VERIFY(sa.element() == BooleanType, "Boolean type expected");
        VERIFY(sb.element() == BooleanType, "Boolean type expected");

        bool scalar_a = sa.is_scalar();
        bool scalar_b = sb.is_scalar();

        if (scalar_a)
        {
            tra = create_scalar(ta);
        }
        else
        {
            tra = extract<TensorReference>(ta);
        }

        if (scalar_b)
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

        TensorReference result;

        auto strides_a = _broadcasting_strides(sa, outsize, tra->strides());
        auto strides_b = _broadcasting_strides(sb, outsize, trb->strides());

        TensorReference tva = create<TensorView>(tra, 0, outsize, strides_a);
        TensorReference tvb = create<TensorView>(trb, 0, outsize, strides_b);

        result = create_tensor(BooleanType, outsize);
        tva = tra.reborrow();
        tvb = trb.reborrow();

        auto a = wrap_xtensor<bool>(tva);
        auto b = wrap_xtensor<bool>(tvb);
        auto out = wrap_xtensor<bool>(result);
        _execute_tensor_binary<T, no_cast<bool, bool>>(a, b, out);
        return result;
    }

    template <typename T>
    TokenReference _logical_unary(const TokenReference &ta)
    {
        Shape sa = ta->shape();

        VERIFY(sa.element() == BooleanType, "Boolean type expected");

        TensorReference tra;

        if (sa.is_scalar())
        {
            tra = create_scalar(ta);
        }
        else
        {
            tra = extract<TensorReference>(ta);
        }

        TensorReference result = create_tensor(sa);

        auto a = wrap_xtensor<bool>(tra);
        auto out = wrap_xtensor<bool>(result);

        T operation;
        out = operation(a);
        return result;
    }

#define _not _logical_unary<xt::detail::logical_not>
    PIXELPIPES_COMPUTE_OPERATION_AUTO("logical_not", _not, _determine_shape_unary_cast<bool>);

#define _and _logical_binary<xt::detail::logical_and>
    PIXELPIPES_COMPUTE_OPERATION_AUTO("logical_and", _and, _determine_shape_binary_cast<bool>);

#define _or _logical_binary<xt::detail::logical_or>
    PIXELPIPES_COMPUTE_OPERATION_AUTO("logical_or", _or, _determine_shape_binary_cast<bool>);


}