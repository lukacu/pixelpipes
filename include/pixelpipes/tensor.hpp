#pragma once

#include <functional>
#include <memory>
#include <iterator>
#include <numeric>

#include <pixelpipes/type.hpp>
#include <pixelpipes/buffer.hpp>

namespace pixelpipes
{

    enum class DataType
    {
        Boolean,
        Char,
        Short,
        UnsignedShort,
        Integer,
        Float
    };

    inline SizeSequence generate_strides(const Sizes &shape, size_t element)
    {

        size_t *strides = new size_t[shape.size()];
        strides[shape.size() - 1] = element;
        for (size_t i = shape.size() - 1; i > 0; i--)
        {
            strides[i - 1] = strides[i] * shape[i];
        }
        return SizeSequence::claim(strides, shape.size());
    }

    class PIXELPIPES_API Tensor : public List, public Buffer
    {
        PIXELPIPES_RTTI(Tensor, List, Buffer)
    public:
        virtual ~Tensor() = default;

        virtual void describe(std::ostream &os) const override = 0;

        virtual Shape shape() const override = 0;

        virtual size_t length() const override = 0;

        virtual size_t size() const override = 0;

        virtual size_t cell_size() const = 0;

        virtual Type datatype() const = 0;

        virtual TokenReference get(size_t i) const override = 0;

        virtual TokenReference get(const Sizes &i) const = 0;

        virtual ReadonlySliceIterator read_slices() const override = 0;

        virtual WriteableSliceIterator write_slices() override = 0;

        virtual ByteView const_data() const override = 0;

        virtual ByteSpan data() override = 0;

        virtual SizeSequence strides() const = 0;
    };

    typedef Pointer<Tensor> TensorReference;

    template <typename T>
    class Scalar;
    template <typename T>
    class Vector;
    template <typename T>
    class Matrix;

    typedef Scalar<int> IntegerScalar;
    typedef Scalar<float> FloatScalar;
    typedef Scalar<bool> BooleanScalar;
    typedef Scalar<char> CharScalar;
    typedef Scalar<short> ShortScalar;
    typedef Scalar<ushort> UShortScalar;

    template <typename T>
    class Scalar : public Tensor
    {
        PIXELPIPES_RTTI(Scalar<T>, Tensor)

    public:
        Scalar(const T &value) : _value(value)
        {
            static_assert(std::is_fundamental_v<T>, "Not a primitive type");
        }

        virtual ~Scalar() = default;

        virtual Shape shape() const override
        {
            return Shape(GetType<T>(), SizeSequence({1}));
        }

        T get() const { return _value; }

        virtual size_t length() const override
        {
            return 1;
        }

        virtual void describe(std::ostream &os) const override
        {
            os << "[Scalar of " << details::TypeName<T>() << ": " << _value << "]";
        }

        virtual size_t size() const override
        {
            return sizeof(T);
        }

        virtual size_t cell_size() const override
        {
            return sizeof(T);
        }

        virtual Type datatype() const override
        {
            return GetType<T>();
        }

        virtual TokenReference get(const Sizes &index) const override
        {
            VERIFY(index.size() == 1, "Rank mismatch");
            VERIFY(index[0] == 0, "Index out of bounds");
            return create<Scalar<T>>(_value);
        }

        virtual TokenReference get(size_t i) const override
        {
            VERIFY(i == 0, "Index out of bounds");
            return create<Scalar<T>>(_value);
        }

        virtual ReadonlySliceIterator read_slices() const override
        {
            return ReadonlySliceIterator(const_data());
        }

        virtual WriteableSliceIterator write_slices() override
        {
            return WriteableSliceIterator(data());
        }

        virtual ByteView const_data() const override
        {
            return ByteView((uchar *)&_value, sizeof(T));
        }

        virtual SizeSequence strides() const override
        {
            return SizeSequence({sizeof(T)});
        }

        virtual ByteSpan data() override
        {
            return ByteSpan((uchar *)&_value, sizeof(T));
        }

    protected:
        T _value;
    };

    class PIXELPIPES_API TensorView : public Tensor
    {
        PIXELPIPES_RTTI(TensorView, Tensor)
    public:
        /*template <typename T>
        TensorView(const ByteSequence &&data, const Sizes &shape) : TensorView(data, 0, shape, generate_strides(shape, sizeof(T)))
        {
        }*/

        template <typename T>
        TensorView(const ByteSequence &&data, size_t offset, const Sizes &shape, const Sizes &strides)
        {

            static_assert(std::is_fundamental_v<T>, "Not a primitive type");

            VERIFY(shape.size() == strides.size(), "Size mismatch");

            _shape = SizeSequence(shape);
            _strides = SizeSequence(strides);

            _cell_size = sizeof(T);
            _cell_type = GetType<T>();

            _size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>()) * _cell_size;

            _data = ByteSpan((uchar *)data.data() + offset, data.size() - offset);

            _owner = new ByteSequence(std::move(data));
            _cleanup = ([](void *v) -> void
                        { delete static_cast<ByteSequence *>(v); });
        }

        TensorView(const TensorReference &source, const Sizes &shape);

        TensorView(const TensorReference &source, size_t offset, const Sizes &shape, const Sizes &strides);

        virtual ~TensorView();

        virtual Shape shape() const override;

        virtual size_t length() const override;

        virtual void describe(std::ostream &os) const override;

        virtual size_t size() const override;

        virtual size_t cell_size() const override;

        virtual Type datatype() const override;

        virtual TokenReference get(const Sizes &index) const override;

        virtual TokenReference get(size_t i) const override;

        virtual ReadonlySliceIterator read_slices() const override;

        virtual WriteableSliceIterator write_slices() override;

        virtual ByteView const_data() const override;

        virtual SizeSequence strides() const override;

        virtual ByteSpan data() override;

    protected:
        typedef void (*owner_cleanup)(void *);

        inline size_t get_offset(const Sizes &index) const
        {
            VERIFY(index.size() == _shape.size(), "Rank mismatch");
            size_t position = 0;
            for (size_t i = 0; i < index.size() - 1; i++)
            {
                position += _strides[i] * index[i];
            }
            // position *= cell_size();

            return position;
        }

        inline TokenReference get_scalar(size_t offset) const
        {
            switch (datatype())
            {
            case CharType:
                return create<CharScalar>(_data.at<uchar>(offset));
            case ShortType:
                return create<ShortScalar>(_data.at<short>(offset));
            case UnsignedShortType:
                return create<UShortScalar>(_data.at<ushort>(offset));
            case IntegerType:
                return create<IntegerScalar>(_data.at<int>(offset));
            case FloatType:
                return create<FloatScalar>(_data.at<float>(offset));
            default:
                return empty();
            }
        }

        ByteSpan _data;
        size_t _size;
        SizeSequence _shape;
        SizeSequence _strides;

        Type _cell_type;
        size_t _cell_size;

        void *_owner = nullptr;
        owner_cleanup _cleanup = nullptr;
    };

    template <>
    inline TokenReference wrap(const int v)
    {
        return create<IntegerScalar>(v);
    }

    template <>
    inline TokenReference wrap(const short v)
    {
        return create<ShortScalar>(v);
    }

    template <>
    inline TokenReference wrap(const ushort v)
    {
        return create<UShortScalar>(v);
    }

    template <>
    inline TokenReference wrap(const bool v)
    {
        return create<BooleanScalar>(v);
    }

    template <>
    inline TokenReference wrap(const char v)
    {
        return create<CharScalar>(v);
    }

    template <>
    inline TokenReference wrap(const float v)
    {
        return create<FloatScalar>(v);
    }

    template <typename T>
    inline bool _extract_scalar(const TokenReference &v, T *value)
    {
        if (v->is<Scalar<T>>())
        {
            *value = v->cast<Scalar<T>>()->get();
            return true;
        }
        if (v->is<ContainerToken<T>>())
        {
            *value = v->cast<ContainerToken<T>>()->get();
            return true;
        }
        if (v->is<Tensor>())
        {
            auto t = v->cast<Tensor>();
            if (t->shape().is_scalar() && t->shape().element() == GetType<T>())
            {
                *value = t->data().reinterpret<T>()[0];
                return true;
            }
        }
        return false;
    }

#define EXTRACT_SCALAR(V, T, O)              \
    {                                        \
        T value;                             \
        if (_extract_scalar<T>((V), &value)) \
        {                                    \
            return static_cast<O>(value);    \
        }                                    \
    }

    template <>
    inline int extract(const TokenReference &v)
    {
        VERIFY((bool)v, "Uninitialized token");

        EXTRACT_SCALAR(v, int, int);
        EXTRACT_SCALAR(v, short, int);
        EXTRACT_SCALAR(v, ushort, int);
        EXTRACT_SCALAR(v, char, int);
        EXTRACT_SCALAR(v, bool, int);

        throw TypeException("Unexpected token type: expected int, got " + v->describe());
    }

    template <>
    inline short extract(const TokenReference &v)
    {
        VERIFY((bool)v, "Uninitialized token");

        EXTRACT_SCALAR(v, short, short);
        EXTRACT_SCALAR(v, ushort, short);
        EXTRACT_SCALAR(v, char, short);
        EXTRACT_SCALAR(v, bool, short);

        throw TypeException("Unexpected token type: expected short, got " + v->describe());
    }

    template <>
    inline ushort extract(const TokenReference &v)
    {
        VERIFY((bool)v, "Uninitialized token");

        EXTRACT_SCALAR(v, ushort, ushort);
        EXTRACT_SCALAR(v, char, ushort);
        EXTRACT_SCALAR(v, bool, ushort);

        throw TypeException("Unexpected token type: expected ushort, got " + v->describe());
    }

    template <>
    inline bool extract(const TokenReference &v)
    {
        VERIFY((bool)v, "Uninitialized token");

        EXTRACT_SCALAR(v, bool, bool);
        EXTRACT_SCALAR(v, char, bool);
        EXTRACT_SCALAR(v, int, bool);
        EXTRACT_SCALAR(v, short, bool);
        EXTRACT_SCALAR(v, ushort, bool);
        EXTRACT_SCALAR(v, float, bool);

        throw TypeException("Unexpected token type: expected bool, got " + v->describe());
    }

    template <>
    inline char extract(const TokenReference &v)
    {
        VERIFY((bool)v, "Uninitialized token");

        EXTRACT_SCALAR(v, char, char);
        EXTRACT_SCALAR(v, bool, char);

        throw TypeException("Unexpected token type: expected char, got " + v->describe());
    }

    template <>
    inline float extract(const TokenReference &v)
    {
        VERIFY((bool)v, "Uninitialized token");

        EXTRACT_SCALAR(v, float, float);
        EXTRACT_SCALAR(v, int, float);
        EXTRACT_SCALAR(v, short, float);
        EXTRACT_SCALAR(v, ushort, float);
        EXTRACT_SCALAR(v, char, float);
        EXTRACT_SCALAR(v, bool, float);

        throw TypeException("Unexpected token type: expected float, got " + v->describe());
    }

    template <typename T, size_t N>
    class ArrayTensor : public Tensor
    {
        PIXELPIPES_RTTI(ArrayTensor<T, N>, Tensor)
    public:
        ArrayTensor(const Sizes &shape, const ByteSequence &&data) : _data(data)
        {

            static_assert(std::is_fundamental_v<T>, "Not a primitive type");
            static_assert(N > 0, "At least one dimension required");

            VERIFY(shape.size() == N, "Number of dimensions does not match");

            size_t s = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>()) * sizeof(T);

            VERIFY(data.size() == s, "Data length does not match its shape");

            _shape = shape;
            _strides = generate_strides(_shape, sizeof(T));
        }

        ArrayTensor(const Sizes &shape, const View<T> &data) : _data(data.template reinterpret<uchar>())
        {

            static_assert(std::is_fundamental_v<T>, "Not a primitive type");
            static_assert(N > 0, "At least one dimension required");

            VERIFY(shape.size() == N, "Number of dimensions does not match");

            size_t s = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());

            VERIFY(data.size() == s, "Data length does not match its shape");

            _shape = shape;
            _strides = generate_strides(_shape, sizeof(T));
        }

        ArrayTensor(const Sizes &shape)
        {

            static_assert(std::is_fundamental_v<T>, "Not a primitive type");
            static_assert(N > 0, "At least one dimension required");

            VERIFY(shape.size() == N, "Number of dimensions does not match");

            size_t s = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>()) * sizeof(T);

            _data = ByteSequence(s);
            _shape = shape;
            _strides = generate_strides(_shape, sizeof(T));
        }

        virtual ~ArrayTensor() = default;

        virtual Shape shape() const override
        {
            return Shape(GetType<T>(), _shape);
        }

        virtual size_t length() const override
        {
            return _shape[0];
        }

        virtual void describe(std::ostream &os) const override
        {
            Shape s = shape();

            os << "[Tensor of " << details::TypeName<T>() << " " << (size_t)s[0];
            for (size_t i = 1; i < s.rank(); i++)
            {
                os << " x " << (size_t)s[i];
            }
            os << "]";
        }

        virtual size_t size() const override
        {
            return _data.size();
        }

        virtual size_t cell_size() const override
        {
            return sizeof(T);
        }

        virtual Type datatype() const override
        {
            return GetType<T>();
        }

        virtual TokenReference get(const Sizes &index) const override
        {
            size_t o = get_offset(index);
            return create<Scalar<T>>(_data.at<T>(o));
        }

        virtual TokenReference get(size_t i) const override
        {

            if constexpr (N == 1)
            {
                return create<Scalar<T>>(_data.at<T>(i * sizeof(T)));
            }
            else
            {

                std::vector<size_t> index(_shape.size(), 0);
                index[0] = i;
                size_t offset = get_offset(make_span(index));

                auto ref = pixelpipes::cast<Tensor>(reference());

                return create<TensorView>(ref, offset, make_view(_shape, 1), make_view(_strides, 1));
            }
        }

        virtual ReadonlySliceIterator read_slices() const override
        {
            return ReadonlySliceIterator(const_data(), _shape, _strides, sizeof(T));
        }

        virtual WriteableSliceIterator write_slices() override
        {
            return WriteableSliceIterator(data(), _shape, _strides, sizeof(T));
        }

        virtual ByteView const_data() const override
        {
            return _data;
        }

        virtual SizeSequence strides() const override
        {
            return SizeSequence(_strides);
        }

        virtual ByteSpan data() override
        {
            return ByteSpan(_data);
        }

    protected:
        inline size_t get_offset(const Sizes &index) const
        {
            VERIFY(index.size() == _shape.size(), "Rank mismatch");
            size_t position = 0;
            for (size_t i = 0; i < index.size() - 1; i++)
            {
                position += _strides[i] * index[i];
            }
            return position;
        }

        ByteSequence _data;
        SizeSequence _shape;
        SizeSequence _strides;
    };

    template <typename T>
    class Vector : public ArrayTensor<T, 1>
    {
        PIXELPIPES_RTTI(Vector<T>, ArrayTensor<T, 1>)
    public:
        Vector(size_t length) : ArrayTensor<T, 1>(SizeSequence({length}))
        {
        }

        Vector(const View<T> &data) : ArrayTensor<T, 1>(SizeSequence({data.size()}), data)
        {
        }

        virtual void describe(std::ostream &os) const override
        {

            os << "[Vector of " << details::TypeName<T>() << ", length " << this->length() << "]";
        }

        using ArrayTensor<T, 1>::get;

        const Span<T> get() const { return Span<T>((T *)this->_data.data(), this->_data.size() / sizeof(T)); }
    };

    template <typename T>
    class Matrix : public ArrayTensor<T, 2>
    {
        PIXELPIPES_RTTI(Matrix<T>, ArrayTensor<T, 2>)
    public:
        Matrix(size_t rows, size_t cols) : ArrayTensor<T, 2>(SizeSequence({rows, cols}))
        {
        }

        Matrix(size_t rows, size_t cols, const View<T> &data) : ArrayTensor<T, 2>(SizeSequence({rows, cols}), data)
        {
        }

        virtual void describe(std::ostream &os) const override
        {
            auto shape = this->shape();
            os << "[Matrix of " << details::TypeName<T>() << ", " << (size_t)shape[0] << " x " << (size_t)shape[1] << "]";
        }
    };

    template <typename T>
    inline TensorReference create_tensor(const Sizes &s)
    {
        if (s.size() == 1)
        {
            if (s[0] == 1)
                return create<Scalar<T>>(0);
            else
                return create<Vector<T>>(s[0]);
        }
        else if (s.size() == 2)
        {
            return create<Matrix<T>>(s[0], s[1]);
        }
        else if (s.size() == 3)
        {
            return create<ArrayTensor<T, 3>>(s);
        }
        else if (s.size() == 4)
        {
            return create<ArrayTensor<T, 4>>(s);
        }
        if (s.size() == 5)
        {
            return create<ArrayTensor<T, 5>>(s);
        }
        if (s.size() == 6)
        {
            return create<ArrayTensor<T, 6>>(s);
        }

        throw TypeException((Formatter() << "Unsupported tensor rank: " << s.size()).str());
    }

    TensorReference PIXELPIPES_API create_tensor(Type element, Sizes sizes);

    TensorReference PIXELPIPES_API create_tensor(Shape s);

    void PIXELPIPES_API copy_tensor(const TensorReference &in, const TensorReference &out);

    template <>
    inline Sequence<int> extract(const TokenReference &v)
    {
        VERIFY((bool)v, "Uninitialized token");

        Shape shape = v->shape();

        if (shape.is_scalar())
        {
            return Sequence<int>({extract<int>(v)});
        }
        if (v->is<Vector<int>>())
        {
            return v->cast<Vector<int>>()->get();
        }
        if (v->is<List>())
        {
            return v->cast<List>()->elements<int>();
        }
        throw TypeException("Cannot unpack integer list");
    }

    template <>
    inline const std::vector<int> extract(const TokenReference &v)
    {
        auto l = extract<Sequence<int>>(v);
        return std::vector<int>(l.begin(), l.end());
    }

    template <>
    inline TokenReference wrap(const Span<int> v)
    {
        return create<Vector<int>>(v);
    }

    template <>
    inline Sequence<float> extract(const TokenReference &v)
    {
        VERIFY((bool)v, "Uninitialized token");

        Shape shape = v->shape();

        if (shape.is_scalar())
        {
            return Sequence<float>({extract<float>(v)});
        }
        if (v->is<Vector<float>>())
        {
            return v->cast<Vector<float>>()->get();
        }
        if (v->is<List>())
        {
            return v->cast<List>()->elements<float>();
        }
        throw TypeException("Cannot unpack float list");
    }

    template <>
    inline const std::vector<float> extract(const TokenReference &v)
    {
        auto l = extract<Sequence<float>>(v);
        return std::vector<float>(l.begin(), l.end());
    }

    template <>
    inline TokenReference wrap(const Span<float> &v)
    {
        return create<Vector<float>>(v);
    }

    template <>
    inline Sequence<bool> extract(const TokenReference &v)
    {
        VERIFY((bool)v, "Uninitialized token");

        Shape shape = v->shape();

        if (shape.is_scalar())
        {
            return Sequence<bool>({extract<bool>(v)});
        }

        if (shape.rank() != 1 ||
            shape.element() != GetType<bool>())
        {
            throw TypeException(
                "Unexpected token type: expected list of booleans, got " + v->describe());
        }
        if (v->is<Vector<bool>>())
        {
            return v->cast<Vector<bool>>()->get();
        }
        if (v->is<List>())
        {
            return v->cast<List>()->elements<bool>();
        }
        throw TypeException("Cannot unpack bool list");
    }

    template <>
    inline const std::vector<bool> extract(const TokenReference &v)
    {
        auto l = extract<Sequence<bool>>(v);
        return std::vector<bool>(l.begin(), l.end());
    }

    template <typename T>
    inline TokenReference wrap(View<T> v)
    {
        return create<Vector<T>>(v);
    }

    template <typename T>
    inline TokenReference wrap(const Sequence<T> &v)
    {
        return create<Vector<T>>(v);
    }

    template <>
    inline TokenReference wrap(const std::vector<bool> &v)
    {
        Sequence<bool> copy(v);
        return wrap(copy);
    }

    template <>
    inline TokenReference wrap(const std::vector<int> &v)
    {
        return wrap(make_view(v));
    }

    template <>
    inline TokenReference wrap(const std::vector<float> &v)
    {
        return wrap(make_view(v));
    }

    template <>
    inline TokenReference wrap(const std::vector<uchar> &v)
    {
        return wrap(make_view(v));
    }

    template <char>
    inline TokenReference wrap(const std::vector<char> &v)
    {
        return wrap(make_view(v));
    }

    template <>
    inline TokenReference wrap(const std::vector<short> &v)
    {
        return wrap(make_view(v));
    }

    template <>
    inline TokenReference wrap(const std::vector<ushort> &v)
    {
        return wrap(make_view(v));
    }

    typedef Vector<float> FloatVector;
    typedef Vector<int> IntegerVector;
    typedef Vector<bool> BooleanVector;

    typedef Matrix<float> FloatMatrix;
    typedef Matrix<int> IntegerMatrix;
    typedef Matrix<bool> BooleanMatrix;

    typedef Pointer<Tensor> TensorReference;

    template <>
    inline TensorReference extract(const TokenReference &v)
    {
        VERIFY((bool)v, "Uninitialized token");

        if (v->is<Tensor>())
        {
            return cast<Tensor>(v);
        }

        if (v->is<List>())
        {
            Shape s = v->shape();

            if (!s.is_fixed())
                throw TypeException("Cannot convert to tensorr");

            if (s.element() == CharType || s.element() == ShortType || s.element() == UnsignedShortType || s.element() == IntegerType || s.element() == FloatType)
            {

                // TODO: convert
            }
            else
            {
                throw TypeException("Cannot convert to tensorr");
            }
        }

        throw TypeException("Not a tensor");
    }

    inline Size get_size(const TokenReference &token) {
        if (_IS_PLACEHOLDER(token)) return unknown;
        return extract<int>(token);
    }

    template<>
    inline Size extract(const TokenReference &v) {
        return get_size(v);
    }

    template<>
    inline TokenReference wrap(Size v) {
        if (v == unknown) return create<Placeholder>(IntegerType);
        return create<IntegerScalar>(v);
    }

#ifdef XTENSOR_TENSOR_HPP

    template <typename T>
    xt::xarray_adaptor<xt::xbuffer_adaptor<T *, xt::no_ownership>, xt::layout_type::dynamic, std::vector<size_t>>
    wrap_xtensor(const TensorReference &tr)
    {
        // TODO: deterine if tensor is contiguous, change adapt call in this case
        auto ts = tr->shape();

        std::vector<size_t> _shape(ts.rank());
        std::vector<size_t> _strides(ts.rank());
        for (size_t i = 0; i < ts.rank(); i++)
        {
            _shape[i] = ts[i];
            _strides[i] = tr->strides()[i] / sizeof(T);
        }

        if constexpr (std::is_same_v<T, uchar>)
        {
            if (GetType<char>() != tr->datatype())
                throw TypeException("Tensor type mismatch, use casting");
            return xt::adapt(tr->data().reinterpret<uchar>().data(), ts.size(), xt::no_ownership(), _shape, _strides);
        }
        else if constexpr (std::is_same_v<T, short>)
        {
            if (GetType<short>() != tr->datatype())
                throw TypeException("Tensor type mismatch, use casting");
            return xt::adapt(tr->data().reinterpret<short>().data(), ts.size(), xt::no_ownership(), _shape, _strides);
        }
        else if constexpr (std::is_same_v<T, ushort>)
        {
            if (GetType<ushort>() != tr->datatype())
                throw TypeException("Tensor type mismatch, use casting");
            return xt::adapt(tr->data().reinterpret<ushort>().data(), ts.size(), xt::no_ownership(), _shape, _strides);
        }
        else if constexpr (std::is_same_v<T, int>)
        {
            if (GetType<int>() != tr->datatype())
                throw TypeException("Tensor type mismatch, use casting");
            return xt::adapt(tr->data().reinterpret<int>().data(), ts.size(), xt::no_ownership(), _shape, _strides);
        }
        else if constexpr (std::is_same_v<T, float>)
        {
            if (GetType<float>() != tr->datatype())
                throw TypeException("Tensor type mismatch, use casting");
            return xt::adapt(tr->data().reinterpret<float>().data(), ts.size(), xt::no_ownership(), _shape, _strides);
        }
        else if constexpr (std::is_same_v<T, bool>)
        {
            if (GetType<bool>() != tr->datatype())
                throw TypeException("Tensor type mismatch, use casting");
            return xt::adapt(tr->data().reinterpret<bool>().data(), ts.size(), xt::no_ownership(), _shape, _strides);
        }
        else
        {
            throw TypeException("Unsupported tensor type");
        }
    }

#endif

    PIXELPIPES_CONVERT_ENUM(DataType)

}
