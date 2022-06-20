#pragma once

#include <functional>
#include <memory>
#include <iterator>
#include <numeric>

#include <pixelpipes/type.hpp>
#include <pixelpipes/operation.hpp>
#include <pixelpipes/buffer.hpp>

namespace pixelpipes
{

    inline SizeSequence generate_strides(const Sizes& shape, size_t element) {

            size_t* strides = new size_t[shape.size()];
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

        virtual void describe(std::ostream &os) const override;

        virtual Shape shape() const override = 0;

        virtual size_t length() const override = 0;

        virtual size_t size() const override = 0;

        virtual size_t cell_size() const = 0;

        virtual TokenReference get(size_t i) const override = 0;

        virtual TokenReference get(const Sizes &i) const = 0;

        virtual ReadonlySliceIterator read_slices() const override = 0;

        virtual WriteableSliceIterator write_slices() override = 0;

        virtual const uchar *const_data() const override = 0;

        virtual uchar *data() override = 0;

        virtual SizeSequence strides() const = 0;
    };

    typedef Pointer<Tensor> TensorReference;

    template <typename T>
    class PIXELPIPES_API Vector;
    template <typename T>
    class PIXELPIPES_API Matrix;

    template <typename T, size_t N>
    class PIXELPIPES_API ArrayTensor : public Tensor
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
            return Shape(GetTypeIdentifier<T>(), _shape);
        }

        virtual size_t length() const override
        {
            return _shape[0];
        }

        virtual void describe(std::ostream &os) const override
        {
            os << "[Tensor]";
        }

        virtual size_t size() const override
        {
            return _data.size();
        }

        virtual size_t cell_size() const override
        {
            return sizeof(T);
        }

        virtual TokenReference get(const Sizes &index) const override
        {
            size_t o = get_offset(index);
            return create<ScalarToken<T>>(_data.at<T>(o));
        }

        virtual TokenReference get(size_t i) const override
        {

            if constexpr (N == 1)
            {
                return create<ScalarToken<T>>(_data.at<T>(i * sizeof(T)));
            }
            else
            {

                std::array<size_t, N> index;
                index.fill(0);
                index[0] = i;
                size_t o1 = get_offset(make_view(index));
                index[0] = i + 1;
                size_t o2 = get_offset(make_view(index));

                auto data = View<T>(reinterpret_cast<const T *>(std::data(_data) + o1), (o2 - o1) / sizeof(T));

                if constexpr (N == 2)
                {
                    return create<Vector<T>>(data);
                }
                else if constexpr (N == 3)
                {
                    return create<Matrix<T>>(_shape[1], _shape[2], data);
                }
                else
                {
                    auto shape = make_view(_shape, (size_t)1);
                    return create<ArrayTensor<T, N - 1>>(shape, data);
                }
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

        virtual const uchar *const_data() const override
        {
            return _data.data();
        }

        virtual SizeSequence strides() const override
        {
            return _strides;
        }

        virtual uchar *data() override
        {
            // TODO: hackish
            return (uchar *)_data.data();
        }

    protected:
        inline size_t get_offset(const Sizes &index) const
        {
            VERIFY(index.size() == _shape.size(), "Rank mismatch");
            size_t position = index[index.size() - 1];
            for (size_t i = 1; i < index.size(); i++)
            {
                position = position * _shape[i] + index[i - 1];
            }
            position *= sizeof(T);

            return position;
        }

        ByteSequence _data;
        SizeSequence _shape;
        SizeSequence _strides;
    };
    /*
        template <typename T, size_t N>
        class PIXELPIPES_API TensorView : public Tensor
        {
            PIXELPIPES_RTTI(TensorView<T, N>, Tensor)
        public:
            TensorView(const TensorReference& source, size_t offset, const Sizes &shape, const Sizes &strides) : _data(data)
            {

                static_assert(std::is_fundamental_v<T>, "Not a primitive type");
                static_assert(N > 0, "At least one dimension required");

                VERIFY(shape.size() == N, "Number of dimensions does not match");

                size_t s = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>()) * sizeof(T);

                VERIFY(data.size() == s, "Data length does not match its shape");

                _shape = shape;

                std::array<size_t, N> strides;
                std::array<size_t, N> index;

                for (size_t i = 0; i < N; i++)
                {
                    index.fill(0);
                    index[i] = 1;
                    strides[i] = get_offset(make_span(index));
                }
                _strides = SizeSequence(strides);
            }

            virtual ~TensorView() = default;

            virtual Shape shape() const
            {
                return Shape(GetTypeIdentifier<T>(), _shape);
            }

            virtual size_t length() const
            {
                return _shape[0];
            }

            virtual void describe(std::ostream &os) const
            {
                os << "[Tensor view]";
            }

            virtual size_t size() const
            {
                return _data.size();
            }

            virtual TokenReference get(const Sizes &index) const
            {
                size_t o = get_offset(index);
                return create<ScalarToken<T>>(_data.at<T>(o));
            }

            virtual TokenReference get(size_t i) const
            {

                if constexpr (N == 1)
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

                    if constexpr (N == 2)
                    {
                        return create<Vector<T>>(data);
                    }
                    else if constexpr (N == 3)
                    {
                        return create<Matrix<T>>(_shape[1], _shape[2], data);
                    }
                    else
                    {
                        auto shape = make_span(_shape, (size_t)1);
                        return create<ArrayTensor<T, N - 1>>(shape, data);
                    }
                }
            }

            virtual ReadonlySliceIterator read_slices() const
            {
                return ReadonlySliceIterator(_data->const_data(), size());
            }

            virtual WriteableSliceIterator write_slices()
            {
                return WriteableSliceIterator(_data->data(), size());
            }

            virtual const uchar *const_data() const
            {
                return _data->const_data();
            }

            virtual SizeSequence strides() const
            {
                return _strides;
            }

            virtual uchar *data()
            {
                return _data->data();
            }

        protected:
            inline size_t get_offset(const Sizes &index) const
            {
                VERIFY(index.size() == _shape.size(), "Rank mismatch");
                size_t position = _offset;
                for (size_t i = 0; i < index.size() - 1; i++)
                {
                    position = position * _shape[i + 1] + index[i];
                }
                position *= sizeof(T);

                return position;
            }

            TensorReference _data;
            size_t _offset;
            SizeSequence _shape;
            SizeSequence _strides;
        };
    */

    template <typename T>
    class PIXELPIPES_API Vector : public ArrayTensor<T, 1>
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
    class PIXELPIPES_API Matrix : public ArrayTensor<T, 2>
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
    inline TensorReference create_tensor(Shape s)
    {

        if (s.dimensions() == 1)
        {
            return create<Vector<T>>(s[0]);
        }
        else if (s.dimensions() == 2)
        {
            return create<Matrix<T>>(s[0], s[1]);
        }
        else if (s.dimensions() == 3)
        {
            return create<ArrayTensor<T, 3>>(SizeSequence(std::vector<size_t>(s.begin(), s.end())));
        }
        else if (s.dimensions() == 4)
        {
            return create<ArrayTensor<T, 4>>(SizeSequence(std::vector<size_t>(s.begin(), s.end())));
        }
        if (s.dimensions() == 5)
        {
            return create<ArrayTensor<T, 5>>(SizeSequence(std::vector<size_t>(s.begin(), s.end())));
        }
        if (s.dimensions() == 6)
        {
            return create<ArrayTensor<T, 6>>(SizeSequence(std::vector<size_t>(s.begin(), s.end())));
        }

        throw TypeException((Formatter() << "Unsupported tensor rank: " << s.dimensions()).str());
    }

    template <>
    inline Sequence<int> extract(const TokenReference &v)
    {
        VERIFY((bool)v, "Uninitialized token");

        Shape shape = v->shape();

        if (shape.is_scalar())
        {
            return Sequence<int>({extract<int>(v)});
        }

        if (shape.dimensions() != 1 ||
            shape.element() != GetTypeIdentifier<int>())
        {
            throw TypeException(
                "Unexpected token type: expected list of integers, got " + v->describe());
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

        if (shape.dimensions() != 1 ||
            shape.element() != GetTypeIdentifier<float>())
        {
            throw TypeException(
                "Unexpected token type: expected list of floats, got " + v->describe());
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

        if (shape.dimensions() != 1 ||
            shape.element() != GetTypeIdentifier<bool>())
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
    inline TokenReference wrap(const View<T> &v)
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

    template <typename V>
    inline TokenReference wrap(const std::vector<V> &v)
    {
        return wrap(make_span(v));
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
        VERIFY(v->is<Tensor>(), "Not a tensor");

        return cast<Tensor>(v);
    }

    class PIXELPIPES_API TensorList : public List
    {
    public:
        TensorList(const View<TensorReference> &tensors);

        ~TensorList();

        virtual Shape shape() const;

        virtual size_t length() const;

        virtual TokenReference get(size_t index) const;

        TensorList(const TensorList &);
        TensorList(TensorList &&);
        TensorList &operator=(const TensorList &);
        TensorList &operator=(TensorList &&);

    private:
        Sequence<TensorReference> _data;
        Shape _shape;
    };

    typedef Pointer<TensorList> TensorListReference;

    template <>
    inline TensorListReference extract(const TokenReference &v)
    {
        VERIFY((bool)v, "Uninitialized variable");
        VERIFY(v->is<TensorList>(), "Not an image type");

        return cast<TensorList>(v);
    }

    template <>
    inline TokenReference wrap(const Sequence<TensorReference> &v)
    {
        return create<TensorList>(v);
    }

}
