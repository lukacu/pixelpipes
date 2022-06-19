#pragma once

#include <functional>
#include <memory>
#include <iterator>
#include <vector>

#include <pixelpipes/type.hpp>
#include <pixelpipes/operation.hpp>

namespace pixelpipes
{
    typedef Sequence<uchar> ByteSequence;
    typedef Span<uchar> ByteSpan;
    typedef View<uchar> ByteView;

    template <typename T, typename C>
    class PIXELPIPES_API SliceIterator : public std::iterator<std::input_iterator_tag, C>
    {
    public:
        using value_type = T;
        using pointer_type = T *;
        using slice = C;

        SliceIterator() : SliceIterator(nullptr, 0){};
        SliceIterator(pointer_type data, size_t length) : SliceIterator(data, make_view(Sequence<size_t>({length})), make_view(Sequence<size_t>({1})), 1)
        {
        }

        SliceIterator(pointer_type data, const Sizes &shape, const Sizes &strides, size_t element) : _data(data)
        {
            VERIFY(strides.size() == shape.size(), "Size mismatch");
            
            size_t stride = element;
            size_t i;
            _length = stride;
            for (i = strides.size(); i > 0; i--)
            {
                if (stride == strides[i - 1])
                {
                    stride = stride * shape[i - 1];
                    _length = stride;
                }
                else
                    break;
            }

            if (i == 0)
            {
                _strides = SizeSequence({0});
                _shape = SizeSequence({0});
            }
            else
            {
                _strides = SizeSequence(strides.data(), i);
                _shape = SizeSequence(shape.data(), i);
            }

            _position = SizeSequence::repeat(_strides.size(), 0);
            _current = slice{_data, _length};
        }

        inline SliceIterator &operator++()
        {
            increment();
            return *this;
        }
        inline SliceIterator operator++(int)
        {
            increment();
            return *this;
        }
        inline bool operator==(const SliceIterator &rhs) const { return _current.data() == rhs._current.data() && _current.size() == rhs._current.size(); }
        inline bool operator!=(const SliceIterator &rhs) const { return !(this->operator==(rhs)); }
        inline const slice operator*() { return _current; }
        inline const slice operator->() { return _current; }

    protected:
        inline void increment()
        {

            if (_data == nullptr || _position[0] > _shape[0])
                return;

            for (int i = _position.size(); i > 0; i--)
            {
                _position[i - 1]++;
                if (_position[i - 1] == _shape[i - 1] && i != 1)
                    _position[i - 1] = 0;
                else
                    break;
            }

            if (_position[0] < _shape[0])
            {

                size_t offset = 0;
                for (size_t i = 0; i < _position.size(); i++)
                    offset += _strides[i] * _position[i];
                _current = slice{_data + offset, _length};
            }
            else
            {
                _current = slice{nullptr, 0};
            }
        }

        slice _current;
        pointer_type _data;
        size_t _length;
        SizeSequence _strides;
        SizeSequence _shape;
        SizeSequence _position;
    };

    typedef SliceIterator<const uchar, View<uchar>> ReadonlySliceIterator;
    typedef SliceIterator<uchar, Span<uchar>> WriteableSliceIterator;

    class PIXELPIPES_API Buffer : public virtual Token
    {
        PIXELPIPES_RTTI(Buffer, Token)
    public:
        virtual void describe(std::ostream &os) const;

        virtual ~Buffer() = default;

        virtual size_t size() const = 0;

        virtual ReadonlySliceIterator read_slices() const = 0;

        virtual WriteableSliceIterator write_slices() = 0;

        virtual const uchar *const_data() const = 0;

        virtual uchar *data() = 0;
    };

    typedef Pointer<Buffer> BufferReference;

    class PIXELPIPES_API FlatBuffer : public Buffer
    {
        PIXELPIPES_RTTI(FlatBuffer, Buffer)
    public:
        virtual Shape shape() const;

        FlatBuffer(const ByteSpan &data);

        FlatBuffer(size_t size);

        virtual ~FlatBuffer() = default;

        virtual size_t size() const;

        virtual ReadonlySliceIterator read_slices() const;

        virtual WriteableSliceIterator write_slices();

        virtual const uchar *const_data() const;

        virtual uchar *data();

    private:
        ByteSequence _data;
    };

    template <>
    inline BufferReference extract(const TokenReference &v)
    {
        VERIFY((bool)v, "Uninitialized token");
        VERIFY(v->is<Buffer>(), "Cannot unpack buffer");
        return cast<Buffer>(v);
    }

    // Extend buffer interface?
    class PIXELPIPES_API String : public List
    {
        PIXELPIPES_RTTI(String, List)
    public:
        String(std::string value);

        ~String() = default;

        virtual Shape shape() const;

        virtual TokenReference get(size_t index) const;

        virtual size_t length() const;

        virtual void describe(std::ostream &os) const;

        inline std::string get() const
        {
            return value;
        }

    protected:
        std::string value;
    };

    template <>
    inline std::string extract(const TokenReference &v)
    {
        VERIFY((bool)v, "Uninitialized token");

        if (v->is<String>())
            return v->cast<String>()->get();

        if (v->is<List>())
        {

            Shape shape = v->shape();

            if (shape.dimensions() != 1 || shape.element() != CharIdentifier)
            {
                throw TypeException("Unexpected token type: expected list of chars, got " + v->describe());
            }

            Sequence<char> chars = (v)->cast<List>()->elements<char>();
            return std::string(chars.begin(), chars.end());
        }

        throw TypeException("Cannot unpack string");
    }

    template <>
    inline TokenReference wrap(const std::string v)
    {
        return create<String>(v);
    }

    /// String list

    class PIXELPIPES_API StringList : public List
    {
        PIXELPIPES_RTTI(StringList, List)
    public:
        StringList();
        StringList(View<std::string> list);
        ~StringList() = default;

        virtual size_t length() const;

        virtual Shape shape() const;

        virtual const Span<std::string> get() const;

        virtual TokenReference get(size_t index) const;

        template <class T>
        const Sequence<T> elements() const
        {
            if constexpr (std::is_same<T, char>::value)
            {
                return _list;
            }
            else
            {
                return List::elements<T>();
            }
        }

    private:
        Sequence<std::string> _list;

        Shape _shape;
    };

    typedef Pointer<StringList> StringListReference;

    template <>
    inline TokenReference wrap(const std::vector<std::string> &v)
    {
        return create<StringList>(make_view(v));
    }

    template <>
    inline TokenReference wrap(const Span<std::string> &v)
    {
        return create<StringList>(v);
    }

    template <>
    inline Sequence<std::string> extract(const TokenReference &v)
    {
        VERIFY((bool)v, "Uninitialized token");

        if (v->is<StringList>())
            return v->cast<StringList>()->get();

        if (v->is<List>())
        {
            return (v)->cast<List>()->elements<std::string>();
        }

        throw TypeException("Cannot unpack string list");
    }

    template <>
    inline StringListReference extract(const TokenReference &v)
    {
        VERIFY((bool)v, "Uninitialized token");
        VERIFY(v->is<StringList>(), "Not a sting list");
        return cast<StringList>(v);
    }

    template <typename A, typename B>
    void copy_buffer(const A &source, B &destination)
    {

        ReadonlySliceIterator sit;
        WriteableSliceIterator dit;

        size_t ssize, dsize;

        if constexpr (std::is_base_of_v<ByteView, A>)
        {
            sit = ReadonlySliceIterator(source.data(), source.size());
            ssize = source.size();
        }
        else
        {
            sit = source->read_slices();
            ssize = source->size();
        }

        if constexpr (std::is_base_of_v<ByteSpan, B>)
        {
            dit = WriteableSliceIterator(destination.data(), destination.size());
            dsize = destination.size();
        }
        else
        {
            dit = destination->write_slices();
            dsize = destination->size();
        }

        VERIFY(ssize == dsize, "Buffer size does not match");

        size_t soffset = 0;
        size_t doffset = 0;

        while (true)
        {
            size_t slen = (*sit).size() - soffset;
            size_t dlen = (*dit).size() - doffset;
            size_t length = (std::min)(slen, dlen);

            if (length == 0)
                break;

            // TODO: remove cast
            std::memcpy((void *)((*dit).data() + doffset), (*sit).data() + soffset, length);

            doffset += length;
            soffset += length;

            if (slen == length)
            {
                soffset = 0;
                sit++;
            }

            if (dlen == length)
            {
                doffset = 0;
                dit++;
            }
        }
    }

}