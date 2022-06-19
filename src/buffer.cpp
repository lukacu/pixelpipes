

#include <memory>

#include <pixelpipes/buffer.hpp>
#include <pixelpipes/serialization.hpp>

namespace pixelpipes
{

    Shape FlatBuffer::shape() const
    {
        return ListType(CharIdentifier, size());
    }

    void Buffer::describe(std::ostream &os) const
    {
        os << "[Buffer of size " << size() << "]";
    }

    FlatBuffer::FlatBuffer(size_t size)
    {
        _data = ByteSequence(size);
    }

    FlatBuffer::FlatBuffer(const ByteSpan& data) : _data(data)
    {

    }

    const uchar* FlatBuffer::const_data() const
    {
        return _data.data();
    }


    uchar* FlatBuffer::data()
    {
        // TODO: hackish, but works
        return (uchar*) _data.data();
    }

    size_t FlatBuffer::size() const
    {
        return _data.size();
    }

    ReadonlySliceIterator FlatBuffer::read_slices() const
    {
        return ReadonlySliceIterator(const_data(), size());
    }

    WriteableSliceIterator FlatBuffer::write_slices()
    {
        return WriteableSliceIterator(data(), size());
    }

    String::String(std::string value) : value(value)
    {
        get();
    }

    size_t String::length() const {
        return value.size();
    }

    Shape String::shape() const {
        return ListType<char>(value.size());
    }

    TokenReference String::get(size_t index) const
    {
        return wrap(value[index]);
    }

    void String::describe(std::ostream &os) const
    {
        os << "[String token " << value << "]";
    }

    StringList::StringList()
    {
        _shape = ListType<char>(unknown).push(0);
    }

    StringList::StringList(View<std::string> list) : _list(list)
    {

        if (list.size() > 0) {
            Size _s = list[0].size();

            for (size_t i = 1; i < list.size(); i++) {
                _s = _s & Size(_list[i].size());
            }

            _shape = ListType<char>(_s).push(length());

        } else {
            _shape = ListType<char>(unknown).push(0);
        }

    }

    size_t StringList::length() const { return _list.size(); }

    Shape StringList::shape() const { return _shape; }

    const Span<std::string> StringList::get() const { return _list; }

    TokenReference StringList::get(size_t index) const { return create<String>(_list[index]); }

    PIXELPIPES_REGISTER_SERIALIZER(GetTypeIdentifier<String>(), "string",
        [](std::istream &source) -> TokenReference { return create<String>(read_t<std::string>(source)); },
        [](const TokenReference& v, std::ostream &drain) { write_t(drain, extract<std::string>(v)); }
    );

    PIXELPIPES_REGISTER_SERIALIZER(GetTypeIdentifier<StringList>(), "string_list",
        [](std::istream &source) -> TokenReference { return create<StringList>(read_sequence<std::string>(source)); },
        [](const TokenReference& v, std::ostream &drain) { write_sequence(drain, extract<StringListReference>(v)->get()); }
    );

    inline size_t write_buffer(const BufferReference& buffer, std::ostream& drain)
    {
        size_t total = 0;
        for (ReadonlySliceIterator it = buffer->read_slices(); (bool)(*it) ; it++)
        {
            drain.write((char *)(*it).data(), (*it).size());
            total += (*it).size();
        }
        return total;
    }

    inline ByteSequence read_buffer(std::istream& source)
    {
        ByteSequence data(read_t<size_t>(source));
        source.read((char *)data.data(), data.size());
        return data;
    }

    PIXELPIPES_REGISTER_SERIALIZER(GetTypeIdentifier<FlatBuffer>(), "buffer",
        [](std::istream &source) -> TokenReference { return create<FlatBuffer>(read_buffer(source)); },
        [](const TokenReference& v, std::ostream &drain) { write_buffer(std::move(extract<BufferReference>(v)), drain); }
    );

}
