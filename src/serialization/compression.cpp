/*@author Merder Kim <hoxnox@gmail.com> 
 *@date 20130117 22:25:30 */

#include <stdexcept>
#include <cstring>
#include <cstdio>
#include <stdint.h>
#include <string>

#include <snappy.h>

#include "compression.hpp"

#include "crc32c.hpp"

#define CURRENT_BYTE_ORDER       (*(int *)"\x01\x02\x03\x04")
#define LITTLE_ENDIAN_BYTE_ORDER 0x04030201
#define BIG_ENDIAN_BYTE_ORDER    0x01020304
#define PDP_ENDIAN_BYTE_ORDER    0x02010403

#define IS_LITTLE_ENDIAN (CURRENT_BYTE_ORDER == LITTLE_ENDIAN_BYTE_ORDER)
#define IS_BIG_ENDIAN    (CURRENT_BYTE_ORDER == BIG_ENDIAN_BYTE_ORDER)
#define IS_PDP_ENDIAN    (CURRENT_BYTE_ORDER == PDP_ENDIAN_BYTE_ORDER)

// Forward declaration

template<typename T>
struct LittleEndian;

template<typename T>
struct BigEndian;

// Little-Endian template

#pragma pack(push,1)
template<typename T>
struct LittleEndian
{
    union
    {
        unsigned char bytes[sizeof(T)];
        T raw_value;
    };

    LittleEndian(T t = T())
    {
        operator =(t);
    }

    LittleEndian(const char * bytes, const size_t bytesz)
        : raw_value(0)
    {
        if(bytesz < sizeof(T))
            return;
        for(unsigned i = 0; i < sizeof(T); ++i)
            this->bytes[sizeof(T) - 1 - i] = bytes[i];
    }

    LittleEndian(const LittleEndian<T> & t)
    {
        raw_value = t.raw_value;
    }

    LittleEndian(const BigEndian<T> & t)
    {
        for (unsigned i = 0; i < sizeof(T); i++)
            bytes[i] = t.bytes[sizeof(T)-1-i];
    }

    operator const T() const
    {
        T t = T();
        for (unsigned i = 0; i < sizeof(T); i++)
            t |= T(bytes[i]) << (i << 3);
        return t;
    }

    const T operator = (const T t)
    {
        for (unsigned i = 0; i < sizeof(T); i++)
            bytes[i] = t >> (i << 3);
        return t;
    }

    // operators

    const T operator += (const T t)
    {
        return (*this = *this + t);
    }

    const T operator -= (const T t)
    {
        return (*this = *this - t);
    }

    const T operator *= (const T t)
    {
        return (*this = *this * t);
    }

    const T operator /= (const T t)
    {
        return (*this = *this / t);
    }

    const T operator %= (const T t)
    {
        return (*this = *this % t);
    }

    LittleEndian<T> operator ++ (int)
    {
        LittleEndian<T> tmp(*this);
        operator ++ ();
        return tmp;
    }

    LittleEndian<T> & operator ++ ()
    {
        for (unsigned i = 0; i < sizeof(T); i++)
        {
            ++bytes[i];
            if (bytes[i] != 0)
                break;
        }
        return (*this);
    }

    LittleEndian<T> operator -- (int)
    {
        LittleEndian<T> tmp(*this);
        operator -- ();
        return tmp;
    }

    LittleEndian<T> & operator -- ()
    {
        for (unsigned i = 0; i < sizeof(T); i++)
        {
            --bytes[i];
            if (bytes[i] != (T)(-1))
                break;
        }
        return (*this);
    }
};
#pragma pack(pop)

// Big-Endian template

#pragma pack(push,1)
template<typename T>
struct BigEndian
{
    union
    {
        unsigned char bytes[sizeof(T)];
        T raw_value;
    };

    BigEndian(T t = T())
    {
        operator =(t);
    }

    BigEndian(const char * bytes, const size_t bytesz)
        : raw_value(0)
    {
        if(bytesz < sizeof(T))
            return;
        for(unsigned i = 0; i < sizeof(T); ++i)
            this->bytes[i] = bytes[i];
    }

    BigEndian(const BigEndian<T> & t)
    {
        raw_value = t.raw_value;
    }

    BigEndian(const LittleEndian<T> & t)
    {
        for (unsigned i = 0; i < sizeof(T); i++)
            bytes[i] = t.bytes[sizeof(T)-1-i];
    }

    operator const T() const
    {
        T t = T();
        for (unsigned i = 0; i < sizeof(T); i++)
            t |= T(bytes[sizeof(T) - 1 - i]) << (i << 3);
        return t;
    }

    const T operator = (const T t)
    {
        for (unsigned i = 0; i < sizeof(T); i++)
            bytes[sizeof(T) - 1 - i] = t >> (i << 3);
        return t;
    }

    // operators

    const T operator += (const T t)
    {
        return (*this = *this + t);
    }

    const T operator -= (const T t)
    {
        return (*this = *this - t);
    }

    const T operator *= (const T t)
    {
        return (*this = *this * t);
    }

    const T operator /= (const T t)
    {
        return (*this = *this / t);
    }

    const T operator %= (const T t)
    {
        return (*this = *this % t);
    }

    BigEndian<T> operator ++ (int)
    {
        BigEndian<T> tmp(*this);
        operator ++ ();
        return tmp;
    }

    BigEndian<T> & operator ++ ()
    {
        for (unsigned i = 0; i < sizeof(T); i++)
        {
            ++bytes[sizeof(T) - 1 - i];
            if (bytes[sizeof(T) - 1 - i] != 0)
                break;
        }
        return (*this);
    }

    BigEndian<T> operator -- (int)
    {
        BigEndian<T> tmp(*this);
        operator -- ();
        return tmp;
    }

    BigEndian<T> & operator -- ()
    {
        for (unsigned i = 0; i < sizeof(T); i++)
        {
            --bytes[sizeof(T) - 1 - i];
            if (bytes[sizeof(T) - 1 - i] != (T)(-1))
                break;
        }
        return (*this);
    }
};
#pragma pack(pop)

typedef LittleEndian<uint8_t>  u8le;
typedef LittleEndian<uint16_t> u16le;
typedef LittleEndian<uint32_t> u32le;

typedef BigEndian<uint8_t>  u8be;
typedef BigEndian<uint16_t> u16be;
typedef BigEndian<uint32_t> u32be;

namespace pixelpipes {
/*
const char Config::magic[] = {'s', 'n', 'a', 'p', 'p', 'y', 0};
const int Config::magic_sz = sizeof(Config::magic);
*/

InputCompressionBuffer::InputCompressionBuffer(std::streambuf *src)
	: src_(src)
{
	/*char source_magic[Config::magic_sz];
	std::streamsize nread = src_->sgetn(source_magic, Config::magic_sz);
	if (memcmp(Config::magic, source_magic, Config::magic_sz))
		throw std::runtime_error("InputCompressionBuffer - bad magic number");*/
	this->setg(0, 0, 0);
}

InputCompressionBuffer::int_type InputCompressionBuffer::underflow()
{
	char header[7];
	if (src_->sgetn(header, 7) != 7)
		return EOF;

	bool compressed = true;
	if(header[0] == 0)
		compressed = false;
	BigEndian<uint16_t> len(&header[1], 2);
	uint32_t cksum = *reinterpret_cast<uint32_t*>(&header[3]);

	if (!len)
		return EOF;

	// expect, that the size of compressed data is less then uncompressed
	if (len > DEFAULT_CHUNK_SIZE) 
		return EOF;
	if (len > in_buffer_.size())
		in_buffer_.resize(len);

	int rs;
	if (src_->sgetn(reinterpret_cast<char*>(&in_buffer_[0]), len) != len)
		return EOF;

	size_t uncompressed_len = len;
	if(compressed)
	{
		if (!snappy::GetUncompressedLength(&in_buffer_[0], len, &uncompressed_len)
			|| !uncompressed_len)
			return EOF;
	}
	out_buffer_.resize(uncompressed_len);
	if(compressed)
	{
		if (!snappy::RawUncompress(&in_buffer_[0], len, &out_buffer_[0]))
			return EOF;
	}
	else
	{
		memcpy(&out_buffer_[0], &in_buffer_[0], uncompressed_len);
	}

	this->setg(&out_buffer_[0], &out_buffer_[0],
			   &out_buffer_[0] + out_buffer_.size());
	return traits_type::to_int_type(*(this->gptr()));
}

InputCompressionStream::InputCompressionStream(std::streambuf& inbuf)
	: isbuf_(&inbuf)
	, std::istream(&isbuf_)
{
}

InputCompressionStream::InputCompressionStream(std::istream& in)
	: isbuf_(in.rdbuf())
	, std::istream(&isbuf_)
{
}

/**@class OutputCompressionBuffer
 * @brief Snappy compressed output streambuf.
 *
 * Usage:
 * @example example.cpp*/

/**@brief Construct compressed streambuf, based on other streambuf*/
OutputCompressionBuffer::OutputCompressionBuffer(std::streambuf* dest, size_t chunksize)
	: dest_(dest)
	, write_cksums_(true)
	, in_buffer_(new char[chunksize])
	, chunksize_(chunksize)
{
	this->init();
}

/**@brief Sync and delete buffer on destruction*/
OutputCompressionBuffer::~OutputCompressionBuffer()
{
	this->sync();
	delete[] in_buffer_;
}

/**@brief Set boundaries of the controlled output sequence*/
void OutputCompressionBuffer::init()
{
	this->setp(in_buffer_, in_buffer_ + chunksize_ - 1);
}

/**@override std::streambuf::overflow(int)*/
OutputCompressionBuffer::int_type OutputCompressionBuffer::overflow(OutputCompressionBuffer::int_type c)
{
	if (!pptr())
		return EOF;
	if (c != EOF) {
		*pptr() = c;
		pbump(1);
	}
	if (sync() == -1) {
		return EOF;
	}
	return c;
}

/**@override std::streambuf::sync()
 * @brief Flush data to dest_*/
int OutputCompressionBuffer::sync()
{
	if (!pptr())
		return -1;

	std::streamsize uncompressed_len = pptr()-pbase();
	if (!uncompressed_len)
		return 0;

	uint32_t crc32c = write_cksums_ ? crc32c_masked(in_buffer_, uncompressed_len) : 0;


	char* compressed = new char[snappy::MaxCompressedLength(uncompressed_len)];
	size_t compressed_len_sz;
	snappy::RawCompress(in_buffer_, uncompressed_len, compressed, &compressed_len_sz);

	// use uncompressed input if less than 12.5% compression
	if (compressed_len_sz >= (uncompressed_len - (uncompressed_len / 8))) {
		delete [] compressed;
		return writeBlock(in_buffer_, uncompressed_len, uncompressed_len, false, crc32c);
	}
    std::streamsize compressed_len = static_cast<std::streamsize>(compressed_len_sz);
	int rs = writeBlock(compressed, uncompressed_len, compressed_len, true, crc32c);
	delete [] compressed;
	return rs;
}

int OutputCompressionBuffer::writeBlock(const char * data, std::streamsize& uncompressed_len, std::streamsize& length, bool compressed, uint32_t cksum)
{
	BigEndian<uint16_t> len((uint16_t)length);
	BigEndian<uint32_t> cksum_be(cksum);
	if (dest_->sputc(compressed ? 1 : 0) == EOF)
		return -1;
	if (dest_->sputn(reinterpret_cast<const char*>(&len), 2) != 2)
		return -1;
	if (dest_->sputn(reinterpret_cast<const char*>(&cksum_be), 4) != 4)
		return -1;
	if (dest_->sputn(&data[0], length) != length)
		return -1;
	pbump(-uncompressed_len);
	return uncompressed_len;
}

/**@brief You can create compressed stream over every stream based on std::streambuf
 * @param chunksize The size of chunks*/
OutputCompressionStream::OutputCompressionStream(std::streambuf& outbuf, unsigned chunksize)
	: osbuf_(&outbuf, chunksize)
	, std::ostream(&osbuf_)
{
}

/**@brief You can create compressed stream over every stream based on std::ostream
 * @param chunksize The size of chunks*/
OutputCompressionStream::OutputCompressionStream(std::ostream& out, unsigned chunksize)
	: osbuf_(out.rdbuf(), chunksize)
	, std::ostream(&osbuf_)
{
}

} // namespace snappy
