#pragma once

#include <iostream>
#include <vector>

namespace pixelpipes {

#define DEFAULT_CHUNK_SIZE (1 << 15)

class InputCompressionBuffer: public std::streambuf
{
	public:
		explicit InputCompressionBuffer(std::streambuf *src);

	protected:
		virtual int_type underflow();

	protected:
		std::streambuf* src_;
		std::vector<char_type> in_buffer_;
		std::vector<char_type> out_buffer_;

		size_t			bytes_in_;
		size_t			bytes_out_;
};

class InputCompressionStream: public std::istream {
public:
	explicit InputCompressionStream(std::streambuf& inbuf);
	explicit InputCompressionStream(std::istream& in);
private:
	InputCompressionBuffer isbuf_;
};

class OutputCompressionBuffer: public std::streambuf
{
	public:
		explicit OutputCompressionBuffer(std::streambuf* dest,
		                          size_t chunksize = DEFAULT_CHUNK_SIZE);
		virtual ~OutputCompressionBuffer();
		void init();

	protected:
		virtual int_type overflow(int_type c = traits_type::eof());
		int writeBlock(const char * data,
		               std::streamsize& uncompressed_length,
		               std::streamsize& length,
		               bool compressed,
		               uint32_t cksum);
		virtual int  sync();

	private:
		std::streambuf*  dest_;
		bool             write_cksums_;
		char*            in_buffer_;
		std::string      out_buffer_;
		size_t           chunksize_;

		size_t			bytes_in_;
		size_t			bytes_out_;

};

class OutputCompressionStream: public std::ostream
{
	public:
		explicit OutputCompressionStream(std::streambuf& outbuf, unsigned chunksize =
				DEFAULT_CHUNK_SIZE);
		explicit OutputCompressionStream(std::ostream& out, unsigned chunksize =
				DEFAULT_CHUNK_SIZE);
		void init() { osbuf_.init(); }
	private:
		OutputCompressionBuffer osbuf_;
};

} // namespace