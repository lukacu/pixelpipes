#pragma once

#include <cstdint>
#include <memory>

namespace pixelpipes {

class xorshift32
{
  private:
    uint32_t m_seed;

  public:
    xorshift32(uint32_t _Seed = 1) : m_seed(_Seed) {}
    void seed(uint32_t _Seed) { m_seed = _Seed; }
    void discard(uint64_t z);
    uint32_t operator()(void);
};

void xorshift32::discard(uint64_t z)
{
    while (z--)
        this->operator()();
}

uint32_t xorshift32::operator()(void)
{
    m_seed ^= m_seed << 13;
    m_seed ^= m_seed >> 17;
    m_seed ^= m_seed << 15;
    return m_seed;
}

}

