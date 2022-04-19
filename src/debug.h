#pragma once

#include <cstdio>

#ifdef PIXELPIPES_DEBUG
#define DEBUGMSG(...)        \
    {                        \
        std::printf(__VA_ARGS__); \
    }
#else
#define DEBUGMSG(...) \
    {                 \
    }
#endif

#define PRINTMSG(...)        \
    {                        \
        std::printf(__VA_ARGS__); \
    }

