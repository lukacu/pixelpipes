#pragma once

#ifdef _WIN32
#ifdef PIXELPIPES_BUILD_CORE
#define PIXELPIPES_API __declspec(dllexport)
#else
#define PIXELPIPES_API __declspec(dllimport)
#endif
#define PIXELPIPES_INTERNAL
#define PIXELPIPES_MODULE_API __declspec(dllexport)
#define PIXELPIPES_TYPE_API
#else
/* allow use of -fvisibility=hidden -fvisibility-inlines-hidden */
#define PIXELPIPES_API __attribute__((visibility("default")))
#define PIXELPIPES_INTERNAL __attribute__((visibility("hidden")))
#define PIXELPIPES_MODULE_API __attribute__((visibility("default")))
#define PIXELPIPES_TYPE_API __attribute__((visibility("default")))
#endif