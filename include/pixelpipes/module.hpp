#pragma once

#include <string>
#include <functional>
#include <list>
#include <map>
#include <memory>
#include <type_traits>

#include <pixelpipes/base.hpp>

// https://stackoverflow.com/questions/53530566/loading-dll-in-windows-c-for-cross-platform-design

namespace pixelpipes {

class PIXELPIPES_API ModuleException : public BaseException {
public:
    ModuleException(std::string reason);
    ModuleException(const ModuleException& e) = default;
};

class Module;

typedef Pointer<Module> ModuleReference;

typedef Function<void(ModuleReference)> ModuleCallback;

class Module {
public:

    ~Module();

    Module(void* handle, std::string name);

    template<typename T> T symbol(std::string name) const {

        void* s = get_symbol(name);

        if (!s) return NULL;

        return (T) (s);

    }

    std::string name() const;

    static PIXELPIPES_API bool load(std::string_view);

    static ModuleReference context();

private:

    static std::map<std::string, ModuleReference>& modules();

    void* handle;

    std::string module_name;

    void* get_symbol(const std::string& name) const;

};

typedef void(*ModuleInitializer) ();

class AddDirect {
public:
    AddDirect(ModuleInitializer initializer);
};


#ifdef PIXELPIPES_BUILD_CORE
typedef AddDirect AddModuleInitializer;
#else
class AddModuleInitializer {
public:
    AddModuleInitializer(ModuleInitializer initializer);
};
#endif

#define PIXELPIPES_MODULE(N) \
std::list<pixelpipes::ModuleInitializer> & ( __pixelpipes_ ## N ## _module_initializers )() { static std::list<pixelpipes::ModuleInitializer> inits;  return inits; } \
pixelpipes::AddModuleInitializer::AddModuleInitializer(pixelpipes::ModuleInitializer initializer) { ( __pixelpipes_ ## N ## _module_initializers)().push_back(initializer); } \
extern "C" { \
const char* pixelpipes_module = STRINGIFY(N); \
void PIXELPIPES_MODULE_API pixelpipes_module_init() { \
    for (auto initializer : ( __pixelpipes_ ## N ## _module_initializers)()) initializer(); \
} }

}