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
};

class Module;

typedef std::shared_ptr<Module> SharedModule;

typedef Function<void(SharedModule)> ModuleCallback;

class Module {
public:

    ~Module();

    template<typename T> T symbol(std::string name) const {

        void* s = get_symbol(name);

        if (!s) return NULL;

        return (T) (s);

    }

    std::string name() const;

    static PIXELPIPES_API bool load(std::string_view);

    static SharedModule context();

private:

    static std::map<std::string, SharedModule>& modules();

    void* handle;

    std::string module_name;

    Module(void* handle, std::string name);

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
void pixelpipes_module_init() { \
    for (auto initializer : ( __pixelpipes_ ## N ## _module_initializers)()) initializer(); \
} }

}