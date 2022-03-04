
#define PIXELPIPES_BUILD_CORE 1

#include <pixelpipes/module.hpp>

#include <mutex>

#ifdef _MSC_VER
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace pixelpipes {

typedef void (*ModuleInitHook)();

ModuleException::ModuleException(std::string reason): BaseException(reason) {}

std::map<std::string, SharedModule>& Module::modules() {
    static std::map<std::string, SharedModule> n;
    return n;
}

static std::recursive_mutex module_load_lock;

static SharedModule load_context;

SharedModule Module::context() {

    std::lock_guard<std::recursive_mutex> lock(module_load_lock);

    return load_context;
}

bool Module::load(const std::string& libname) {

    std::lock_guard<std::recursive_mutex> lock(module_load_lock);

#ifdef _MSC_VER
    auto fullname = libname + ".dll";
#else
    auto fullname = libname + ".so";
#endif

    DEBUGMSG("Loading module %s \n", fullname.c_str());

    if (modules().find(libname) != modules().end()) {
        return true;
    }

#ifdef _MSC_VER
    auto handle = LoadLibrary(_T(fullname.c_str()));
#else
    auto handle = dlopen(fullname.c_str(), RTLD_DEEPBIND | RTLD_NOW);
#endif

    if (!handle) {
        DEBUGMSG("Cannot load module: %s \n", dlerror());
        return false;
    }

    SharedModule mod = std::shared_ptr<Module>(new Module(handle, libname));

    modules().insert(std::pair<std::string, SharedModule>(libname, mod));
    
    auto init = mod->symbol<ModuleInitHook>("pixelpipes_module_init");

    load_context = mod;

    if (init) {
        init();
    }

    load_context.reset();

    return true;
}

void* Module::get_symbol(const std::string& name) const {

#ifdef _MSC_VER
    void* sym = GetProcAddress( (HMODULE) handle, _T(name.c_str()) );
#else
    void* sym = dlsym(handle, name.c_str());
#endif

    if (!sym) {
        DEBUGMSG("Cannot find symbol %s: %s \n", name.c_str(), dlerror());
    }

    return sym;
}


Module::Module(void* handle, std::string name): handle(handle), module_name(name) {}

Module::~Module() {}

std::string Module::name() const {
    return module_name;
}

AddDirect::AddDirect(ModuleInitializer initializer) {

    initializer();

}

}