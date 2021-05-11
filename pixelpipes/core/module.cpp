

#include <pixelpipes/module.hpp>

#ifdef _MSC_VER
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace pixelpipes {

ModuleException::ModuleException(std::string reason): BaseException(reason) {}

std::list<std::function<void(Module&)>>& Module::callbacks() {
    static std::list<std::function<void(Module&)>> n;
    return n;
}

std::map<std::string, Module>& Module::modules() {
    static std::map<std::string, Module> n;
    return n;
}

bool Module::load(const std::string& libname) {

    DEBUGMSG("Loading module %s \n", libname.c_str());

    if (modules().find(libname) != modules().end()) {
        return true;
    }


#ifdef _MSC_VER
    auto handle = LoadLibrary(_T(libname.c_str()));
#else
    auto handle = dlopen(libname.c_str(), RTLD_DEEPBIND | RTLD_NOW);
#endif

    if (!handle) {
        DEBUGMSG("Cannot load module: %s \n", dlerror());
        return false;
    }

    Module mod(handle);

    modules().insert(std::pair<std::string, Module>(libname, handle));
    
    for (auto cb : callbacks()) {
        cb(mod);
    }

    return true;
}

bool Module::callback(std::function<void(Module& module)> cb) {
    callbacks().push_back(cb);
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


Module::Module(void* handle): handle(handle) {}

Module::~Module() {}


}