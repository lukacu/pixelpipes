
#include <cstdlib>
#include <mutex>
#include <filesystem>
#include <algorithm>
#include <iostream>

#ifndef PIXELPIPES_BUILD_CORE
#define PIXELPIPES_BUILD_CORE
#endif
#include <pixelpipes/module.hpp>

#ifdef _WIN32
#include <windows.h>
#include <tchar.h>
#else
#include <dlfcn.h>
#endif

#include "debug.h"

using namespace std::filesystem;

namespace pixelpipes
{

    typedef void (*ModuleInitHook)();

    ModuleException::ModuleException(std::string reason) : BaseException(reason) {}

    std::map<std::string, ModuleReference> &Module::modules()
    {
        static std::map<std::string, ModuleReference> n;
        return n;
    }

    static std::recursive_mutex module_load_lock;

    static ModuleReference load_context;

    static std::vector<path> module_paths;

    const std::vector<std::string> module_path_envs{"PIXELPIPES_MODULES_PATH", "LD_LIBRARY_PATH", "PATH"};

    std::vector<std::string> split(const std::string &str, const std::string &delim)
    {
        std::vector<std::string> tokens;
        size_t prev = 0, pos = 0;
        do
        {
            pos = str.find(delim, prev);
            if (pos == std::string::npos)
                pos = str.length();
            std::string token = str.substr(prev, pos - prev);
            if (!token.empty())
                tokens.push_back(token);
            prev = pos + delim.length();
        } while (pos < str.length() && prev < str.length());
        return tokens;
    }

    path find_module(std::string name)
    {

        if (module_paths.size() == 0)
        {

            for (auto envname : module_path_envs)
            {

                if (!std::getenv(envname.c_str()))
                    continue;

                auto paths = split(std::getenv(envname.c_str()), os_pathsep);

                for (auto p : paths)
                    module_paths.push_back(path(p));
            }

            module_paths.push_back(current_path());

            //std::sort(module_paths.begin(), module_paths.end());
            //module_paths.erase(std::unique(module_paths.begin(), module_paths.end()), module_paths.end());
        }

        for (auto dir : module_paths)
        {
            path candidate = dir;

#ifdef _MSC_VER
            candidate /= "pixelpipes_" + name + ".dll";
#else
            candidate /= "libpixelpipes_" + name + ".so";
#endif

            if (exists(candidate) && (is_regular_file(candidate)))
            {
                return candidate;
            }
        }

        throw ModuleException(Formatter() << "Module " << name << " not found");
    }

    ModuleReference Module::context()
    {

        std::lock_guard<std::recursive_mutex> lock(module_load_lock);

        return load_context.reborrow();
    }

    bool Module::load(std::string_view libname)
    {
        std::lock_guard<std::recursive_mutex> lock(module_load_lock);

        std::string libname_str(libname);

        if (modules().find(libname_str) != modules().end())
        {
            return true;
        }

        DEBUGMSG("Loading module %s \n", libname_str.c_str());

        auto fullname = find_module(libname_str);

#ifdef _MSC_VER
        auto handle = LoadLibrary(fullname.string().c_str());
#elif defined(__APPLE__)
        auto handle = dlopen(fullname.c_str(), RTLD_NOW);
#else
        auto handle = dlopen(fullname.c_str(), RTLD_DEEPBIND | RTLD_NOW);
#endif

        if (!handle)
        {
            DEBUGMSG("Cannot load module: %s \n", dlerror());
            return false;
        }

        ModuleReference mod = create<Module>(handle, libname_str);

        modules().insert(std::pair<std::string, ModuleReference>(libname, mod.reborrow()));

        auto init = mod->symbol<ModuleInitHook>("pixelpipes_module_init");

        load_context = mod.reborrow();

        DEBUGMSG("Running init hook with module context\n");

        if (init)
        {
            init();
        }

        load_context.relinquish();

        DEBUGMSG("Context reset\n");

        return true;
    }

    void *Module::get_symbol(const std::string &name) const
    {

#ifdef _MSC_VER
        void *sym = GetProcAddress((HMODULE)handle, _T(name.c_str()));
#else
        void *sym = dlsym(handle, name.c_str());
#endif

        if (!sym)
        {
            DEBUGMSG("Cannot find symbol %s: %s \n", name.c_str(), dlerror());
        }

        return sym;
    }

    Module::Module(void *handle, std::string name) : handle(handle), module_name(name) {}

    Module::~Module() {}

    std::string Module::name() const
    {
        return module_name;
    }

    AddDirect::AddDirect(ModuleInitializer initializer)
    {
        initializer();
    }

}
