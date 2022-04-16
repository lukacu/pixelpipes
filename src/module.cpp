
#include <cstdlib>
#include <mutex>
#include <filesystem>
#include <algorithm>
#include <iostream>

#ifndef PIXELPIPES_BUILD_CORE
#define PIXELPIPES_BUILD_CORE
#endif
#include <pixelpipes/module.hpp>

#ifdef _MSC_VER
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include "debug.h"

using namespace std::filesystem;

namespace pixelpipes
{

    typedef void (*ModuleInitHook)();

    ModuleException::ModuleException(std::string reason) : BaseException(reason) {}

    std::map<std::string, SharedModule> &Module::modules()
    {
        static std::map<std::string, SharedModule> n;
        return n;
    }

    static std::recursive_mutex module_load_lock;

    static SharedModule load_context;

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
            candidate /= "pixelpipes_" + libname + ".dll";
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

    SharedModule Module::context()
    {

        std::lock_guard<std::recursive_mutex> lock(module_load_lock);

        return load_context;
    }

    bool Module::load(const std::string &libname)
    {

        std::lock_guard<std::recursive_mutex> lock(module_load_lock);

        DEBUGMSG("Loading module %s \n", libname.c_str());

        if (modules().find(libname) != modules().end())
        {
            return true;
        }

        auto fullname = find_module(libname);

#ifdef _MSC_VER
        auto handle = LoadLibrary(_T(fullname.c_str()));
#else
        auto handle = dlopen(fullname.c_str(), RTLD_DEEPBIND | RTLD_NOW);
#endif

        if (!handle)
        {
            DEBUGMSG("Cannot load module: %s \n", dlerror());
            return false;
        }

        SharedModule mod = std::shared_ptr<Module>(new Module(handle, libname));

        modules().insert(std::pair<std::string, SharedModule>(libname, mod));

        auto init = mod->symbol<ModuleInitHook>("pixelpipes_module_init");

        load_context = mod;

        DEBUGMSG("Running init hook with module context\n");

        if (init)
        {
            init();
        }

        load_context.reset();

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