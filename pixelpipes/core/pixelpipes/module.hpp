#pragma once

#include <string>
#include <functional>
#include <list>
#include <map>
#include <type_traits>

#include <pixelpipes/base.hpp>

namespace pixelpipes {

class PIXELPIPES_API ModuleException : public BaseException {
public:
    ModuleException(std::string reason);
};

class PIXELPIPES_API Module {
public:

    ~Module();

    template<typename T> T symbol(std::string name) {

        void* s = get_symbol(name);

        if (!s) return NULL;

        return (T) (s);

    }

    static bool callback(std::function<void(Module& module)>);

    static bool load(const std::string&);

private:

    static std::map<std::string, Module>& modules();

    static std::list<std::function<void(Module& module)>>& callbacks();

    void* handle;

    Module(void* handle);

    void* get_symbol(const std::string& name) const;

};

}