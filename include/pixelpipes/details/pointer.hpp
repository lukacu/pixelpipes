#pragma once

#include <functional>
#include <atomic>

#include <pixelpipes/details/api.hpp>
#include <pixelpipes/details/rtti.hpp>

/*
 * Based on borrowed-ptr: https://github.com/3rdparty/stout
 *
 */

namespace pixelpipes
{
    // These functions can only be used in debug mode to resolve memory problems
    uint32_t PIXELPIPES_API debug_ref_count();
    uint32_t PIXELPIPES_API debug_ref_inc();
    uint32_t PIXELPIPES_API debug_ref_dec();
    template <typename T>
    class Pointer;

    class TypeErasedBorrowable
    {
    public:

        size_t borrows()
        {
            return _count.load();
        }

        bool relinquish()
        {
            return _count.fetch_sub(1) == 1;
        }

    protected:
        TypeErasedBorrowable()
            : _count(0) { 
                #ifdef PIXELPIPES_DEBUG
                    debug_ref_inc();
                #endif
            }

        TypeErasedBorrowable(const TypeErasedBorrowable &that) = delete;

        TypeErasedBorrowable(TypeErasedBorrowable &&that) = delete;

        virtual ~TypeErasedBorrowable() {
            #ifdef PIXELPIPES_DEBUG
                debug_ref_dec();
            #endif
        }

        // TODO: move implementation so that STL class is not public
        std::atomic<unsigned long> _count;

    private:
        // Only 'Pointer' can reborrow!
        template <typename>
        friend class Pointer;

        template <typename B, typename A>
        friend Pointer<B>
        cast(const Pointer<A> &p);

        template <typename U, typename... _Args>
        friend Pointer<U>
        create(_Args &&...__args);

        void reborrow()
        {
            _count.fetch_add(1);
        }
    };

    ////////////////////////////////////////////////////////////////////////

    template <typename T>
    class Borrowable : public TypeErasedBorrowable
    {
    public:
        template <
            typename... Args,
            std::enable_if_t<std::is_constructible_v<T, Args...>, int> = 0>
        Borrowable(Args &&...args)
            : TypeErasedBorrowable(),
              t_(std::forward<Args>(args)...) {
              }

        Borrowable(const Borrowable &that)
            : TypeErasedBorrowable(that),
              t_(that.t_) {}

        Borrowable(Borrowable &&that)
            : TypeErasedBorrowable(std::move(that)),
              t_(std::move(that.t_)) {}

        virtual ~Borrowable() = default;

        T *get()
        {
            return &t_;
        }

        const T *get() const
        {
            return &t_;
        }

    private:
        T t_;
    };

    // Similar to a raw pointer (and
    // 'std::unique_ptr') it can be a 'nullptr', for example, by
    // constructing a 'Pointer' with the default constructor or after
    // calling 'relinquish()'.
    template <typename T>
    class Pointer final
    {
    public:
        Pointer() {}

        // Deleted copy constructor to force use of 'reborrow()' which makes
        // the copying more explicit!
        Pointer(const Pointer &that) = delete;

        Pointer(Pointer &&that)
        {
            std::swap(borrowable_, that.borrowable_);
            std::swap(t_, that.t_);
        }

        ~Pointer()
        {
            relinquish();
        }

        Pointer &operator=(Pointer &&that)
        {
            std::swap(borrowable_, that.borrowable_);
            std::swap(t_, that.t_);
            return *this;
        }

        explicit operator bool() const
        {
            return borrowable_ != nullptr;
        }

        [[nodiscard]] inline bool
        operator<(const Pointer<T> &b) noexcept
        {
            return std::less<T *>(get(), b.get());
        }

        template <
            typename U,
            std::enable_if_t<
                std::conjunction_v<
                    std::negation<std::is_pointer<U>>,
                    std::negation<std::is_reference<U>>,
                    std::is_convertible<T *, U *>>,
                int> = 0>
        operator Pointer<U>() const &
        {
            if (borrowable_ != nullptr)
            {
                return Pointer<U>(borrowable_, t_);
            }
            else
            {
                return Pointer<U>();
            }
        }

        template <
            typename U,
            std::enable_if_t<
                std::conjunction_v<
                    std::negation<std::is_pointer<U>>,
                    std::negation<std::is_reference<U>>,
                    std::is_convertible<T *, U *>>,
                int> = 0>
        operator Pointer<U>() &
        {
            if (borrowable_ != nullptr)
            {
                return Pointer<U>(borrowable_, t_);
            }
            else
            {
                return Pointer<U>();
            }
        }

        template <
            typename U,
            std::enable_if_t<
                std::conjunction_v<
                    std::negation<std::is_pointer<U>>,
                    std::negation<std::is_reference<U>>,
                    std::is_convertible<T *, U *>>,
                int> = 0>
        operator Pointer<U>() &&
        {
            // Don't reborrow since we're being moved!
            TypeErasedBorrowable *borrowable = nullptr;
            T *t = nullptr;
            std::swap(borrowable, borrowable_);
            std::swap(t, t_);
            return Pointer<U>(borrowable, t, false);
        }

        Pointer reborrow() const
        {
            if (borrowable_ != nullptr)
            {
                return Pointer<T>(borrowable_, t_);
            }
            else
            {
                return Pointer<T>();
            }
        }

        void relinquish()
        {
            if (borrowable_ != nullptr)
            {
                if (borrowable_->relinquish()) {
                    delete borrowable_;
                }
                borrowable_ = nullptr;
                t_ = nullptr;
            }
        }

        template <typename U>
        U *get_as() const
        {
            return static_cast<U *>(t_);
        }

        T *get() const
        {
            return t_;
        }

        T *operator->() const
        {
            return get();
        }

        T &operator*() const
        {
            // NOTE: just like with 'std::unique_ptr' the behavior is
            // undefined if 'get() == nullptr'.
            return *get();
        }

        template <typename H>
        friend H hash_value(H h, const Pointer &that)
        {
            return H::combine(std::move(h), that.t_);
        }

        size_t borrows() const {
            if (borrowable_) return borrowable_->borrows();
            return 0;
        }

    private:
        template <typename>
        friend class Pointer;

        template <typename>
        friend class Borrowable;

        template <typename>
        friend class enable_pointer_from_this;

        template <typename U, typename... _Args>
        friend Pointer<U>
        create(_Args &&...__args);

        template <typename B, typename A>
        friend Pointer<B>
        cast(const Pointer<A> &p);

        Pointer(TypeErasedBorrowable *borrowable, T *t, bool borrow = true)
            : borrowable_(borrowable),
              t_(t) {
                if (borrowable_ && borrow)
                    borrowable_->reborrow();
              }

        TypeErasedBorrowable *borrowable_ = nullptr;
        T *t_ = nullptr;
    };

    template <typename T, typename... _Args>
    Pointer<T>
    create(_Args &&...__args)
    {
        if constexpr (std::is_base_of<TypeErasedBorrowable, T>::value) {
            auto data = new T(__args...);
            return Pointer<T>(data, data);
        } else {
            auto data = new Borrowable<T>(__args...);
            return Pointer<T>(data, data->get());
        }
    }

    template <typename B, typename A>
    Pointer<B>
    cast(const Pointer<A> &p)
    {
        return Pointer<B>(p.borrowable_, p.t_->template cast<B>());
    }

    template <typename T>
    struct is_pointer : std::false_type
    {
    };

    template <typename T>
    struct is_pointer<Pointer<T>> : std::true_type
    {
    };

    template <typename T>
    inline constexpr bool is_pointer_v = is_pointer<T>::value;
        
    template <typename T>
    class enable_pointer_from_this : public TypeErasedBorrowable
    {
    public:
        Pointer<T> reference() const
        {
            static_assert(
                std::is_base_of_v<enable_pointer_from_this<T>, T>,
                "Type 'T' must derive from 'pixelpipes::enable_pointer_from_this<T>'");
            // TODO: This is not very nice :(
            return Pointer<T>((TypeErasedBorrowable *) static_cast<const TypeErasedBorrowable*>(this),
                 (T *) static_cast<const T*>(this));
        }

    };

}