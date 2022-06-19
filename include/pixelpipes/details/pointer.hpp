#pragma once

#include <functional>

#include <pixelpipes/details/atomic.hpp>
#include <pixelpipes/details/rtti.hpp>

////////////////////////////////////////////////////////////////////////

namespace pixelpipes
{

    ////////////////////////////////////////////////////////////////////////

    // Forward dependencies.

    template <typename T>
    class Pointer;

    ////////////////////////////////////////////////////////////////////////

    // NOTE: currently this implementation of Borrowable does an atomic
    // backoff instead of blocking the thread when the destructor waits
    // for all borrows to be relinquished. This will be much less
    // efficient (and hold up a CPU) if the borrowers take a while to
    // relinquish. However, since Borrowable will mostly be used in
    // cirumstances where the tally is definitely back to 0 when we wait
    // no backoff will occur. For circumstances where Borrowable is being
    // used to wait until work is completed consider using a Notification
    // to be notified when the work is complete and then Borrowable should
    // destruct without any atomic backoff (because any workers/threads
    // will have relinquished).
    class TypeErasedBorrowable
    {
    public:
        template <typename F>
        bool Watch(F &&f)
        {
            auto [state, count] = tally_.Wait([](auto, size_t)
                                              { return true; });

            do
            {
                if (state == State::Watching)
                {
                    return false;
                }
                else if (count == 0)
                {
                    f();
                    return true;
                }

                CHECK_EQ(state, State::Borrowing);

            } while (!tally_.Update(state, count, State::Watching, count + 1));

            watch_ = std::move(f);

            Relinquish();

            return true;
        }

        void WaitUntilBorrowsEquals(size_t borrows)
        {
            tally_.Wait([&](auto /* state */, size_t count)
                        { return count == borrows; });
        }

        size_t borrows()
        {
            return tally_.count();
        }

        void Relinquish()
        {
            auto [state, count] = tally_.Decrement();

            if (state == State::Watching && count == 0)
            {
                // Move out 'watch_' in case it gets reset either in the
                // callback or because a concurrent call to 'borrow()' occurs
                // after we've updated the tally below.
                auto f = std::move(watch_);
                watch_ = std::function<void()>();

                tally_.Update(state, State::Borrowing);

                // At this point a call to 'borrow()' may mean that there are
                // outstanding 'borrowed_ref/ptr' when the watch callback gets
                // invoked and thus it's up to the users of this abstraction to
                // avoid making calls to 'borrow()' until after the watch
                // callback gets invoked if they want to guarantee that there
                // are no outstanding 'borrowed_ref/ptr'.

                f();
            }
        }

    protected:
        TypeErasedBorrowable()
            : tally_(State::Borrowing) {}

        TypeErasedBorrowable(const TypeErasedBorrowable &that)
            : tally_(State::Borrowing) { UNUSED(that); }

        TypeErasedBorrowable(TypeErasedBorrowable &&that)
            : tally_(State::Borrowing)
        {
            // We need to wait until all borrows have been relinquished so
            // any memory associated with 'that' can be safely released.
            that.WaitUntilBorrowsEquals(0);
        }

        virtual ~TypeErasedBorrowable()
        {
            auto state = State::Borrowing;
            if (!tally_.Update(state, State::Destructing))
            {
                // LOG(FATAL) << "Unable to transition to Destructing from state " << state;
            }
            else
            {
                // NOTE: it's possible that we'll block forever if exceptions
                // were thrown and destruction was not successful.
                // if (!std::uncaught_exceptions() > 0) {
                WaitUntilBorrowsEquals(0);
                // }
            }
        }

        enum class State : uint8_t
        {
            Borrowing,
            Watching,
            Destructing,
        };

        // We need to overload '<<' operator for 'State' enum class in
        // order to use 'CHECK_*' family macros.
        friend std::ostream &operator<<(
            std::ostream &os,
            const TypeErasedBorrowable::State &state)
        {
            switch (state)
            {
            case TypeErasedBorrowable::State::Borrowing:
                return os << "Borrowing";
            case TypeErasedBorrowable::State::Watching:
                return os << "Watching";
            case TypeErasedBorrowable::State::Destructing:
                return os << "Destructing";
            }
        };

        // NOTE: 'stateful_tally' ensures this is non-moveable (but still
        // copyable). What would it mean to be able to borrow a pointer to
        // something that might move!? If an implemenetation ever replaces
        // 'stateful_tally' with something else care will need to be taken
        // to ensure that 'Borrowable' doesn't become moveable.
        pixelpipes::details::StatefulTally<State> tally_;

        std::function<void()> watch_;

    private:
        // Only 'Pointer' can reborrow!
        template <typename>
        friend class Pointer;

        template <typename B, typename A>
        friend Pointer<B>
        cast(const Pointer<A> &p);

        void Reborrow()
        {
            auto [state, count] = tally_.Wait([](auto, size_t)
                                              { return true; });

            // CHECK_GT(count, 0u);

            do
            {
                // CHECK_NE(state, State::Destructing);
            } while (!tally_.Increment(state));
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
              t_(std::forward<Args>(args)...) {}

        Borrowable(const Borrowable &that)
            : TypeErasedBorrowable(that),
              t_(that.t_) {}

        Borrowable(Borrowable &&that)
            : TypeErasedBorrowable(std::move(that)),
              t_(std::move(that.t_)) {}

        T *get()
        {
            return &t_;
        }

        const T *get() const
        {
            return &t_;
        }

        T *operator->()
        {
            return get();
        }

        const T *operator->() const
        {
            return get();
        }

        T &operator*()
        {
            return t_;
        }

        const T &operator*() const
        {
            return t_;
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
            return std::less<T*>(get(), b.get());
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
                borrowable_->Reborrow();
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
                borrowable_->Reborrow();
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
            return Pointer<U>(borrowable, t);
        }

        Pointer reborrow() const
        {
            if (borrowable_ != nullptr)
            {
                borrowable_->Reborrow();
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
                borrowable_->Relinquish();
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

        // TODO(benh): operator[]

        template <typename H>
        friend H AbslHashValue(H h, const Pointer &that)
        {
            return H::combine(std::move(h), that.t_);
        }

    private:
        template <typename>
        friend class Pointer;

        template <typename>
        friend class Borrowable;

        template <typename U, typename... _Args>
        friend Pointer<U>
        create(_Args &&...__args);

        template <typename B, typename A>
        friend Pointer<B>
        cast(const Pointer<A> &p);

        Pointer(TypeErasedBorrowable *borrowable, T *t)
            : borrowable_(borrowable),
              t_(t) {}

        TypeErasedBorrowable *borrowable_ = nullptr;
        T *t_ = nullptr;
    };

    template <typename T, typename... _Args>
    Pointer<T>
    create(_Args &&...__args)
    {
        auto data = new Borrowable<T>(__args...);

        return Pointer<T>(data, data->get());
    }

    template <typename B, typename A>
    Pointer<B>
    cast(const Pointer<A> &p)
    {
        p.borrowable_->Reborrow();
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

}