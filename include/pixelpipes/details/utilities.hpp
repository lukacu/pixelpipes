#pragma once

#include <memory>

#define UNUSED(expr)  \
    do                \
    {                 \
        (void)(expr); \
    } while (0)

namespace pixelpipes
{

    namespace details
    {

        template <std::size_t N>
        class static_string
        {
        public:
            constexpr explicit static_string(std::string_view str) noexcept : static_string{str, std::make_index_sequence<N>{}}
            {
                // static_assert(str.size() == N, "Illegal size");
            }

            constexpr const char *data() const noexcept { return chars_; }

            constexpr std::size_t size() const noexcept { return N; }

            constexpr operator std::string_view() const noexcept { return {data(), size()}; }

        private:
            template <std::size_t... I>
            constexpr static_string(std::string_view str, std::index_sequence<I...>) noexcept : chars_{str[I]..., '\0'} {}

            char chars_[N + 1];
        };

        template <>
        class static_string<0>
        {
        public:
            constexpr explicit static_string(std::string_view) noexcept {}

            constexpr const char *data() const noexcept { return nullptr; }

            constexpr std::size_t size() const noexcept { return 0; }

            constexpr operator std::string_view() const noexcept { return {}; }
        };

        template <typename T>
        struct has_contiguous_memory : std::false_type
        {
        };

        template <typename T, typename U>
        struct has_contiguous_memory<std::vector<T, U>> : std::true_type
        {
        };

        template <typename T, typename U>
        struct has_contiguous_memory<const std::vector<T, U>> : std::true_type
        {
        };

        template <typename T>
        struct has_contiguous_memory<std::vector<bool, T>> : std::false_type
        {
        };

        template <typename T>
        struct has_contiguous_memory<const std::vector<bool, T>> : std::false_type
        {
        };

        template <typename T, typename U, typename V>
        struct has_contiguous_memory<std::basic_string<T, U, V>> : std::true_type
        {
        };

        template <typename T, std::size_t N>
        struct has_contiguous_memory<std::array<T, N>> : std::true_type
        {
        };

        template <typename T>
        struct has_contiguous_memory<T[]> : std::true_type
        {
        };

        template <typename T, std::size_t N>
        struct has_contiguous_memory<T[N]> : std::true_type
        {
        };

        template <bool B, typename T = void>
        using enable_if_t = typename std::enable_if<B, T>::type;

        template <typename T, typename = void>
        struct is_input_iterator : std::false_type
        {
        };
        template <typename T>
        struct is_input_iterator<T, std::void_t<decltype(*std::declval<T &>()), decltype(++std::declval<T &>())>>
            : std::true_type
        {
        };

/*
        template <typename T>
        struct is_reference
        {
            static constexpr bool value = std::is_pointer<T>::value;
        };

        template <template <typename...> typename C, typename U>
        struct is_reference<C<U>>
        {
            static constexpr bool value =
                std::is_same<C<U>, std::shared_ptr<U>>::value || std::is_same<C<U>, std::weak_ptr<U>>::value;
        };
*/
        template <typename C>
        struct is_string
        {
            static constexpr bool value =
                std::is_same<C, std::string>::value;
        };

        template <typename T, typename = void>
        struct is_iterable : std::false_type
        {
        };

        // this gets used only when we can call std::begin() and std::end() on that type
        template <typename T>
        struct is_iterable<T, std::void_t<decltype(std::begin(std::declval<T>())),
                                          decltype(std::end(std::declval<T>()))>> : std::true_type
        {
        };

        // Here is a helper:
        template <typename T>
        constexpr bool is_iterable_v = is_iterable<T>::value;

        template <class... T>
        constexpr bool always_false = false;


    }

}