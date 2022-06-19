#pragma once

/**
 *  Based on magic_enum
 */

#include <type_traits>
#include <string_view>

#include <pixelpipes/details/utilities.hpp>

namespace pixelpipes {

namespace details {

template <typename E>
struct enum_range {
  static_assert(std::is_enum_v<E>, "enum_range requires enum type.");
  inline static constexpr int min = -128;
  inline static constexpr int max = 128;
  static_assert(max > min, "enum_range requires max > min.");
};

template <typename E>
constexpr std::string_view enum_name(E) noexcept {
  static_assert(std::is_enum_v<E>, "enum_name requires enum type.");
  return {};
}

template <typename T, std::size_t N, std::size_t... I>
constexpr std::array<std::remove_cv_t<T>, N> to_array(T (&a)[N], std::index_sequence<I...>) {
  return {{a[I]...}};
}

constexpr std::string_view pretty_name(std::string_view name) noexcept {
  for (std::size_t i = name.size(); i > 0; --i) {
    if (!((name[i - 1] >= '0' && name[i - 1] <= '9') ||
          (name[i - 1] >= 'a' && name[i - 1] <= 'z') ||
          (name[i - 1] >= 'A' && name[i - 1] <= 'Z') ||
          (name[i - 1] == '_'))) {
      name.remove_prefix(i);
      break;
    }
  }

  if (name.size() > 0 && ((name.front() >= 'a' && name.front() <= 'z') ||
                          (name.front() >= 'A' && name.front() <= 'Z') ||
                          (name.front() == '_'))) {
    return name;
  }

  return {}; // Invalid name.
}

template <typename L, typename R>
constexpr bool cmp_less(L lhs, R rhs) noexcept {
  static_assert(std::is_integral_v<L> && std::is_integral_v<R>, "cmp_less requires integral type.");

  if constexpr (std::is_signed_v<L> == std::is_signed_v<R>) {
    // If same signedness (both signed or both unsigned).
    return lhs < rhs;
  } else if constexpr (std::is_signed_v<R>) {
    // If 'right' is negative, then result is 'false', otherwise cast & compare.
    return rhs > 0 && lhs < static_cast<std::make_unsigned_t<R>>(rhs);
  } else {
    // If 'left' is negative, then result is 'true', otherwise cast & compare.
    return lhs < 0 || static_cast<std::make_unsigned_t<L>>(lhs) < rhs;
  }
}


template <typename E, bool IsFlags, typename U = std::underlying_type_t<E>>
constexpr int reflected_min() noexcept {
  static_assert(std::is_enum_v<E>, "reflected_min requires enum type.");

  if constexpr (IsFlags) {
    return 0;
  } else {
    constexpr auto lhs = enum_range<E>::min;
    static_assert(lhs > (std::numeric_limits<std::int16_t>::min)(), "enum_range requires min must be greater than INT16_MIN.");
    constexpr auto rhs = (std::numeric_limits<U>::min)();

    if constexpr (cmp_less(lhs, rhs)) {
      return rhs;
    } else {
      return lhs;
    }
  }
}

template <typename E, bool IsFlags, typename U = std::underlying_type_t<E>>
constexpr int reflected_max() noexcept {
  static_assert(std::is_enum_v<E>, "reflected_max requires enum type.");

  if constexpr (IsFlags) {
    return std::numeric_limits<U>::digits - 1;
  } else {
    constexpr auto lhs = enum_range<E>::max;
    static_assert(lhs < (std::numeric_limits<std::int16_t>::max)(), "enum_range requires max must be less than INT16_MAX.");
    constexpr auto rhs = (std::numeric_limits<U>::max)();

    if constexpr (cmp_less(lhs, rhs)) {
      return lhs;
    } else {
      return rhs;
    }
  }
}

template <typename E>
constexpr auto n() noexcept {
  static_assert(std::is_enum_v<E>, "n requires enum type.");
#  if defined(__clang__)
  constexpr std::string_view name{__PRETTY_FUNCTION__ + 34, sizeof(__PRETTY_FUNCTION__) - 36};
#  elif defined(__GNUC__)
  constexpr std::string_view name{__PRETTY_FUNCTION__ + 49, sizeof(__PRETTY_FUNCTION__) - 51};
#  elif defined(_MSC_VER)
  constexpr std::string_view name{__FUNCSIG__ + 40, sizeof(__FUNCSIG__) - 57};
#  endif
  return static_string<name.size()>{name};
}

template <typename E>
inline constexpr auto type_name_v = n<E>();

template <typename E, E V>
constexpr auto n() noexcept {
  static_assert(std::is_enum_v<E>, "n requires enum type.");
  constexpr auto custom_name = enum_name<E>(V);

  if constexpr (custom_name.empty()) {
    static_cast<void>(custom_name);
#  if defined(__clang__) || defined(__GNUC__)
    constexpr auto name = pretty_name({__PRETTY_FUNCTION__, sizeof(__PRETTY_FUNCTION__) - 2});
#  elif defined(_MSC_VER)
    constexpr auto name = pretty_name({__FUNCSIG__, sizeof(__FUNCSIG__) - 17});
#  endif
    return static_string<name.size()>{name};
  } else {
    return static_string<custom_name.size()>{custom_name};
  }
}

template <typename E, auto V>
constexpr bool is_valid() noexcept {
  static_assert(std::is_enum_v<E>, "is_valid requires enum type.");

  return n<E, static_cast<E>(V)>().size() != 0;
}

template <typename E, bool IsFlags = false>
inline constexpr auto reflected_min_v = reflected_min<E, IsFlags>();

template <typename E, bool IsFlags = false>
inline constexpr auto reflected_max_v = reflected_max<E, IsFlags>();

template <typename E, int O, bool IsFlags = false, typename U = std::underlying_type_t<E>>
constexpr E value(std::size_t i) noexcept {
  static_assert(std::is_enum_v<E>, "value requires enum type.");

  if constexpr (IsFlags) {
    return static_cast<E>(U{1} << static_cast<U>(static_cast<int>(i) + O));
  } else {
    return static_cast<E>(static_cast<int>(i) + O);
  }
}

template <std::size_t N>
constexpr std::size_t values_count(const bool (&valid)[N]) noexcept {
  auto count = std::size_t{0};
  for (std::size_t i = 0; i < N; ++i) {
    if (valid[i]) {
      ++count;
    }
  }

  return count;
}

template <typename E, bool IsFlags, int Min, std::size_t... I>
constexpr auto values(std::index_sequence<I...>) noexcept {
  static_assert(std::is_enum_v<E>, "details::values requires enum type.");
  constexpr bool valid[sizeof...(I)] = {is_valid<E, value<E, Min, IsFlags>(I)>()...};
  constexpr std::size_t count = values_count(valid);

  E values[count] = {};
  for (std::size_t i = 0, v = 0; v < count; ++i) {
    if (valid[i]) {
      values[v++] = value<E, Min, IsFlags>(i);
    }
  }

  return to_array(values, std::make_index_sequence<count>{});
}

template <typename E, bool IsFlags, typename U = std::underlying_type_t<E>>
constexpr auto values() noexcept {
  static_assert(std::is_enum_v<E>, "values requires enum type.");
  constexpr auto min = reflected_min_v<E, IsFlags>;
  constexpr auto max = reflected_max_v<E, IsFlags>;
  constexpr auto range_size = max - min + 1;
  static_assert(range_size > 0, "enum_range requires valid size.");
  static_assert(range_size < (std::numeric_limits<std::uint16_t>::max)(), "enum_range requires valid size.");
  if constexpr (cmp_less((std::numeric_limits<U>::min)(), min) && !IsFlags) {
    static_assert(!is_valid<E, value<E, min - 1, IsFlags>(0)>(), "enum_range detects enum value smaller than min range size.");
  }
  if constexpr (cmp_less(range_size, (std::numeric_limits<U>::max)()) && !IsFlags) {
    static_assert(!is_valid<E, value<E, min, IsFlags>(range_size + 1)>(), "enum_range detects enum value larger than max range size.");
  }

  return values<E, IsFlags, reflected_min_v<E, IsFlags>>(std::make_index_sequence<range_size>{});
}

template <typename E, E V>
inline constexpr auto enum_name_v = n<E, V>();

template <typename E, bool IsFlags = false>
inline constexpr auto values_v = values<E, IsFlags>();

template <typename E, bool IsFlags = false, typename D = std::decay_t<E>>
using values_t = decltype((values_v<D, IsFlags>));

template <typename E, bool IsFlags = false>
inline constexpr auto count_v = values_v<E, IsFlags>.size();

template <typename E, bool IsFlags, std::size_t... I>
constexpr auto entries(std::index_sequence<I...>) noexcept {
  static_assert(std::is_enum_v<E>, "requires enum type.");

  return std::array<std::pair<E, std::string_view>, sizeof...(I)>{{{values_v<E, IsFlags>[I], enum_name_v<E, values_v<E, IsFlags>[I]>}...}};
}

template <typename E, bool IsFlags = false>
inline constexpr auto entries_v = entries<E, IsFlags>(std::make_index_sequence<count_v<E, IsFlags>>{});

template <typename E, bool IsFlags = false, typename D = std::decay_t<E>>
using entries_t = decltype((entries_v<D, IsFlags>));

template <typename T, typename R = void>
using enable_if_enum_t = std::enable_if_t<std::is_enum_v<std::decay_t<T>>, R>;

// Returns std::array with pairs (value, name), sorted by enum value.
template <typename E>
[[nodiscard]] constexpr auto enum_entries() noexcept -> enable_if_enum_t<E, entries_t<E>> {
using D = std::decay_t<E>;
static_assert(count_v<D> > 0, "enum requires enum implementation and valid max and min.");

return entries_v<D>;
}

}

}