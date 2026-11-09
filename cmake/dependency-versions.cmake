# Dependency source pins for release builds. Update these revisions deliberately
# and review the corresponding upstream changes before changing a pin.
set(PIXELPIPES_OPENCV_VERSION "4.5.5")
set(PIXELPIPES_OPENCV_REVISION "dad26339a975b49cfb6c7dbe4bd5276c9dcb36e2")
set(PIXELPIPES_XTL_VERSION "0.7.4")
set(PIXELPIPES_XTL_REVISION "e697c91e2a3ac571d120d2b093fb3b250d060a7d")
set(PIXELPIPES_XSIMD_VERSION "10.0.0")
set(PIXELPIPES_XSIMD_REVISION "e12bf0a928bd6668ff701db55803a9e316cb386c")
set(PIXELPIPES_XTENSOR_VERSION "0.24.3")
set(PIXELPIPES_XTENSOR_REVISION "9e8662ddad7fcd1a96969b96bf8955e59cf44978")
# Existing zlib-ng revision: 2.0.4-326-gc882034d (not an upstream release tag).
set(PIXELPIPES_ZLIBNG_REVISION "c882034d48afc0b32a38e8f7ca63a2e4e91ab42d")
# Existing lodepng revision is identified by its immutable commit, not a release.
set(PIXELPIPES_LODEPNG_REVISION "5601b8272a6850b7c5d693dd0c0e16da50be8d8d")
# hash-library has no upstream release tags; pin its source to an immutable commit.
set(PIXELPIPES_HASHLIB_REVISION "d389d18112bcf7e4786ec5e8723f3658a7f433d7")

set(PIXELPIPES_OPENCV_REPOSITORY "https://github.com/opencv/opencv")
set(PIXELPIPES_XTL_REPOSITORY "https://github.com/xtensor-stack/xtl")
set(PIXELPIPES_XSIMD_REPOSITORY "https://github.com/xtensor-stack/xsimd")
set(PIXELPIPES_XTENSOR_REPOSITORY "https://github.com/xtensor-stack/xtensor")
set(PIXELPIPES_ZLIBNG_REPOSITORY "https://github.com/zlib-ng/zlib-ng")
set(PIXELPIPES_LODEPNG_REPOSITORY "https://github.com/lvandeve/lodepng.git")
set(PIXELPIPES_HASHLIB_REPOSITORY "https://github.com/stbrumme/hash-library")
