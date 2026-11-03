# External C++ module example

This minimal module adds an `example:example_triple` operation and demonstrates using
the installed PixelPipes SDK. The installed package exposes its public include
and library directories through `pixelpipes.include_dirs()` and
`pixelpipes.link_dirs()`; the CMake project uses those paths rather than
flattening or hard-coding the SDK layout.

With PixelPipes installed in the active Python environment:

```sh
cmake -S examples/custom_module -B build/custom_module
cmake --build build/custom_module
PIXELPIPES_MODULES_PATH="$PWD/build/custom_module" python examples/custom_module/run.py
```

On Windows, set `PIXELPIPES_MODULES_PATH` to the CMake build directory using
the shell's environment-variable syntax.
