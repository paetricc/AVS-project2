set(CMAKE_C_COMPILER "/apps/all/intel-compilers/2022.1.0/compiler/2022.1.0/linux/bin/intel64/icc")
set(CMAKE_C_COMPILER_ARG1 "")
set(CMAKE_C_COMPILER_ID "Intel")
set(CMAKE_C_COMPILER_VERSION "2021.6.0.20220226")
set(CMAKE_C_COMPILER_VERSION_INTERNAL "")
set(CMAKE_C_COMPILER_WRAPPER "")
set(CMAKE_C_STANDARD_COMPUTED_DEFAULT "11")
set(CMAKE_C_EXTENSIONS_COMPUTED_DEFAULT "ON")
set(CMAKE_C_COMPILE_FEATURES "c_std_90;c_function_prototypes;c_std_99;c_restrict;c_variadic_macros;c_std_11;c_static_assert")
set(CMAKE_C90_COMPILE_FEATURES "c_std_90;c_function_prototypes")
set(CMAKE_C99_COMPILE_FEATURES "c_std_99;c_restrict;c_variadic_macros")
set(CMAKE_C11_COMPILE_FEATURES "c_std_11;c_static_assert")
set(CMAKE_C17_COMPILE_FEATURES "")
set(CMAKE_C23_COMPILE_FEATURES "")

set(CMAKE_C_PLATFORM_ID "Linux")
set(CMAKE_C_SIMULATE_ID "GNU")
set(CMAKE_C_COMPILER_FRONTEND_VARIANT "")
set(CMAKE_C_SIMULATE_VERSION "11.3.0")




set(CMAKE_AR "/apps/all/binutils/2.38-GCCcore-11.3.0/bin/ar")
set(CMAKE_C_COMPILER_AR "")
set(CMAKE_RANLIB "/apps/all/binutils/2.38-GCCcore-11.3.0/bin/ranlib")
set(CMAKE_C_COMPILER_RANLIB "")
set(CMAKE_LINKER "/apps/all/binutils/2.38-GCCcore-11.3.0/bin/ld")
set(CMAKE_MT "")
set(CMAKE_COMPILER_IS_GNUCC )
set(CMAKE_C_COMPILER_LOADED 1)
set(CMAKE_C_COMPILER_WORKS TRUE)
set(CMAKE_C_ABI_COMPILED TRUE)

set(CMAKE_C_COMPILER_ENV_VAR "CC")

set(CMAKE_C_COMPILER_ID_RUN 1)
set(CMAKE_C_SOURCE_FILE_EXTENSIONS c;m)
set(CMAKE_C_IGNORE_EXTENSIONS h;H;o;O;obj;OBJ;def;DEF;rc;RC)
set(CMAKE_C_LINKER_PREFERENCE 10)

# Save compiler ABI information.
set(CMAKE_C_SIZEOF_DATA_PTR "8")
set(CMAKE_C_COMPILER_ABI "ELF")
set(CMAKE_C_BYTE_ORDER "LITTLE_ENDIAN")
set(CMAKE_C_LIBRARY_ARCHITECTURE "")

if(CMAKE_C_SIZEOF_DATA_PTR)
  set(CMAKE_SIZEOF_VOID_P "${CMAKE_C_SIZEOF_DATA_PTR}")
endif()

if(CMAKE_C_COMPILER_ABI)
  set(CMAKE_INTERNAL_PLATFORM_ABI "${CMAKE_C_COMPILER_ABI}")
endif()

if(CMAKE_C_LIBRARY_ARCHITECTURE)
  set(CMAKE_LIBRARY_ARCHITECTURE "")
endif()

set(CMAKE_C_CL_SHOWINCLUDES_PREFIX "")
if(CMAKE_C_CL_SHOWINCLUDES_PREFIX)
  set(CMAKE_CL_SHOWINCLUDES_PREFIX "${CMAKE_C_CL_SHOWINCLUDES_PREFIX}")
endif()





set(CMAKE_C_IMPLICIT_INCLUDE_DIRECTORIES "/apps/all/intel-compilers/2022.1.0/tbb/2021.6.0/include;/apps/all/libarchive/3.6.1-GCCcore-11.3.0/include;/apps/all/XZ/5.2.5-GCCcore-11.3.0/include;/apps/all/cURL/7.83.0-GCCcore-11.3.0/include;/apps/all/OpenSSL/1.1/include;/apps/all/bzip2/1.0.8-GCCcore-11.3.0/include;/apps/all/ncurses/6.3-GCCcore-11.3.0/include;/apps/all/binutils/2.38-GCCcore-11.3.0/include;/apps/all/zlib/1.2.12-GCCcore-11.3.0/include;/apps/all/intel-compilers/2022.1.0/compiler/2022.1.0/linux/compiler/include/intel64;/apps/all/intel-compilers/2022.1.0/compiler/2022.1.0/linux/compiler/include/icc;/apps/all/intel-compilers/2022.1.0/compiler/2022.1.0/linux/compiler/include;/usr/local/include;/apps/all/GCCcore/11.3.0/lib/gcc/x86_64-pc-linux-gnu/11.3.0/include;/apps/all/GCCcore/11.3.0/include;/usr/include")
set(CMAKE_C_IMPLICIT_LINK_LIBRARIES "imf;svml;irng;m;ipgo;decimal;cilkrts;stdc++;gcc;gcc_s;irc;svml;c;gcc;gcc_s;irc_s;dl;c")
set(CMAKE_C_IMPLICIT_LINK_DIRECTORIES "/apps/all/intel-compilers/2022.1.0/tbb/2021.6.0/lib/intel64/gcc4.8;/apps/all/intel-compilers/2022.1.0/compiler/2022.1.0/linux/compiler/lib/intel64_lin;/apps/all/intel-compilers/2022.1.0/compiler/2022.1.0/linux/lib/x64;/apps/all/intel-compilers/2022.1.0/compiler/2022.1.0/linux/lib;/apps/all/libarchive/3.6.1-GCCcore-11.3.0/lib;/apps/all/XZ/5.2.5-GCCcore-11.3.0/lib;/apps/all/cURL/7.83.0-GCCcore-11.3.0/lib;/apps/all/OpenSSL/1.1/lib;/apps/all/bzip2/1.0.8-GCCcore-11.3.0/lib;/apps/all/ncurses/6.3-GCCcore-11.3.0/lib;/apps/all/binutils/2.38-GCCcore-11.3.0/lib;/apps/all/zlib/1.2.12-GCCcore-11.3.0/lib;/apps/all/libarchive/3.6.1-GCCcore-11.3.0/lib64;/apps/all/XZ/5.2.5-GCCcore-11.3.0/lib64;/apps/all/cURL/7.83.0-GCCcore-11.3.0/lib64;/apps/all/OpenSSL/1.1/lib64;/apps/all/bzip2/1.0.8-GCCcore-11.3.0/lib64;/apps/all/ncurses/6.3-GCCcore-11.3.0/lib64;/apps/all/binutils/2.38-GCCcore-11.3.0/lib64;/apps/all/zlib/1.2.12-GCCcore-11.3.0/lib64;/apps/all/GCCcore/11.3.0/lib/gcc/x86_64-pc-linux-gnu/11.3.0;/apps/all/GCCcore/11.3.0/lib64;/lib64;/usr/lib64;/apps/all/GCCcore/11.3.0/lib;/lib;/usr/lib")
set(CMAKE_C_IMPLICIT_LINK_FRAMEWORK_DIRECTORIES "")
