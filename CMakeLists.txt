cmake_minimum_required(VERSION 3.18)
project(a_miner LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "RelWithDebInfo" "MinSizeRel")
endif()

# ---------- optional CUDA support ----------
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CUDA_FOUND TRUE)
    add_definitions(-DHAVE_CUDA)
    message(STATUS "CUDA found – GPU mining enabled")
else()
    set(CUDA_FOUND FALSE)
    message(STATUS "CUDA not found – building CPU-only (verification / test mode)")
endif()

# ---------- third-party ----------
find_package(OpenSSL QUIET)
find_package(Threads REQUIRED)

# ---------- sources ----------
set(CORE_SOURCES
    src/main.cpp
    src/miner.cpp
    src/kawpow/keccak.cpp
    src/kawpow/ethash.cpp
    src/kawpow/kawpow.cpp
    src/stratum/stratum_client.cpp
    src/utils/log.cpp
    src/utils/hex.cpp
)

if(CUDA_FOUND)
    list(APPEND CORE_SOURCES src/kawpow/cuda/kawpow_kernel.cu)
endif()

add_executable(a_miner ${CORE_SOURCES})

target_include_directories(a_miner PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)

target_compile_definitions(a_miner PRIVATE
    AMINER_BUILD_TYPE="${CMAKE_BUILD_TYPE}"
)

target_link_libraries(a_miner PRIVATE Threads::Threads)
if(OPENSSL_FOUND)
    target_link_libraries(a_miner PRIVATE OpenSSL::SSL OpenSSL::Crypto)
endif()

if(CUDA_FOUND)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 86)
    endif()

    set_target_properties(a_miner PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    )

    target_compile_definitions(a_miner PRIVATE
        AMINER_CUDA_ARCHS="${CMAKE_CUDA_ARCHITECTURES}"
    )
else()
    target_compile_definitions(a_miner PRIVATE
        AMINER_CUDA_ARCHS="CPU_ONLY"
    )
endif()

# ---------- tests ----------
enable_testing()

add_executable(test_keccak
    tests/test_keccak.cpp
    src/kawpow/keccak.cpp
    src/utils/hex.cpp
)
target_include_directories(test_keccak PRIVATE ${CMAKE_SOURCE_DIR}/src)
add_test(NAME test_keccak COMMAND test_keccak)

add_executable(test_ethash
    tests/test_ethash.cpp
    src/kawpow/keccak.cpp
    src/kawpow/ethash.cpp
    src/utils/hex.cpp
)
target_include_directories(test_ethash PRIVATE ${CMAKE_SOURCE_DIR}/src)
add_test(NAME test_ethash COMMAND test_ethash)
