## Clef Operator: Nx Meta analytics plugin. Build against the Nx Server Plugin SDK 6.1:
##   cmake -S plugin -B build -DnxSdkDir=$HOME/nx_sdk/server_plugin_sdk -DCMAKE_BUILD_TYPE=Release
##   cmake --build build -j

cmake_minimum_required(VERSION 3.14)
project(clef_operator_plugin CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Same linking rules as the SDK samples: look up dependencies next to the plugin first.
string(APPEND CMAKE_SHARED_LINKER_FLAGS " -Wl,--disable-new-dtags")
set(CMAKE_SKIP_BUILD_RPATH ON)
set(CMAKE_BUILD_WITH_INSTALL_RPATH ON)
set(CMAKE_INSTALL_RPATH "$ORIGIN")
set(API_EXPORT_MACRO "__attribute__((visibility(\"default\")))")

set(nxSdkDir "" CACHE PATH "Path to the unpacked Nx Server Plugin SDK")
if(NOT EXISTS ${nxSdkDir}/src/nx/sdk)
    message(FATAL_ERROR "Set -DnxSdkDir=<path to server_plugin_sdk>")
endif()

set(nxKitLibraryType "STATIC" CACHE STRING "" FORCE)
set(nxKitWithTests "NO" CACHE STRING "" FORCE)
add_subdirectory(${nxSdkDir}/nx_kit ${CMAKE_CURRENT_BINARY_DIR}/nx_kit)

file(GLOB_RECURSE SDK_SRC CONFIGURE_DEPENDS ${nxSdkDir}/src/*)
add_library(nx_sdk STATIC ${SDK_SRC})
target_include_directories(nx_sdk PUBLIC ${nxSdkDir}/src)
target_link_libraries(nx_sdk PUBLIC nx_kit)
target_compile_definitions(nx_sdk PUBLIC NX_PLUGIN_API=${API_EXPORT_MACRO})

file(GLOB_RECURSE PLUGIN_SRC CONFIGURE_DEPENDS ${CMAKE_CURRENT_LIST_DIR}/src/*.cpp)
add_library(clef_operator_plugin SHARED ${PLUGIN_SRC})
target_include_directories(clef_operator_plugin PRIVATE ${CMAKE_CURRENT_LIST_DIR}/src)
target_link_libraries(clef_operator_plugin PRIVATE nx_sdk pthread)
target_compile_options(clef_operator_plugin PRIVATE -Wall -Wextra -Wno-unused-parameter)

# Unit tests for the parts that do not need the Server.
file(GLOB TEST_SRC CONFIGURE_DEPENDS ${CMAKE_CURRENT_LIST_DIR}/tests/*.cpp)
if(TEST_SRC)
    add_executable(clef_operator_tests ${TEST_SRC}
        ${CMAKE_CURRENT_LIST_DIR}/src/clef_operator/config.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/clef_operator/http_client.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/clef_operator/clip_buffer.cpp)
    target_include_directories(clef_operator_tests PRIVATE ${CMAKE_CURRENT_LIST_DIR}/src)
    target_link_libraries(clef_operator_tests PRIVATE nx_kit pthread)
endif()
