#
# Created by KYL.ai on 2021-03-17.
#

cmake_minimum_required(VERSION 3.4.1)

## Declares and names the project.
#project("facetool")

set(distribution_DIR ${CMAKE_SOURCE_DIR}/../distribution)

add_library(libonnxruntime SHARED IMPORTED)
add_library(lib_opencv SHARED IMPORTED)

set_target_properties(libonnxruntime PROPERTIES IMPORTED_LOCATION ${distribution_DIR}/lib/${ANDROID_ABI}/libonnxruntime.so)
set_target_properties(lib_opencv PROPERTIES IMPORTED_LOCATION ${distribution_DIR}/lib/${ANDROID_ABI}/libopencv_java4.so)

add_library( # Sets the name of the library.
             native-lib

             # Sets the library as a shared library.
             SHARED

             # Provides a relative path to your source file(s).
             src/main/cpp/native-lib.cpp  src/main/cpp/onnxruntime_inference.cpp)

include_directories(native-lib PRIVATE ${distribution_DIR}/include)
include_directories(${distribution_DIR}/include/onnxruntime/core/session)
include_directories(${distribution_DIR}/include/opencv2)

find_library( # Sets the name of the path variable.
              log-lib

              # Specifies the name of the NDK library that
              # you want CMake to locate.
              log )
find_library( # Sets the name of the path variable.
        jnigraphics-lib

        # Specifies the name of the NDK library that
        # you want CMake to locate.
        jnigraphics )

# Specifies libraries CMake should link to your target library. You
# can link multiple libraries, such as libraries you define in this
# build script, prebuilt third-party libraries, or system libraries.

target_link_libraries( # Specifies the target library.
                       native-lib
                       libonnxruntime
                       lib_opencv

                       # Links the target library to the log library
                       # included in the NDK.
                       ${log-lib}
                       ${jnigraphics-lib})