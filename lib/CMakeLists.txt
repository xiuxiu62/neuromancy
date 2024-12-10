file(GLOB_RECURSE LIB_SOURCES "src/*.c" "src/*.cpp")
file(GLOB_RECURSE LIB_HEADERS "include/*.h" "include/*.hpp")

# Create the neural network library
add_library(neuromancy-lib
  ${LIB_SOURCES}
  ${LIB_HEADERS}
)

add_executable(neuromancy-lib-dev
  ${LIB_SOURCES}
  ${LIB_HEADERS}
)

target_link_libraries(neuromancy-lib-dev PRIVATE
  neuromancy-lib
)

# Add OpenCL dependencies
add_subdirectory(
  ${CMAKE_SOURCE_DIR}/deps/OpenCL-SDK/external/OpenCL-Headers
  ${CMAKE_BINARY_DIR}/deps/OpenCL-Headers
  EXCLUDE_FROM_ALL
)
add_subdirectory(
  ${CMAKE_SOURCE_DIR}/deps/OpenCL-SDK/external/OpenCL-ICD-Loader
  ${CMAKE_BINARY_DIR}/deps/OpenCL-ICD-Loader
  EXCLUDE_FROM_ALL
)

target_include_directories(neuromancy-lib 
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/deps/OpenCL-SDK/external/OpenCL-Headers
    ${CMAKE_SOURCE_DIR}/deps/OpenCL-SDK/external/OpenCL-CLHPP/include
)

target_link_libraries(neuromancy-lib PUBLIC
  OpenCL::OpenCL
)
