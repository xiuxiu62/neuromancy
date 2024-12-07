cmake_minimum_required(VERSION 3.12)

project(
  example
  VERSION 0.1.0
  LANGUAGES C CXX
)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  add_compile_definitions(DEBUG)
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
  add_compile_definitions(RELEASE)
endif()

file(GLOB_RECURSE SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.c" "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc" "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
file(GLOB_RECURSE HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.h" "${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp")

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/deps/OpenCL-SDK/external/OpenCL-Headers)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/deps/OpenCL-SDK/external/OpenCL-ICD-Loader)

# set(IMGUI_DIR ${CMAKE_CURRENT_SOURCE_DIR}/deps/imgui)
# set(IMGUI_SOURCES 
#   ${IMGUI_DIR}/imgui.cpp
#   ${IMGUI_DIR}/imgui_draw.cpp
#   ${IMGUI_DIR}/imgui_tables.cpp
#   ${IMGUI_DIR}/imgui_widgets.cpp
#   ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
#   ${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp
# )

add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS} 
  # ${IMGUI_SOURCES}
)

# add_subdirectory(deps/glfw)
# add_subdirectory(deps/glad)

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/deps/OpenCL-SDK/external/OpenCL-Headers
    ${CMAKE_CURRENT_SOURCE_DIR}/deps/OpenCL-SDK/external/OpenCL-CLHPP/include
    # ${GLFW_INCLUDE_DIR}
    # ${GLAD_INCLUDE_DIR}
    # ${IMGUI_DIR}
    # ${IMGUI_DIR}/backends
)

target_link_libraries(${PROJECT_NAME} PRIVATE
  OpenCL::OpenCL
)

# target_link_libraries(${PROJECT_NAME}
#   glfw
#   glad
# )

# # Force include GLAD before any other code
# if(MSVC)
#   target_compile_options(${PROJECT_NAME} PRIVATE /FI"glad/glad.h")
# else()
#   target_compile_options(${PROJECT_NAME} PRIVATE -include glad/glad.h)
# endif()

# target_compile_definitions(${PROJECT_NAME} PRIVATE GLAD_GL_IMPLEMENTATION)
