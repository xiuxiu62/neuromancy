cmake_minimum_required(VERSION 3.12)

project(
  nm
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

find_package(Vulkan REQUIRED)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/deps/OpenCL-SDK/external/OpenCL-Headers)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/deps/OpenCL-SDK/external/OpenCL-ICD-Loader)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/deps/glfw)

set(IMGUI_DIR ${CMAKE_CURRENT_SOURCE_DIR}/deps/imgui)
set(IMGUI_SOURCES 
  ${IMGUI_DIR}/imgui.cpp
  ${IMGUI_DIR}/imgui_draw.cpp
  ${IMGUI_DIR}/imgui_tables.cpp
  ${IMGUI_DIR}/imgui_widgets.cpp
  ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
  ${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp
)

add_executable(${PROJECT_NAME} 
  ${SOURCES} 
  ${HEADERS} 
  ${IMGUI_SOURCES}
)

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/deps/OpenCL-SDK/external/OpenCL-Headers
    ${CMAKE_CURRENT_SOURCE_DIR}/deps/OpenCL-SDK/external/OpenCL-CLHPP/include
    ${Vulkan_INCLUDE_DIRS}
    ${IMGUI_DIR}
    ${IMGUI_DIR}/backends
)

target_link_libraries(${PROJECT_NAME} PRIVATE
  OpenCL::OpenCL
  Vulkan::Vulkan
  glfw
)

file(GLOB SHADER_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/shaders/*")

find_program(GLSLC glslc)
if(GLSLC)
  foreach(SHADER ${SHADER_SOURCES})
    get_filename_component(SHADER_NAME ${SHADER} NAME)
    add_custom_command(
      OUTPUT "${CMAKE_BINARY_DIR}/shaders/${SHADER_NAME}.spv"
      COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/shaders/"
      COMMAND ${GLSLC} ${SHADER} -o "${CMAKE_BINARY_DIR}/shaders/${SHADER_NAME}.spv"
      DEPENDS ${SHADER}
      COMMENT "Compiling shader ${SHADER_NAME}"
    )
  endforeach()

  add_custom_target(shaders ALL DEPENDS ${SHADER_BINARY_FILES})
  add_dependencies(${PROJECT_NAME} shaders)
endif()
