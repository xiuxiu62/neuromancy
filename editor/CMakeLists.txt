file(GLOB_RECURSE EDITOR_SOURCES "src/*.c" "src/*.cpp")
file(GLOB_RECURSE EDITOR_HEADERS "include/*.h" "include/*.hpp")

# Find Vulkan
find_package(Vulkan REQUIRED)

# Add GLFW
add_subdirectory(
  ${CMAKE_SOURCE_DIR}/deps/glfw
  ${CMAKE_BINARY_DIR}/deps/glfw
  EXCLUDE_FROM_ALL
)

# Setup ImGui sources
set(IMGUI_DIR ${CMAKE_SOURCE_DIR}/deps/imgui)
set(IMGUI_SOURCES 
  ${IMGUI_DIR}/imgui.cpp
  ${IMGUI_DIR}/imgui_draw.cpp
  ${IMGUI_DIR}/imgui_tables.cpp
  ${IMGUI_DIR}/imgui_widgets.cpp
  ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
  ${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp
)

# Create the editor executable
add_executable(neuromancy-editor
  ${EDITOR_SOURCES}
  ${EDITOR_HEADERS}
  ${IMGUI_SOURCES}
)

target_include_directories(neuromancy-editor PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${Vulkan_INCLUDE_DIRS}
  ${IMGUI_DIR}
  ${IMGUI_DIR}/backends
)

target_link_libraries(neuromancy-editor PRIVATE
  neuromancy-lib
  Vulkan::Vulkan
  glfw
)

# Shader compilation
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
    list(APPEND SHADER_BINARY_FILES "${CMAKE_BINARY_DIR}/shaders/${SHADER_NAME}.spv")
  endforeach()

  add_custom_target(shaders ALL DEPENDS ${SHADER_BINARY_FILES})
  add_dependencies(neuromancy-editor shaders)
endif()
