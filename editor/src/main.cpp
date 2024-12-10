#include "core/logger.h"
#include "core/types.h"

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <cstdlib>
#include <cstring>
#include <vulkan/vulkan_core.h>

struct GraphicsContext {
    VkInstance instance;
    VkPhysicalDevice physical_device;
    VkDevice logical_device;
    VkQueue graphics_queue;
    VkQueue present_queue;
    GLFWwindow *window;
    VkSurfaceKHR surface;
    VkDebugUtilsMessengerEXT debug_messenger;
    bool framebuffer_resized;
};

#ifdef DEBUG
const bool enable_validation_layers = true;
#else
const bool enable_validation_layers = false;
#endif

const char *validation_layers[] = {
    "VK_LAYER_KHRONOS_validation",
};

static VKAPI_ATTR VkBool32 VKAPI_CALL debug_callback(VkDebugUtilsMessageSeverityFlagBitsEXT message_severity,
                                                     VkDebugUtilsMessageTypeFlagsEXT message_type,
                                                     const VkDebugUtilsMessengerCallbackDataEXT *callback_data,
                                                     void *user_data);

bool init(GraphicsContext &ctx, GLFWwindow *window, const char *title);
void deinit(GraphicsContext &ctx);

bool init(GLFWwindow *&window, const char *title, u32 width, u32 height);
void deinit(GLFWwindow *window);

#define TITLE "neuromancy"
#define WIDTH 1920
#define HEIGHT 1080

#define MAX_FRAMES_IN_FLIGHT 2

int main() {
    GLFWwindow *window;
    GraphicsContext ctx;

    if (!init(window, TITLE, WIDTH, HEIGHT)) {
        return 1;
    }
    if (!init(ctx, window, TITLE)) {
        return 1;
    }

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
    }

    deinit(ctx);
    return 0;
}

static bool check_validation_layer_support() {
    u32 layer_count;
    vkEnumerateInstanceLayerProperties(&layer_count, nullptr);

    VkLayerProperties *available_layers =
        static_cast<VkLayerProperties *>(malloc(sizeof(VkLayerProperties) * layer_count));
    vkEnumerateInstanceLayerProperties(&layer_count, available_layers);

    for (usize i = 0; i < sizeof(validation_layers) / sizeof(const char *); i++) {
        bool layer_found = false;

        for (usize j = 0; j < layer_count; j++) {
            if (strcmp(validation_layers[i], available_layers[j].layerName) == 0) {
                layer_found = true;
                break;
            }
        }

        if (!layer_found) {
            free(available_layers);
            return false;
        }
    }

    free(available_layers);
    return true;
}

static bool init_instance(GraphicsContext &ctx, const char *title) {
    if (enable_validation_layers && !check_validation_layer_support()) {
        error("Validation layers requested were not available");
        return false;
    }

    VkApplicationInfo app_info = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = title,
        .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
        .pEngineName = "No Engine",
        .engineVersion = VK_MAKE_VERSION(1, 0, 0),
        .apiVersion = VK_API_VERSION_1_0,
    };

    VkInstanceCreateInfo create_info = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &app_info,
    };

    // Get required extensions
    u32 glfw_extension_count = 0;
    const char **glfw_extensions = glfwGetRequiredInstanceExtensions(&glfw_extension_count);

    create_info.enabledExtensionCount = glfw_extension_count;
    create_info.ppEnabledExtensionNames = glfw_extensions;

    if (enable_validation_layers) {
        create_info.enabledLayerCount = sizeof(validation_layers) / sizeof(const char *);
        create_info.ppEnabledLayerNames = validation_layers;
    }

    if (vkCreateInstance(&create_info, nullptr, &ctx.instance) != VK_SUCCESS) {
        error("Failed to create Vulkan instance");
        return false;
    }

    return true;
}

static bool init_surface(GraphicsContext &ctx) {
    if (glfwCreateWindowSurface(ctx.instance, ctx.window, nullptr, &ctx.surface) != VK_SUCCESS) {
        error("Failed to create window surface");
        return false;
    }

    return true;
}

static bool pick_physical_device(GraphicsContext &ctx) {
    u32 device_count = 0;
    vkEnumeratePhysicalDevices(ctx.instance, &device_count, nullptr);

    if (device_count == 0) {
        error("Failed to find physical device with Vulkan support");
        return false;
    }

    VkPhysicalDevice *devices = static_cast<VkPhysicalDevice *>(malloc(sizeof(VkPhysicalDevice) * device_count));
    vkEnumeratePhysicalDevices(ctx.instance, &device_count, devices);

    // TODO: score devices and pick the highest scoring one
    ctx.physical_device = devices[0];

    free(devices);

    return true;
}

static void frame_buffer_resize_callback(GLFWwindow *window, int width, int height) {
    GraphicsContext &ctx =
        static_cast<GraphicsContext &>(*static_cast<GraphicsContext *>(glfwGetWindowUserPointer(window)));
    ctx.framebuffer_resized = true;
}

bool init(GraphicsContext &ctx, GLFWwindow *window, const char *title) {
#define ret_on_fail(...)                                                                                               \
    if (!__VA_ARGS__) return false;

    memset(&ctx, 0, sizeof(GraphicsContext));

    ctx.window = window;

    glfwSetWindowUserPointer(window, &ctx);
    glfwSetFramebufferSizeCallback(window, frame_buffer_resize_callback);

    // Initialize Vulkan
    ret_on_fail(init_instance(ctx, title));
    ret_on_fail(init_surface(ctx));
    ret_on_fail(pick_physical_device(ctx));

    return true;
}

void deinit(GraphicsContext &ctx) {
    if (ctx.surface) vkDestroySurfaceKHR(ctx.instance, ctx.surface, nullptr);
    if (ctx.instance) vkDestroyInstance(ctx.instance, nullptr);
}

bool init(GLFWwindow *&window, const char *title, u32 width, u32 height) {
    if (!glfwInit()) {
        error("Failed to initialize GLFW");
        return false;
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    window = glfwCreateWindow(width, height, title, nullptr, nullptr);
    if (!window) {
        error("Failed to create window");
        return false;
    }

    return true;
}
void deinit(GLFWwindow *window) {
    glfwDestroyWindow(window);
    glfwTerminate();
}

static VKAPI_ATTR VkBool32 VKAPI_CALL debug_callback(VkDebugUtilsMessageSeverityFlagBitsEXT message_severity,
                                                     VkDebugUtilsMessageTypeFlagsEXT message_type,
                                                     const VkDebugUtilsMessengerCallbackDataEXT *callback_data,
                                                     void *user_data) {
    warn("[VK] %s", callback_data->pMessage);
    return VK_FALSE;
}
