/*
 * wall-render addition (not upstream).
 *
 * Fullscreen wall output (plan 4.4): a dedicated fullscreen window on the
 * HELIOS head with its own vsync-locked swapchain; each frame the warped
 * canvas is blitted 1:1 into it. The operator UI keeps its own window on
 * the other display.
 *
 * Bring-up note: presentation happens in onPreRender with the canvas the
 * GPU wrote during the PREVIOUS frame's submit (nvapp has no post-submit
 * hook), costing one frame of wall latency. The production output loop
 * will own the frame loop and reclaim it.
 */
#pragma once

#include <cstdint>
#include <vector>
#include <vulkan/vulkan_core.h>

#include <nvvk/resources.hpp>

struct GLFWwindow;

namespace vk_gaussian_splatting {

class WallOutput
{
public:
  // Creates the fullscreen window on the monitor matching canvasW x canvasH
  // (falls back to the primary monitor) and the swapchain. Check valid().
  WallOutput(VkInstance instance, VkPhysicalDevice physical, VkDevice device, const nvvk::QueueInfo& queue, uint32_t canvasW, uint32_t canvasH);
  ~WallOutput();
  WallOutput(const WallOutput&) = delete;
  WallOutput& operator=(const WallOutput&) = delete;

  bool valid() const { return m_swapchain != VK_NULL_HANDLE; }

  // True when the user closed the output window (Alt+F4); owner should destroy.
  bool closeRequested() const;

  // Blit the canvas (GENERAL layout, written by a previous submit on the same
  // queue) into the next swapchain image and present it. Synchronous with at
  // most one frame in flight. Returns false on swapchain loss.
  bool present(VkImage canvas, VkExtent2D canvasSize);

private:
  bool createSwapchain();
  void destroySwapchain();

  VkInstance       m_instance{VK_NULL_HANDLE};
  VkPhysicalDevice m_physical{VK_NULL_HANDLE};
  VkDevice         m_device{VK_NULL_HANDLE};
  nvvk::QueueInfo  m_queue{};

  GLFWwindow*  m_window{nullptr};
  VkSurfaceKHR m_surface{VK_NULL_HANDLE};

  VkSwapchainKHR           m_swapchain{VK_NULL_HANDLE};
  VkFormat                 m_format{VK_FORMAT_UNDEFINED};
  VkExtent2D               m_extent{};
  std::vector<VkImage>     m_images;
  std::vector<VkSemaphore> m_renderDone;  // per swapchain image

  VkCommandPool   m_cmdPool{VK_NULL_HANDLE};
  VkCommandBuffer m_cmd{VK_NULL_HANDLE};
  VkSemaphore     m_acquireSem{VK_NULL_HANDLE};
  VkFence         m_fence{VK_NULL_HANDLE};
};

}  // namespace vk_gaussian_splatting
