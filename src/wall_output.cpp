/*
 * wall-render addition (not upstream). See wall_output.h.
 */
#include "wall_output.h"

#include <volk.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <nvutils/logger.hpp>
#include <nvvk/check_error.hpp>

namespace vk_gaussian_splatting {

WallOutput::WallOutput(VkInstance             instance,
                       VkPhysicalDevice       physical,
                       VkDevice               device,
                       const nvvk::QueueInfo& queue,
                       uint32_t               canvasW,
                       uint32_t               canvasH)
    : m_instance(instance)
    , m_physical(physical)
    , m_device(device)
    , m_queue(queue)
{
  // Pick the monitor whose current mode matches the canvas (the HELIOS head),
  // else the primary.
  GLFWmonitor* monitor = glfwGetPrimaryMonitor();
  int          count   = 0;
  GLFWmonitor** monitors = glfwGetMonitors(&count);
  for(int i = 0; i < count; i++)
  {
    const GLFWvidmode* mode = glfwGetVideoMode(monitors[i]);
    if(mode && uint32_t(mode->width) == canvasW && uint32_t(mode->height) == canvasH)
    {
      monitor = monitors[i];
      break;
    }
  }
  const GLFWvidmode* mode = glfwGetVideoMode(monitor);
  if(!mode)
  {
    LOGE("WallOutput: no video mode on target monitor\n");
    return;
  }

  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  glfwWindowHint(GLFW_AUTO_ICONIFY, GLFW_FALSE);  // NEVER minimize the wall on focus loss
  glfwWindowHint(GLFW_FOCUSED, GLFW_FALSE);
  glfwWindowHint(GLFW_FOCUS_ON_SHOW, GLFW_FALSE);
  m_window = glfwCreateWindow(mode->width, mode->height, "Wall Output", monitor, nullptr);
  if(!m_window)
  {
    LOGE("WallOutput: window creation failed\n");
    return;
  }

  if(glfwCreateWindowSurface(m_instance, m_window, nullptr, &m_surface) != VK_SUCCESS)
  {
    LOGE("WallOutput: surface creation failed\n");
    glfwDestroyWindow(m_window);
    m_window = nullptr;
    return;
  }

  VkBool32 supported = VK_FALSE;
  vkGetPhysicalDeviceSurfaceSupportKHR(m_physical, m_queue.familyIndex, m_surface, &supported);
  if(!supported)
  {
    LOGE("WallOutput: queue family cannot present to the wall surface\n");
    return;
  }

  const VkCommandPoolCreateInfo poolInfo{
      .sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
      .queueFamilyIndex = m_queue.familyIndex,
  };
  NVVK_CHECK(vkCreateCommandPool(m_device, &poolInfo, nullptr, &m_cmdPool));
  const VkCommandBufferAllocateInfo cmdInfo{
      .sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool        = m_cmdPool,
      .level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1,
  };
  NVVK_CHECK(vkAllocateCommandBuffers(m_device, &cmdInfo, &m_cmd));

  const VkSemaphoreCreateInfo semInfo{.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
  NVVK_CHECK(vkCreateSemaphore(m_device, &semInfo, nullptr, &m_acquireSem));
  const VkFenceCreateInfo fenceInfo{.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, .flags = VK_FENCE_CREATE_SIGNALED_BIT};
  NVVK_CHECK(vkCreateFence(m_device, &fenceInfo, nullptr, &m_fence));

  if(createSwapchain())
    LOGI("WallOutput: fullscreen %ux%u on the wall head, format %d\n", m_extent.width, m_extent.height, int(m_format));
}

bool WallOutput::createSwapchain()
{
  VkSurfaceCapabilitiesKHR caps{};
  vkGetPhysicalDeviceSurfaceCapabilitiesKHR(m_physical, m_surface, &caps);

  uint32_t formatCount = 0;
  vkGetPhysicalDeviceSurfaceFormatsKHR(m_physical, m_surface, &formatCount, nullptr);
  std::vector<VkSurfaceFormatKHR> formats(formatCount);
  vkGetPhysicalDeviceSurfaceFormatsKHR(m_physical, m_surface, &formatCount, formats.data());
  VkSurfaceFormatKHR pick = formats[0];
  for(const auto& f : formats)
    if(f.format == VK_FORMAT_B8G8R8A8_UNORM && f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
      pick = f;

  m_format = pick.format;
  m_extent = caps.currentExtent;

  uint32_t imageCount = caps.minImageCount + 1;
  if(caps.maxImageCount > 0 && imageCount > caps.maxImageCount)
    imageCount = caps.maxImageCount;

  const VkSwapchainCreateInfoKHR scInfo{
      .sType            = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
      .surface          = m_surface,
      .minImageCount    = imageCount,
      .imageFormat      = m_format,
      .imageColorSpace  = pick.colorSpace,
      .imageExtent      = m_extent,
      .imageArrayLayers = 1,
      .imageUsage       = VK_IMAGE_USAGE_TRANSFER_DST_BIT,
      .imageSharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .preTransform     = caps.currentTransform,
      .compositeAlpha   = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
      .presentMode      = VK_PRESENT_MODE_FIFO_KHR,  // vsync-locked, no tearing
      .clipped          = VK_TRUE,
  };
  if(vkCreateSwapchainKHR(m_device, &scInfo, nullptr, &m_swapchain) != VK_SUCCESS)
  {
    LOGE("WallOutput: swapchain creation failed\n");
    m_swapchain = VK_NULL_HANDLE;
    return false;
  }

  uint32_t n = 0;
  vkGetSwapchainImagesKHR(m_device, m_swapchain, &n, nullptr);
  m_images.resize(n);
  vkGetSwapchainImagesKHR(m_device, m_swapchain, &n, m_images.data());

  const VkSemaphoreCreateInfo semInfo{.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
  m_renderDone.resize(n);
  for(auto& s : m_renderDone)
    NVVK_CHECK(vkCreateSemaphore(m_device, &semInfo, nullptr, &s));
  return true;
}

void WallOutput::destroySwapchain()
{
  for(auto& s : m_renderDone)
    vkDestroySemaphore(m_device, s, nullptr);
  m_renderDone.clear();
  if(m_swapchain != VK_NULL_HANDLE)
    vkDestroySwapchainKHR(m_device, m_swapchain, nullptr);
  m_swapchain = VK_NULL_HANDLE;
  m_images.clear();
}

WallOutput::~WallOutput()
{
  if(m_device != VK_NULL_HANDLE)
    vkDeviceWaitIdle(m_device);
  destroySwapchain();
  if(m_fence != VK_NULL_HANDLE)
    vkDestroyFence(m_device, m_fence, nullptr);
  if(m_acquireSem != VK_NULL_HANDLE)
    vkDestroySemaphore(m_device, m_acquireSem, nullptr);
  if(m_cmdPool != VK_NULL_HANDLE)
    vkDestroyCommandPool(m_device, m_cmdPool, nullptr);
  if(m_surface != VK_NULL_HANDLE)
    vkDestroySurfaceKHR(m_instance, m_surface, nullptr);
  if(m_window)
    glfwDestroyWindow(m_window);
}

bool WallOutput::closeRequested() const
{
  return m_window && glfwWindowShouldClose(m_window);
}

bool WallOutput::present(VkImage canvas, VkExtent2D canvasSize)
{
  if(m_swapchain == VK_NULL_HANDLE)
    return false;

  NVVK_CHECK(vkWaitForFences(m_device, 1, &m_fence, VK_TRUE, UINT64_MAX));

  uint32_t imageIndex = 0;
  VkResult r = vkAcquireNextImageKHR(m_device, m_swapchain, UINT64_MAX, m_acquireSem, VK_NULL_HANDLE, &imageIndex);
  if(r == VK_ERROR_OUT_OF_DATE_KHR)
  {
    vkDeviceWaitIdle(m_device);
    destroySwapchain();
    return createSwapchain();
  }
  if(r != VK_SUCCESS && r != VK_SUBOPTIMAL_KHR)
    return false;

  NVVK_CHECK(vkResetFences(m_device, 1, &m_fence));

  const VkCommandBufferBeginInfo begin{.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                       .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
  NVVK_CHECK(vkBeginCommandBuffer(m_cmd, &begin));

  // canvas: warp compute wrote it in an earlier submit on this queue
  VkMemoryBarrier canvasBarrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  canvasBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  canvasBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
  vkCmdPipelineBarrier(m_cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1,
                       &canvasBarrier, 0, nullptr, 0, nullptr);

  VkImageMemoryBarrier toDst{
      .sType            = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
      .srcAccessMask    = 0,
      .dstAccessMask    = VK_ACCESS_TRANSFER_WRITE_BIT,
      .oldLayout        = VK_IMAGE_LAYOUT_UNDEFINED,
      .newLayout        = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .image            = m_images[imageIndex],
      .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
  };
  vkCmdPipelineBarrier(m_cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                       nullptr, 1, &toDst);

  VkImageBlit blit{
      .srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
      .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
  };
  blit.srcOffsets[1] = {int32_t(canvasSize.width), int32_t(canvasSize.height), 1};
  blit.dstOffsets[1] = {int32_t(m_extent.width), int32_t(m_extent.height), 1};
  vkCmdBlitImage(m_cmd, canvas, VK_IMAGE_LAYOUT_GENERAL, m_images[imageIndex], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                 &blit, VK_FILTER_LINEAR);

  VkImageMemoryBarrier toPresent = toDst;
  toPresent.srcAccessMask        = VK_ACCESS_TRANSFER_WRITE_BIT;
  toPresent.dstAccessMask        = 0;
  toPresent.oldLayout            = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  toPresent.newLayout            = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
  vkCmdPipelineBarrier(m_cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, nullptr, 0,
                       nullptr, 1, &toPresent);

  NVVK_CHECK(vkEndCommandBuffer(m_cmd));

  const VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
  const VkSubmitInfo         submit{
      .sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .waitSemaphoreCount   = 1,
      .pWaitSemaphores      = &m_acquireSem,
      .pWaitDstStageMask    = &waitStage,
      .commandBufferCount   = 1,
      .pCommandBuffers      = &m_cmd,
      .signalSemaphoreCount = 1,
      .pSignalSemaphores    = &m_renderDone[imageIndex],
  };
  NVVK_CHECK(vkQueueSubmit(m_queue.queue, 1, &submit, m_fence));

  const VkPresentInfoKHR present{
      .sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
      .waitSemaphoreCount = 1,
      .pWaitSemaphores    = &m_renderDone[imageIndex],
      .swapchainCount     = 1,
      .pSwapchains        = &m_swapchain,
      .pImageIndices      = &imageIndex,
  };
  r = vkQueuePresentKHR(m_queue.queue, &present);
  if(r == VK_ERROR_OUT_OF_DATE_KHR || r == VK_SUBOPTIMAL_KHR)
  {
    vkDeviceWaitIdle(m_device);
    destroySwapchain();
    return createSwapchain();
  }
  return r == VK_SUCCESS;
}

}  // namespace vk_gaussian_splatting
