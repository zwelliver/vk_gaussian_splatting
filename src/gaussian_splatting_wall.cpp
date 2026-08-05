/*
 * wall-render addition (not upstream).
 *
 * N simultaneous off-axis wall views + warp to the HELIOS canvas
 * (plan 4.2/4.3, integration steps 3-4).
 *
 * Per frame: for each wall view, the shared FrameInfo UBO is rewritten in
 * command order (vkCmdUpdateBuffer), the GPU dist/cull + radix sort run for
 * that view, and the splats are rasterized into that view's offscreen
 * target. A final compute pass warps all views onto the 4800x1920 canvas
 * through the exact per-column wall geometry. Runs before the regular
 * pipeline in onRender, which re-uploads the UBO for the interactive view.
 *
 * Self-test mode replaces the splat renders with an analytic direction
 * pattern and compares the warped canvas against ground truth evaluated
 * per canvas pixel — the GPU twin of phase0_sim's exactness gate.
 */
#include <cstring>

#include "gaussian_splatting.h"
#include "wall_shaderio.h"

namespace vk_gaussian_splatting {

namespace {

// One barrier after a vkCmdUpdateBuffer so every consuming stage sees it.
void cmdBarrierUboUpload(VkCommandBuffer cmd)
{
  VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  barrier.dstAccessMask = VK_ACCESS_UNIFORM_READ_BIT;
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                       VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT
                           | VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                       0, 1, &barrier, 0, nullptr, 0, nullptr);
}

inline glm::vec4 toV4(const Vec3& v, float scale = 1.0f)
{
  return {float(v.x) * scale, float(v.y) * scale, float(v.z) * scale, 0.0f};
}

}  // namespace

void GaussianSplatting::ensureWallResources()
{
  auto& st = m_wallRender;
  if(st.initialized && st.colorFormat == prmRender.colorFormat)
    return;

  if(st.initialized)
  {
    // color format changed: view targets must match the raster pipeline
    vkDeviceWaitIdle(m_device);
    st.viewBuffers.deinit();
    st.canvas.deinit();
  }
  else
  {
    // one-time objects ------------------------------------------------------
    VkSamplerCreateInfo samplerInfo{
        .sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .magFilter    = VK_FILTER_LINEAR,
        .minFilter    = VK_FILTER_LINEAR,
        .mipmapMode   = VK_SAMPLER_MIPMAP_MODE_NEAREST,
        .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
    };
    NVVK_CHECK(vkCreateSampler(m_device, &samplerInfo, nullptr, &st.linearSampler));

    m_alloc.createBuffer(st.infoBuffer, sizeof(shaderio::WallWarpInfo),
                         VK_BUFFER_USAGE_2_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT,
                         VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
    m_alloc.createBuffer(st.errBuffer, sizeof(shaderio::WallErr),
                         VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT,
                         VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                         VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT);
    NVVK_DBG_NAME(st.infoBuffer.buffer);
    NVVK_DBG_NAME(st.errBuffer.buffer);

    st.bindings.clear();
    st.bindings.addBinding(WALL_BINDING_INFO_UBO, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
    st.bindings.addBinding(WALL_BINDING_VIEW_TEXTURES, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, WALL_MAX_VIEWS,
                           VK_SHADER_STAGE_COMPUTE_BIT);
    st.bindings.addBinding(WALL_BINDING_CANVAS_IMAGE, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT);
    st.bindings.addBinding(WALL_BINDING_ERR_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
    st.bindings.addBinding(WALL_BINDING_VIEW_IMAGES, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, WALL_MAX_VIEWS, VK_SHADER_STAGE_COMPUTE_BIT);
    NVVK_CHECK(st.bindings.createDescriptorSetLayout(m_device, 0, &st.dsetLayout));

    std::vector<VkDescriptorPoolSize> poolSize;
    st.bindings.appendPoolSizes(poolSize);
    VkDescriptorPoolCreateInfo poolInfo{
        .sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets       = 1,
        .poolSizeCount = uint32_t(poolSize.size()),
        .pPoolSizes    = poolSize.data(),
    };
    NVVK_CHECK(vkCreateDescriptorPool(m_device, &poolInfo, nullptr, &st.dsetPool));

    VkDescriptorSetAllocateInfo allocInfo{
        .sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool     = st.dsetPool,
        .descriptorSetCount = 1,
        .pSetLayouts        = &st.dsetLayout,
    };
    NVVK_CHECK(vkAllocateDescriptorSets(m_device, &allocInfo, &st.dset));

    const VkPushConstantRange pcRange{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(shaderio::WallPush)};

    VkPipelineLayoutCreateInfo plInfo{
        .sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount         = 1,
        .pSetLayouts            = &st.dsetLayout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges    = &pcRange,
    };
    NVVK_CHECK(vkCreatePipelineLayout(m_device, &plInfo, nullptr, &st.pipeLayout));
  }

  // view + canvas targets ---------------------------------------------------
  st.viewBuffers.init({
      .allocator    = &m_alloc,
      .colorFormats = {prmRender.colorFormat, prmRender.colorFormat, prmRender.colorFormat, prmRender.colorFormat,
                       prmRender.colorFormat},
      .depthFormat  = m_depthFormat,
      .imageSampler = st.linearSampler,
      .descriptorPool = m_app->getTextureDescriptorPool(),
  });
  st.canvas.init({
      .allocator      = &m_alloc,
      .colorFormats   = {VK_FORMAT_R16G16B16A16_SFLOAT},
      .depthFormat    = m_depthFormat,
      .imageSampler   = st.linearSampler,
      .descriptorPool = m_app->getTextureDescriptorPool(),
  });

  VkCommandBuffer cmd = m_app->createTempCmdBuffer();
  NVVK_CHECK(st.viewBuffers.update(cmd, {uint32_t(st.viewRes), uint32_t(st.viewRes)}));
  NVVK_CHECK(st.canvas.update(cmd, {uint32_t(WallGeometry::canvasW()), uint32_t(WallGeometry::canvasH())}));
  m_app->submitAndWaitTempCmdBuffer(cmd);

  // descriptor writes -------------------------------------------------------
  VkDescriptorImageInfo texInfos[WALL_MAX_VIEWS], imgInfos[WALL_MAX_VIEWS];
  for(int i = 0; i < WALL_MAX_VIEWS; i++)
  {
    texInfos[i] = {st.linearSampler, st.viewBuffers.getColorImageView(i), VK_IMAGE_LAYOUT_GENERAL};
    imgInfos[i] = {VK_NULL_HANDLE, st.viewBuffers.getColorImageView(i), VK_IMAGE_LAYOUT_GENERAL};
  }
  const VkDescriptorBufferInfo uboInfo{st.infoBuffer.buffer, 0, VK_WHOLE_SIZE};
  const VkDescriptorBufferInfo errInfo{st.errBuffer.buffer, 0, VK_WHOLE_SIZE};
  const VkDescriptorImageInfo  canvasInfo{VK_NULL_HANDLE, st.canvas.getColorImageView(0), VK_IMAGE_LAYOUT_GENERAL};

  std::array<VkWriteDescriptorSet, 5> writes{};
  for(auto& w : writes)
    w = {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = st.dset, .descriptorCount = 1};
  writes[0].dstBinding      = WALL_BINDING_INFO_UBO;
  writes[0].descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  writes[0].pBufferInfo     = &uboInfo;
  writes[1].dstBinding      = WALL_BINDING_VIEW_TEXTURES;
  writes[1].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  writes[1].descriptorCount = WALL_MAX_VIEWS;
  writes[1].pImageInfo      = texInfos;
  writes[2].dstBinding      = WALL_BINDING_CANVAS_IMAGE;
  writes[2].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
  writes[2].pImageInfo      = &canvasInfo;
  writes[3].dstBinding      = WALL_BINDING_ERR_BUFFER;
  writes[3].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  writes[3].pBufferInfo     = &errInfo;
  writes[4].dstBinding      = WALL_BINDING_VIEW_IMAGES;
  writes[4].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
  writes[4].descriptorCount = WALL_MAX_VIEWS;
  writes[4].pImageInfo      = imgInfos;
  vkUpdateDescriptorSets(m_device, uint32_t(writes.size()), writes.data(), 0, nullptr);

  st.colorFormat = prmRender.colorFormat;
  st.initialized = true;
}

void GaussianSplatting::wallCreatePipelines()
{
  auto& st = m_wallRender;
  if(m_shaders.wallWarpShader == VK_NULL_HANDLE || m_shaders.wallPatternShader == VK_NULL_HANDLE)
    return;

  if(st.warpPipeline != VK_NULL_HANDLE)
    vkDestroyPipeline(m_device, st.warpPipeline, nullptr);
  if(st.patternPipeline != VK_NULL_HANDLE)
    vkDestroyPipeline(m_device, st.patternPipeline, nullptr);

  VkComputePipelineCreateInfo pipelineInfo{
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage =
          {
              .sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
              .stage  = VK_SHADER_STAGE_COMPUTE_BIT,
              .module = m_shaders.wallWarpShader,
              .pName  = "main",
          },
      .layout = st.pipeLayout,
  };
  NVVK_CHECK(vkCreateComputePipelines(m_device, m_pipelineCache, 1, &pipelineInfo, nullptr, &st.warpPipeline));
  pipelineInfo.stage.module = m_shaders.wallPatternShader;
  NVVK_CHECK(vkCreateComputePipelines(m_device, m_pipelineCache, 1, &pipelineInfo, nullptr, &st.patternPipeline));
}

void GaussianSplatting::wallDeinit()
{
  auto& st = m_wallRender;
  if(st.warpPipeline != VK_NULL_HANDLE)
    vkDestroyPipeline(m_device, st.warpPipeline, nullptr);
  if(st.patternPipeline != VK_NULL_HANDLE)
    vkDestroyPipeline(m_device, st.patternPipeline, nullptr);
  if(st.pipeLayout != VK_NULL_HANDLE)
    vkDestroyPipelineLayout(m_device, st.pipeLayout, nullptr);
  if(st.dsetPool != VK_NULL_HANDLE)
    vkDestroyDescriptorPool(m_device, st.dsetPool, nullptr);
  if(st.dsetLayout != VK_NULL_HANDLE)
    vkDestroyDescriptorSetLayout(m_device, st.dsetLayout, nullptr);
  if(st.linearSampler != VK_NULL_HANDLE)
    vkDestroySampler(m_device, st.linearSampler, nullptr);
  if(st.initialized)
  {
    st.viewBuffers.deinit();
    st.canvas.deinit();
    m_alloc.destroyBuffer(st.infoBuffer);
    m_alloc.destroyBuffer(st.errBuffer);
  }
  st = WallRenderState{};
}

void GaussianSplatting::renderWallCanvas(VkCommandBuffer cmd, uint32_t splatCount)
{
  auto& st = m_wallRender;
  if(!m_wallViews || !m_wallViews->enabled || !st.canvasEnabled || !m_shaders.valid || !m_wallTracking)
    return;

  const bool selfTest = st.selfTest;
  if(!selfTest)
  {
    if(splatCount == 0)
      return;
    if(!usesGpuDistSort())
    {
      static bool logged = false;
      if(!logged)
        LOGW("Wall canvas requires the GPU radix or stochastic sorting method.\n");
      logged = true;
      return;
    }
    if(isRtxPipelineOnly() || needSurfaceInfo())
    {
      static bool logged = false;
      if(!logged)
        LOGW("Wall canvas requires a raster pipeline with lighting and DoF off.\n");
      logged = true;
      return;
    }
  }

  ensureWallResources();
  if(st.warpPipeline == VK_NULL_HANDLE)
    wallCreatePipelines();
  if(st.warpPipeline == VK_NULL_HANDLE)
    return;

  if(!m_wallViews->rebuild(*m_wallTracking))
    return;  // no pupil yet

  // Previous frames' self-test results (host-visible, ~2 frames latency).
  // Only accept a fully-accumulated frame: the GPU may still be mid-count.
  if(st.errBuffer.mapping)
  {
    const auto* e = reinterpret_cast<const shaderio::WallErr*>(st.errBuffer.mapping);
    if(e->samples == uint32_t(WallGeometry::canvasW()) * uint32_t(WallGeometry::canvasH()))
    {
      const uint32_t bits = e->maxErrBits;
      float          f;
      std::memcpy(&f, &bits, sizeof(f));
      st.lastMaxErr  = f;
      st.lastBad     = e->badCount;
      st.lastSamples = e->samples;
    }
  }

  NVVK_DBG_SCOPE(cmd);
  auto timerSection = m_profilerGpuTimer.cmdFrameSection(cmd, "Wall canvas");

  const int   numViews = m_wallViews->numViews;
  const auto& views    = m_wallViews->views();
  const auto  c2v      = columnToView(views);
  const auto& wall     = m_wallViews->wall();

  // Upload the warp info: views + exact per-column wall geometry (meters).
  shaderio::WallWarpInfo info{};
  for(int i = 0; i < numViews; i++)
  {
    const ViewFrustum& v = views[i];
    info.views[i].eye    = toV4(v.eye, 0.001f);
    info.views[i].right  = toV4(v.right);
    info.views[i].up     = toV4(v.up);
    info.views[i].fwd    = toV4(v.fwd);
    info.views[i].lrbt   = {float(v.l), float(v.r), float(v.b), float(v.t)};
  }
  for(int c = 0; c < WALL_COLUMNS; c++)
  {
    const double px0    = double(c) * WallGeometry::kPanelPx;
    const Vec3   origin = wall.pixelToPoint(px0, WallGeometry::canvasH());
    const Vec3   step   = wall.pixelToPoint(px0 + 1.0, WallGeometry::canvasH()) - origin;
    info.colOrigin[c]   = toV4(origin, 0.001f);
    info.colRight[c]    = toV4(step, 0.001f);
    info.colView[c].x   = c2v[c];
  }
  const float pitchM = float(WallGeometry::kPanelMm / WallGeometry::kPanelPx) * 0.001f;
  info.upStep        = {0.0f, pitchM, 0.0f, 0.0f};
  info.numViews      = numViews;
  info.canvasW       = WallGeometry::canvasW();
  info.canvasH       = WallGeometry::canvasH();
  info.selfTest      = selfTest ? 1 : 0;
  vkCmdUpdateBuffer(cmd, st.infoBuffer.buffer, 0, sizeof(info), &info);
  cmdBarrierUboUpload(cmd);

  // Fill the views ----------------------------------------------------------
  if(selfTest)
  {
    // analytic pattern instead of splats
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, st.patternPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, st.pipeLayout, 0, 1, &st.dset, 0, nullptr);
    for(int i = 0; i < numViews; i++)
    {
      shaderio::WallPush push{.viewIndex = i};
      vkCmdPushConstants(cmd, st.pipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
      const uint32_t wg = (uint32_t(st.viewRes) + 7) / 8;
      vkCmdDispatch(cmd, wg, wg, 1);
    }
    VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, 0, 1,
                         &barrier, 0, nullptr, 0, nullptr);
  }
  else
  {
    // splats: per view — UBO rewrite, GPU sort/cull, raster into the view target
    for(int i = 0; i < numViews; i++)
    {
      shaderio::FrameInfo fi = prmFrame;
      fi.splatCount          = int32_t(splatCount);
      fi.frameSampleId       = 0;
      m_wallViews->fillViewFrameInfo(i, st.viewRes, fi);
      vkCmdUpdateBuffer(cmd, m_frameInfoBuffer.buffer, 0, sizeof(fi), &fi);
      cmdBarrierUboUpload(cmd);

      processSortingOnGPU(cmd, splatCount);

      nvvk::cmdImageMemoryBarrier(cmd, {st.viewBuffers.getColorImage(i), VK_IMAGE_LAYOUT_GENERAL,
                                        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL});

      VkRenderingAttachmentInfo colorAttachment = DEFAULT_VkRenderingAttachmentInfo;
      colorAttachment.imageView                 = st.viewBuffers.getColorImageView(i);
      colorAttachment.clearValue                = {m_clearColor};
      VkRenderingAttachmentInfo depthAttachment = DEFAULT_VkRenderingAttachmentInfo;
      depthAttachment.imageView                 = st.viewBuffers.getDepthImageView();
      depthAttachment.imageLayout               = VK_IMAGE_LAYOUT_GENERAL;
      depthAttachment.clearValue                = {.depthStencil = DEFAULT_VkClearDepthStencilValue};

      VkRenderingInfo renderingInfo      = DEFAULT_VkRenderingInfo;
      renderingInfo.renderArea           = DEFAULT_VkRect2D(VkExtent2D{uint32_t(st.viewRes), uint32_t(st.viewRes)});
      renderingInfo.colorAttachmentCount = 1;
      renderingInfo.pColorAttachments    = &colorAttachment;
      renderingInfo.pDepthAttachment     = &depthAttachment;

      vkCmdBeginRendering(cmd, &renderingInfo);
      const VkViewport viewport{0.0F, 0.0F, float(st.viewRes), float(st.viewRes), 0.0F, 1.0F};
      const VkRect2D   scissor{{0, 0}, {uint32_t(st.viewRes), uint32_t(st.viewRes)}};
      vkCmdSetViewportWithCount(cmd, 1, &viewport);
      vkCmdSetScissorWithCount(cmd, 1, &scissor);
      drawSplatPrimitives(cmd, splatCount);
      vkCmdEndRendering(cmd);

      nvvk::cmdImageMemoryBarrier(cmd, {st.viewBuffers.getColorImage(i), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                        VK_IMAGE_LAYOUT_GENERAL});
    }
  }

  // Warp to the canvas ------------------------------------------------------
  if(selfTest)
  {
    vkCmdFillBuffer(cmd, st.errBuffer.buffer, 0, sizeof(shaderio::WallErr), 0);
    VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, 0, 1, &barrier,
                         0, nullptr, 0, nullptr);
  }

  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, st.warpPipeline);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, st.pipeLayout, 0, 1, &st.dset, 0, nullptr);
  shaderio::WallPush push{.viewIndex = 0};
  vkCmdPushConstants(cmd, st.pipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
  vkCmdDispatch(cmd, (uint32_t(WallGeometry::canvasW()) + 7) / 8, (uint32_t(WallGeometry::canvasH()) + 7) / 8, 1);

  // canvas: compute write -> ImGui fragment sampling
  VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, 0, 1,
                       &barrier, 0, nullptr, 0, nullptr);
}

}  // namespace vk_gaussian_splatting
