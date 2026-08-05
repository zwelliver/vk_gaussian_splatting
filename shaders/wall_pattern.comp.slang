/*
 * wall-render addition (not upstream).
 *
 * Self-test view fill: writes the analytic direction pattern into one view
 * texture, using the exact raster convention (image row 0 = frustum top),
 * so the warp's sampling flip is genuinely validated.
 */
#include "wall_shaderio.h"
#include "wall_common.h.slang"

// clang-format off
[[vk::binding(WALL_BINDING_INFO_UBO, 0)]]    ConstantBuffer<WallWarpInfo> wallInfo;
[[vk::binding(WALL_BINDING_VIEW_IMAGES, 0)]] RWTexture2D<float4>          viewImages[WALL_MAX_VIEWS];
[[vk::push_constant]]                        ConstantBuffer<WallPush>     pc;
// clang-format on

[numthreads(8, 8, 1)]
[shader("compute")]
void main(uint3 tid: SV_DispatchThreadID)
{
  uint w, h;
  viewImages[pc.viewIndex].GetDimensions(w, h);
  if(tid.x >= w || tid.y >= h)
    return;

  const WallView v = wallInfo.views[pc.viewIndex];

  const float u      = (tid.x + 0.5) / float(w);
  const float vTex   = (tid.y + 0.5) / float(h);
  const float vFrust = 1.0 - vTex;  // image row 0 = top of frustum

  const float x = v.lrbt.x + u * (v.lrbt.y - v.lrbt.x);
  const float y = v.lrbt.z + vFrust * (v.lrbt.w - v.lrbt.z);

  const float3 dir = x * v.right.xyz + y * v.up.xyz + v.fwd.xyz;

  viewImages[pc.viewIndex][int2(tid.xy)] = float4(wallTestPattern(dir), 1.0);
}
