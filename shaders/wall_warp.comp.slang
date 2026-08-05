/*
 * wall-render addition (not upstream).
 *
 * Warp pass: for every HELIOS canvas pixel, compute its 3D point on the
 * piecewise-planar wall (mirror of vpcore WallModel::pixelToPoint), project
 * it into the wall view assigned to its column (ViewFrustum::project), and
 * bilinearly sample that view. Geometrically exact by construction — any
 * visible column seam is a bug, never something to blend away.
 *
 * Self-test mode: the views hold the analytic direction pattern; the warped
 * result is compared against the pattern evaluated directly on the canvas
 * pixel's own ray, and the canvas shows |error| amplified 50x (should be
 * near-black; structure = geometry bug). Max error and bad-pixel count land
 * in the error buffer. This is the GPU twin of phase0_sim's exactness gate.
 */
#include "wall_shaderio.h"
#include "wall_common.h.slang"

// clang-format off
[[vk::binding(WALL_BINDING_INFO_UBO, 0)]]      ConstantBuffer<WallWarpInfo>  wallInfo;
[[vk::binding(WALL_BINDING_VIEW_TEXTURES, 0)]] Sampler2D                     viewTex[WALL_MAX_VIEWS];
[[vk::binding(WALL_BINDING_CANVAS_IMAGE, 0)]]  RWTexture2D<float4>           canvas;
[[vk::binding(WALL_BINDING_ERR_BUFFER, 0)]]    RWStructuredBuffer<WallErr>   errBuf;
// clang-format on

[numthreads(8, 8, 1)]
[shader("compute")]
void main(uint3 tid: SV_DispatchThreadID)
{
  if(tid.x >= uint(wallInfo.canvasW) || tid.y >= uint(wallInfo.canvasH))
    return;

  // Canvas pixel center -> 3D point on the wall (affine per flat column).
  const int    colPx  = wallInfo.canvasW / WALL_COLUMNS;
  const int    col    = int(tid.x) / colPx;
  const float  px     = (tid.x + 0.5) - col * colPx;
  const float  rowsUp = float(wallInfo.canvasH) - (tid.y + 0.5);
  const float3 p = wallInfo.colOrigin[col].xyz + px * wallInfo.colRight[col].xyz + rowsUp * wallInfo.upStep.xyz;

  // Project into this column's view.
  const int      vi = wallInfo.colView[col].x;
  const WallView v  = wallInfo.views[vi];
  const float3   d  = p - v.eye.xyz;
  const float    z  = dot(d, v.fwd.xyz);

  float4 color = float4(1, 0, 1, 1);  // magenta = projection failure (always a bug)
  if(z > 1e-9)
  {
    const float u  = (dot(d, v.right.xyz) / z - v.lrbt.x) / (v.lrbt.y - v.lrbt.x);
    const float vv = (dot(d, v.up.xyz) / z - v.lrbt.z) / (v.lrbt.w - v.lrbt.z);
    color          = viewTex[vi].SampleLevel(float2(u, 1.0 - vv), 0);

    if(wallInfo.selfTest != 0)
    {
      const float3 expected = wallTestPattern(d);
      const float3 diff     = abs(color.xyz - expected);
      const float  err      = max(diff.x, max(diff.y, diff.z));
      InterlockedMax(errBuf[0].maxErrBits, asuint(err));
      if(err > 0.01)
        InterlockedAdd(errBuf[0].badCount, 1);
      InterlockedAdd(errBuf[0].samples, 1);
      color = float4(diff * 50.0, 1.0);  // near-black canvas = pass
    }
  }

  canvas[int2(tid.xy)] = color;
}
