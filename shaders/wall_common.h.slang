/*
 * wall-render addition (not upstream).
 * Shared by wall_pattern.comp.slang (fill) and wall_warp.comp.slang (compare).
 */

// Smooth direction-based test pattern: smooth in image space, so bilinear
// resampling error stays tiny and any geometric error shows up immediately.
float3 wallTestPattern(float3 dir)
{
  const float3 d = normalize(dir);
  return 0.5 + 0.5 * d;
}
