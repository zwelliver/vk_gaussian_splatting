/*
 * wall-render addition (not upstream). See wall_views.h.
 */
#include "wall_views.h"
#include "wall_tracking.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include <glm/gtc/matrix_transform.hpp>

namespace vk_gaussian_splatting {

namespace {
inline glm::vec3 toGlm(const Vec3& v)
{
  return {float(v.x), float(v.y), float(v.z)};
}
}  // namespace

WallViews::WallViews() = default;

bool WallViews::overrideCamera(WallTracking&        tracking,
                               shaderio::FrameInfo& frame,
                               glm::vec3&           eye,
                               glm::vec3&           center,
                               glm::vec3&           up)
{
  Vec3 pupilMm;
  if(!tracking.pupilStageMm(pupilMm))
    return false;

  numViews    = std::clamp(numViews, 2, 5);
  displayView = std::clamp(displayView, 0, numViews - 1);

  try
  {
    // viewRes is only metadata until the offscreen view targets exist
    views_ = buildViews(wall_, pupilMm, numViews, 2048);
  }
  catch(const std::runtime_error&)
  {
    return false;  // pupil on/behind a wall face: keep the interactive camera
  }

  const ViewFrustum& v = views_[displayView];

  // Stage mm -> scene meters; directions are unit and carry over.
  const glm::vec3 eyeM = toGlm(v.eye) * 0.001f;
  const glm::vec3 fwd  = toGlm(v.fwd);
  const glm::vec3 vup  = toGlm(v.up);

  eye    = eyeM;
  center = eyeM + fwd;
  up     = vup;

  frame.cameraPosition = eyeM;
  frame.viewMatrix     = glm::lookAt(eyeM, eyeM + fwd, vup);
  frame.viewInverse    = glm::inverse(frame.viewMatrix);

  // Off-axis projection. ViewFrustum bounds are tangents at unit distance;
  // frustumRH_ZO takes them at the near plane. Vulkan's downward NDC y flips
  // the whole y row: the linear term AND the off-center offset term.
  const float n = frame.nearFar.x, f = frame.nearFar.y;
  glm::mat4   proj = glm::frustumRH_ZO(float(v.l) * n, float(v.r) * n, float(v.b) * n, float(v.t) * n, n, f);
  proj[1][1] *= -1;
  proj[2][1] *= -1;
  frame.projectionMatrix = proj;

  // Nominal vertical fov of the off-axis view (for the fisheye/UI paths).
  frame.fovRad = 2.0f * std::atan(0.5f * float(v.t - v.b));

  return true;
}

}  // namespace vk_gaussian_splatting
