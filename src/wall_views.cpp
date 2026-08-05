/*
 * wall-render addition (not upstream). See wall_views.h.
 */
#include "wall_views.h"
#include "wall_tracking.h"
#include "wall_shaderio.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

namespace vk_gaussian_splatting {

namespace {
inline glm::vec3 toGlm(const Vec3& v)
{
  return {float(v.x), float(v.y), float(v.z)};
}
}  // namespace

WallViews::WallViews() = default;

bool WallViews::rebuild(WallTracking& tracking)
{
  Vec3 pupilMm;
  if(!tracking.pupilStageMm(pupilMm))
    return false;

  numViews    = std::clamp(numViews, 2, WALL_MAX_VIEWS);
  displayView = std::clamp(displayView, 0, numViews - 1);

  try
  {
    // viewRes here is only metadata; the render resolution is set per pass
    views_ = buildViews(wall_, pupilMm, numViews, 2048);
  }
  catch(const std::runtime_error&)
  {
    return false;  // pupil on/behind a wall face: keep the previous views
  }
  return true;
}

void WallViews::applyViewCamera(int i, shaderio::FrameInfo& fi) const
{
  const ViewFrustum& v = views_[i];

  // Stage mm -> scene meters; directions are unit and carry over.
  const glm::vec3 eyeM = toGlm(v.eye) * 0.001f;
  const glm::vec3 fwd  = toGlm(v.fwd);
  const glm::vec3 vup  = toGlm(v.up);

  fi.cameraPosition = eyeM;
  fi.viewMatrix     = glm::lookAt(eyeM, eyeM + fwd, vup);
  fi.viewInverse    = glm::inverse(fi.viewMatrix);

  // Off-axis projection. ViewFrustum bounds are tangents at unit distance;
  // frustumRH_ZO takes them at the near plane. Vulkan's downward NDC y flips
  // the whole y row: the linear term AND the off-center offset term.
  const float n = fi.nearFar.x, f = fi.nearFar.y;
  glm::mat4   proj = glm::frustumRH_ZO(float(v.l) * n, float(v.r) * n, float(v.b) * n, float(v.t) * n, n, f);
  proj[1][1] *= -1;
  proj[2][1] *= -1;
  fi.projectionMatrix = proj;

  // Nominal vertical fov of the off-axis view (for the fisheye/UI paths).
  fi.fovRad = 2.0f * std::atan(0.5f * float(v.t - v.b));
}

bool WallViews::overrideCamera(WallTracking&        tracking,
                               shaderio::FrameInfo& frame,
                               glm::vec3&           eye,
                               glm::vec3&           center,
                               glm::vec3&           up)
{
  if(!rebuild(tracking))
    return false;

  applyViewCamera(displayView, frame);

  const ViewFrustum& v = views_[displayView];
  eye    = glm::vec3(frame.cameraPosition);
  center = eye + toGlm(v.fwd);
  up     = toGlm(v.up);
  return true;
}

void WallViews::fillViewFrameInfo(int i, int viewRes, shaderio::FrameInfo& fi) const
{
  applyViewCamera(i, fi);

  // The offscreen path bypasses updateAndUploadFrameInfoUBO's derivation
  // code, so derive the dependent fields here, mirroring it exactly.
  fi.projectionMatrixJittered = fi.projectionMatrix;
  fi.projInverse              = glm::inverse(fi.projectionMatrix);

  fi.viewport      = glm::vec2(float(viewRes), float(viewRes));
  fi.basisViewport = glm::vec2(1.0f / viewRes, 1.0f / viewRes);
  fi.focal = glm::vec2(fi.projectionMatrix[0][0] * 0.5f * viewRes, fi.projectionMatrix[1][1] * 0.5f * viewRes);
  fi.inverseFocalAdjustment = 1.0f;

  const glm::quat q = glm::quat_cast(fi.viewMatrix);
  fi.viewQuat       = glm::vec4(q.x, q.y, q.z, q.w);
  fi.viewTrans      = glm::vec3(fi.viewMatrix[3]);
}

}  // namespace vk_gaussian_splatting
