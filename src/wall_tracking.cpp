/*
 * wall-render addition (not upstream). See wall_tracking.h.
 */
#include "wall_tracking.h"

#include <chrono>
#include <stdexcept>

#include <nvutils/logger.hpp>

namespace vk_gaussian_splatting {

namespace {
double nowSec()
{
  using namespace std::chrono;
  return duration<double>(steady_clock::now().time_since_epoch()).count();
}
}  // namespace

WallTracking::WallTracking(uint16_t port)
{
  try
  {
    rx_ = std::make_unique<FreeDReceiver>(port, [this](const FreeDPose& pose, double tSec) {
      if(!lockedCamera_)
      {
        core_.setActiveCamera(pose.cameraId);
        lockedCamera_ = true;
      }
      core_.ingest(pose, tSec);
    });
    LOGI("WallTracking: listening for FreeD on UDP port %u\n", unsigned(port));
  }
  catch(const std::exception& e)
  {
    LOGW("WallTracking: disabled (%s)\n", e.what());
  }
}

glm::dvec3 WallTracking::marsToScene(const Vec3& mm)
{
  // Mars/FreeD: X right, Y toward the wall, Z up, millimeters.
  // Viewer scene: X right, Y up, Z toward the viewer, meters.
  return {mm.x / 1000.0, mm.z / 1000.0, -mm.y / 1000.0};
}

bool WallTracking::pupilScene(glm::dvec3& out)
{
  if(!rx_)
    return false;
  Vec3 mm;
  if(!core_.pupil(nowSec(), mm))
    return false;
  out = marsToScene(mm);
  return true;
}

}  // namespace vk_gaussian_splatting
