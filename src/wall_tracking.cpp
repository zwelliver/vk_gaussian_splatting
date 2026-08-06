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
  start(port);
}

void WallTracking::start(uint16_t port)
{
  port_ = port;
  try
  {
    rx_ = std::make_unique<FreeDReceiver>(port, [this](const FreeDPose& pose, double tSec) {
      if(!lockedCamera_ && !manualCamera_)
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

void WallTracking::restart(uint16_t port)
{
  rx_.reset();
  start(port);
}

std::vector<int> WallTracking::cameraIds() const
{
  return core_.cameraIds();
}

int WallTracking::activeCamera() const
{
  return core_.activeCamera();
}

void WallTracking::setActiveCamera(int id)
{
  manualCamera_ = true;
  lockedCamera_ = true;
  core_.setActiveCamera(id);
}

Vec3 WallTracking::nodalOffsetMm(int id) const
{
  return core_.nodalOffset(id);
}

void WallTracking::setNodalOffsetMm(int id, const Vec3& offsetMm)
{
  core_.setNodalOffset(id, offsetMm);
}

glm::dvec3 WallTracking::marsToScene(const Vec3& mm)
{
  // Mars/FreeD: X right, Y toward the wall, Z up, millimeters.
  // Viewer scene: X right, Y up, Z toward the viewer, meters.
  return {mm.x / 1000.0, mm.z / 1000.0, -mm.y / 1000.0};
}

bool WallTracking::pupilScene(glm::dvec3& out)
{
  Vec3 stageMm;
  if(!pupilStageMm(stageMm))
    return false;
  out = glm::dvec3(stageMm.x, stageMm.y, stageMm.z) / 1000.0;
  return true;
}

bool WallTracking::pupilStageMm(Vec3& out)
{
  if(!rx_)
    return false;
  Vec3 mm;
  if(!core_.pupil(nowSec(), mm))
    return false;
  // Same axis mapping as marsToScene, kept in millimeters.
  out = {mm.x, mm.z, -mm.y};
  return true;
}

}  // namespace vk_gaussian_splatting
