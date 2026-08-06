/*
 * wall-render addition (not upstream).
 *
 * Live FreeD tracking driving the render eye — plan 4.1 / phase 1.1 of
 * custom-vp-system. Owns the UDP receiver and the TrackingCore from vpcore.
 * Until the first valid packet arrives the viewer behaves exactly as
 * upstream; once packets flow, the entrance pupil overrides the camera eye.
 */
#pragma once

#include <cstdint>
#include <memory>
#include <vector>
#include <glm/vec3.hpp>

#include <tracking/freed_receiver.h>
#include <tracking/tracking_core.h>

namespace vk_gaussian_splatting {

class WallTracking
{
public:
  // Port must match the Vive Mars FreeD output configuration.
  explicit WallTracking(uint16_t port);

  // Entrance pupil in scene coordinates (meters, Y-up), smoothed and
  // predicted to "now". Returns false until the first valid packet, or if
  // the receiver could not bind its port.
  bool pupilScene(glm::dvec3& out);

  // Same pupil in the wall/stage frame (mm, Y-up, wall toward -Z) — the
  // frame vpcore's WallModel and buildViews() work in.
  bool pupilStageMm(Vec3& out);

  bool     listening() const { return rx_ != nullptr; }
  uint64_t packets() const { return rx_ ? rx_->packets() : 0; }
  uint16_t port() const { return port_; }

  // Rebind the receiver to a new UDP port (must match the Mars FreeD output).
  void restart(uint16_t port);

  // Cameras seen so far. The active camera auto-locks to the first id seen
  // until the operator picks one explicitly.
  std::vector<int> cameraIds() const;
  int              activeCamera() const;
  void             setActiveCamera(int id);  // manual pick; disables auto-lock

  // Per-camera tracker -> entrance pupil offset, camera-local {right, forward, up} mm.
  Vec3 nodalOffsetMm(int id) const;
  void setNodalOffsetMm(int id, const Vec3& offsetMm);

private:
  // Seed mapping Mars frame (mm, Z-up) -> viewer scene frame (m, Y-up).
  // The authoritative wall/stage registration comes from the touch-point
  // fit (plan 3.2); this is only for desktop bring-up.
  static glm::dvec3 marsToScene(const Vec3& mm);

  void start(uint16_t port);

  TrackingCore                   core_;
  std::unique_ptr<FreeDReceiver> rx_;
  uint16_t                       port_         = 0;
  bool                           lockedCamera_ = false;  // active camera chosen
  bool                           manualCamera_ = false;  // operator picked; no auto-lock
};

}  // namespace vk_gaussian_splatting
