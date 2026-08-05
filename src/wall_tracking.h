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

  bool     listening() const { return rx_ != nullptr; }
  uint64_t packets() const { return rx_ ? rx_->packets() : 0; }

private:
  // Seed mapping Mars frame (mm, Z-up) -> viewer scene frame (m, Y-up).
  // The authoritative wall/stage registration comes from the touch-point
  // fit (plan 3.2); this is only for desktop bring-up.
  static glm::dvec3 marsToScene(const Vec3& mm);

  TrackingCore                   core_;
  std::unique_ptr<FreeDReceiver> rx_;
  bool                           lockedCamera_ = false;  // active camera = first id seen
};

}  // namespace vk_gaussian_splatting
