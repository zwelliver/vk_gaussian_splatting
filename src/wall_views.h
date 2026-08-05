/*
 * wall-render addition (not upstream).
 *
 * Off-axis wall views (plan 4.2, integration step 2). Builds 2-3 ViewFrustums
 * from vpcore covering the LED wall's 70-degree arc from the tracked entrance
 * pupil, and can override the frame camera with one of them so the desktop
 * viewer renders exactly what a wall view will contain. The warp pass that
 * resamples these views onto the physical columns comes next.
 *
 * Wall/stage frame (vpcore WallModel): X right, Y up, wall center on -Z,
 * millimeters. The viewer scene uses the same axes in meters, so only a
 * 1/1000 scale applies here.
 */
#pragma once

#include <vector>
#include <glm/glm.hpp>

#include <core/render_views.h>

#include "shaderio.h"

namespace vk_gaussian_splatting {

class WallTracking;

class WallViews
{
public:
  WallViews();

  // UI state (wall-render panel)
  bool enabled     = false;
  int  numViews    = 3;  // 2-5 contiguous column groups
  int  displayView = 1;  // which view drives the desktop viewer

  // Rebuild the views from the tracked pupil. Returns false (views left
  // stale) if no pupil is available yet or the pupil is behind a wall face.
  bool rebuild(WallTracking& tracking);

  // Rebuild the views and replace the frame camera (view/projection/eye/fov)
  // with the selected off-axis view. Everything derived downstream (jittered
  // matrix, inverses, focal) follows the overridden values. Returns false
  // and leaves the frame untouched on rebuild failure.
  bool overrideCamera(WallTracking&        tracking,
                      shaderio::FrameInfo& frame,
                      glm::vec3&           eye,
                      glm::vec3&           center,
                      glm::vec3&           up);

  // Fill a complete FrameInfo for offscreen view i at viewRes: camera,
  // both projection matrices, inverses, focal, viewport — everything the
  // sort/cull and raster shaders read. `fi` should start as a copy of the
  // frame's prmFrame so all non-camera settings carry over.
  void fillViewFrameInfo(int i, int viewRes, shaderio::FrameInfo& fi) const;

  const std::vector<ViewFrustum>& views() const { return views_; }
  const WallModel&                wall() const { return wall_; }

private:
  // Camera basis + off-axis projection for view i (shared by the two paths).
  void applyViewCamera(int i, shaderio::FrameInfo& fi) const;

  WallModel                wall_;
  std::vector<ViewFrustum> views_;
};

}  // namespace vk_gaussian_splatting
