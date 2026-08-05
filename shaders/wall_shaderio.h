/*
 * wall-render addition (not upstream).
 * Shared host/device structs for the wall views + warp pass.
 */
#ifndef _WALL_SHADERIO_H_
#define _WALL_SHADERIO_H_

// bindings for the wall warp/pattern compute descriptor set
#define WALL_BINDING_INFO_UBO 0
#define WALL_BINDING_VIEW_TEXTURES 1  // sampled view colors, WALL_MAX_VIEWS
#define WALL_BINDING_CANVAS_IMAGE 2   // canvas storage image
#define WALL_BINDING_ERR_BUFFER 3     // self-test result buffer
#define WALL_BINDING_VIEW_IMAGES 4    // view colors as storage (pattern fill)

#define WALL_MAX_VIEWS 5
#define WALL_COLUMNS 15

#ifdef __cplusplus
#include "nvshaders/slang_types.h"
namespace shaderio {
#endif

// One off-axis wall view. float4-only members so host/device layouts agree
// under any packing rules.
struct WallView
{
  float4 eye;    // entrance pupil, scene meters (w unused)
  float4 right;  // unit camera basis
  float4 up;
  float4 fwd;
  float4 lrbt;  // frustum bounds: tangents at unit distance along fwd
};

// Everything the warp needs: the views and the analytic per-column wall
// geometry (mirror of vpcore WallModel::pixelToPoint, meters).
struct WallWarpInfo
{
  WallView views[WALL_MAX_VIEWS];
  float4   colOrigin[WALL_COLUMNS];  // bottom-left corner of column face
  float4   colRight[WALL_COLUMNS];   // step along the face per canvas pixel
  int4     colView[WALL_COLUMNS];    // x = view index rendering this column
  float4   upStep;                   // step upward per canvas pixel (0, pitch, 0)
  int      numViews;
  int      canvasW;
  int      canvasH;
  int      selfTest;  // 1 = views hold the analytic pattern; compare and record error
};

struct WallErr
{
  uint maxErrBits;  // float bits of max abs error (positive floats order as uints)
  uint badCount;    // pixels with error > 0.01
  uint samples;     // pixels compared
  uint pad;
};

struct WallPush
{
  int viewIndex;
  int pad0;
  int pad1;
  int pad2;
};

#ifdef __cplusplus
}  // namespace shaderio
#endif

#endif
