#!/usr/bin/env python3

import cv2
import depthai as dai
import numpy as np
import sys
sys.path = [p for p in sys.path if "depthai-python" not in p]

# ====== YOUR CHECKERBOARD SETTINGS HERE =====
pattern_size = (9, 6)  # (cols, rows) -> number of inner corners!
square_size_mm = 23.17  # real-world square size in mm (optional, for scale)
# ============================================

# Create pipeline
pipeline = dai.Pipeline()

# Create nodes
cam_rgb = pipeline.create(dai.node.ColorCamera)
stereo = pipeline.create(dai.node.StereoDepth)
xout_rgb = pipeline.create(dai.node.XLinkOut)
xout_depth = pipeline.create(dai.node.XLinkOut)

# Set stream names
xout_rgb.setStreamName("rgb")
xout_depth.setStreamName("depth")

# Color camera properties
cam_rgb.setPreviewSize(1280, 800)
cam_rgb.setInterleaved(False)
cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

# Stereo properties
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_ACCURACY)
stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
stereo.setOutputSize(1280, 800)
stereo.setSubpixel(True)

# Mono cameras
mono_left = pipeline.create(dai.node.MonoCamera)
mono_right = pipeline.create(dai.node.MonoCamera)
mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
mono_left.setCamera("left")
mono_right.setCamera("right")

# Linking
mono_left.out.link(stereo.left)
mono_right.out.link(stereo.right)
cam_rgb.preview.link(xout_rgb.input)
stereo.depth.link(xout_depth.input)

# Start device
with dai.Device(pipeline) as device:
    print("Starting pipeline...")
    # Get queues
    q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    q_depth = device.getOutputQueue(name="depth", maxSize=4, blocking=False)

    # Get intrinsics
    calib_data = device.readCalibration()
    intrinsics = calib_data.getCameraIntrinsics(dai.CameraBoardSocket.RGB, 640, 400)
    fx, fy = intrinsics[0][0], intrinsics[1][1]
    cx, cy = intrinsics[0][2], intrinsics[1][2]

    print(f"Camera intrinsics: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")

    while True:
        in_rgb = q_rgb.get()
        in_depth = q_depth.get()

        # Get frames
        frame_rgb = in_rgb.getCvFrame()
        frame_depth = in_depth.getFrame()

        # Normalize depth for display
        depth_display = cv2.normalize(frame_depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)

        # Chessboard detection
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

        if ret:
            # Refine corner locations
            corners_subpix = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                              (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.001))

            # Draw corners
            cv2.drawChessboardCorners(frame_rgb, pattern_size, corners_subpix, ret)

            # Calculate 3D points
            points_3d = []
            for corner in corners_subpix:
                u, v = int(corner[0][0]), int(corner[0][1])
                if u < 0 or v < 0 or u >= frame_depth.shape[1] or v >= frame_depth.shape[0]:
                    continue
                Z = frame_depth[v, u]
                if Z == 0:
                    continue
                X = (u - cx) * Z / fx
                Y = (v - cy) * Z / fy
                points_3d.append([X, Y, Z])
                # Draw point on depth image
                cv2.circle(depth_colored, (u, v), 3, (0, 255, 0), -1)

            points_3d = np.array(points_3d)
            if len(points_3d) >= pattern_size[0] * pattern_size[1]:
                # Compute width and height
                width_vec = points_3d[pattern_size[0]-1] - points_3d[0]
                height_vec = points_3d[-1] - points_3d[0]
                width_mm = np.linalg.norm(width_vec)
                height_mm = np.linalg.norm(height_vec)
                print(f"Estimated width: {width_mm/10:.1f} cm")
                print(f"Estimated height: {height_mm/10:.1f} cm")

        # Show images
        cv2.imshow("RGB", frame_rgb)
        cv2.imshow("Depth", depth_colored)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()