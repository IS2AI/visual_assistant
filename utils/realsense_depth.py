import pyrealsense2 as rs
import numpy as np
import cv2

class DepthCamera:
    def __init__(self):
        
        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        config = rs.config()


        # align_to = rs.stream.color
        # self.alignedFs = rs.align(align_to)

        # Get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))
        depth_sensor = pipeline_profile.get_device().first_depth_sensor()
        visulpreset = depth_sensor.get_option_value_description(rs.option.visual_preset,1)
        print(visulpreset)
       
        #config.device_orientation(rs.90_DEGREES_CLOCKWISE)
        config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

        depth_sensor.set_option(rs.option.visual_preset, 4)
        
        print("Starting up the Intel Realsense D455...") 
        # Start streaming
        self.pipeline.start(config)


    def get_frame(self):
        frames = self.pipeline.wait_for_frames()
        
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        # frames = self.pipeline.wait_for_frames()
        # aligned_frames = self.alignedFs.process(frames)

        # depth_frame = aligned_frames.get_depth_frame()
        # color_frame = aligned_frames.get_color_frame()

        spatial = rs.spatial_filter()
        spatial.set_option(rs.option.filter_magnitude, 2)
        spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
        spatial.set_option(rs.option.filter_smooth_delta, 31)
        spatial.set_option(rs.option.holes_fill, 3)
        depth_frame = spatial.process(depth_frame)

        hole_filling = rs.hole_filling_filter(2)
        depth_frame = hole_filling.process(depth_frame)
        
        depth_color_frame = rs.colorizer().colorize(depth_frame)
        depth_color_image = np.asanyarray(depth_color_frame.get_data())
        depth_color_image_rotated = cv2.rotate(depth_color_image, cv2.ROTATE_180) 

        depth_image = np.asanyarray(depth_frame.get_data())
        depth_image_rotated = cv2.rotate(depth_image, cv2.ROTATE_180) 

        color_image = np.asanyarray(color_frame.get_data())
        
        if not depth_frame or not color_frame:
            return False, None, None
        return True, depth_image_rotated, color_image, depth_color_image_rotated
        
    def release(self):
        self.pipeline.stop()

        