#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

ROS_NODE_NAME = "circle_bounding_node"

# Initialize CvBridge globally to avoid recreating it multiple times
bridge = CvBridge()

def image_process(img_msg):
    global bridge
    try:
        # Convert ROS Image message to OpenCV format
        cv2_img = bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
    except Exception as e:
        rospy.logerr(f"Error converting image: {e}")
        return

    # Log image dimensions
    rospy.loginfo(f"Image width: {cv2_img.shape[1]} height: {cv2_img.shape[0]}")

    # Process the image to find contours and bound in a circle
    processed_img = find_and_circle_object(cv2_img)

    # Display the processed image
    cv2.imshow("Frame", processed_img)
    cv2.waitKey(1)  # Keep the OpenCV window open

def cleanup():
    rospy.loginfo("Shutting down...")
    cv2.destroyAllWindows()

def find_and_circle_object(img):
    # Convert the image to HSV color space
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define HSV range for filtering (adjust as needed)
    lower_bound = np.array([59, 100, 100])  # Example: Green color lower bound
    upper_bound = np.array([70, 255, 255])  # Example: Green color upper bound

    # Create a binary mask for the color
    mask = cv2.inRange(hsv_img, lower_bound, upper_bound)

    # Apply morphological operations to clean up the mask
    mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Find the largest contour
        largest_contour = max(contours, key=cv2.contourArea)

        # Compute the minimum enclosing circle
        (x, y), radius = cv2.minEnclosingCircle(largest_contour)

        # Draw the circle on the image
        center = (int(x), int(y))
        radius = int(radius)
        cv2.circle(img, center, radius, (0, 255, 0), 3)  # Green circle

        # Optionally draw the center point
        cv2.circle(img, center, 5, (0, 0, 255), -1)  # Red center point

        rospy.loginfo(f"Object detected at center: {center}, radius: {radius}")

    return img

if __name__ == "__main__": 
    rospy.init_node(ROS_NODE_NAME, log_level=rospy.INFO)
    rospy.on_shutdown(cleanup)

    # Subscribe to the camera topic
    rospy.Subscriber("/usb_cam/image_raw", Image, image_process)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        pass
