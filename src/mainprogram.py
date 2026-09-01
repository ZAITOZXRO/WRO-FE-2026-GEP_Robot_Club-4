"""
WRO Open Challenge - Autonomous Lane Following Robot
==================================================
Hardware: Raspberry Pi, standard servo (steering), DC motor (drive), camera.
Description: 
    This script implements an autonomous lane-following control loop using 
    OpenCV for bottom-ROI image thresholding, centroid-based error calculation, 
    and a PID controller with dynamic steering boost to safely navigate the track.
"""

import pigpio
import time
import cv2
import numpy as np

# ==========================================
# 1. HARDWARE & PIN DEFINITIONS
# ==========================================
SERVO_PIN = 18
MOTOR_PWMA = 12
MOTOR_AIN1 = 23
MOTOR_AIN2 = 24
MOTOR_STBY = 25

SERVO_CENTER_US = 1500   
STEERING_TRIM_DEG = -13  
MAX_STEER_ANGLE = 60

pi = pigpio.pi()
if not pi.connected:
    exit("Cannot connect to pigpio daemon! Run 'sudo pigpiod' first.")

pi.set_mode(SERVO_PIN, pigpio.OUTPUT)
pi.set_mode(MOTOR_PWMA, pigpio.OUTPUT)
pi.set_mode(MOTOR_AIN1, pigpio.OUTPUT)
pi.set_mode(MOTOR_AIN2, pigpio.OUTPUT)
pi.set_mode(MOTOR_STBY, pigpio.OUTPUT)
pi.write(MOTOR_STBY, 1)

# ==========================================
# 2. PID CONTROLLER VARIABLES
# ==========================================
Kp = 0.25   
Ki = 0.00   
Kd = 0.15   

last_error = 0
integral = 0

def compute_pid(error):
    """Calculate PID steering adjustment based on tracking error."""
    global last_error, integral
    integral += error
    derivative = error - last_error
    steer_angle = (Kp * error) + (Ki * integral) + (Kd * derivative)
    last_error = error
    return steer_angle

def set_steering_angle(angle_deg):
    """Calibrate, clamp, and apply the steering angle to the servo."""
    calibrated_angle = angle_deg + STEERING_TRIM_DEG
    clamped_angle = max(-MAX_STEER_ANGLE, min(MAX_STEER_ANGLE, calibrated_angle))
    pulse_width = SERVO_CENTER_US + (clamped_angle * 11.4)
    pi.set_servo_pulsewidth(SERVO_PIN, pulse_width)

def set_motor_speed(speed_pct):
    """Control the DC drive motor speed and direction."""
    if speed_pct > 0:
        pi.write(MOTOR_AIN1, 1)
        pi.write(MOTOR_AIN2, 0)
    elif speed_pct < 0:
        pi.write(MOTOR_AIN1, 0)
        pi.write(MOTOR_AIN2, 1)
    else:
        pi.write(MOTOR_AIN1, 0)
        pi.write(MOTOR_AIN2, 0)
    duty_cycle = int(abs(speed_pct) * 2.55)
    pi.set_PWM_dutycycle(MOTOR_PWMA, duty_cycle)

# ==========================================
# 3. MAIN OPEN CHALLENGE LOOP
# ==========================================
def main():
    print("=== OPEN CHALLENGE (PID LANE FOLLOWING) ===")
    print("Starting in 3 seconds... Press Ctrl+C anytime to stop!")
    for i in range(3, 0, -1):
        print(f"{i}...")
        time.sleep(1)
        
    print("=== RUNNING ===")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    CRUISE_SPEED = 30  
    
    try:
        set_motor_speed(CRUISE_SPEED)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            height, width, _ = frame.shape
            
            # Crop lower ROI (75% - 100%) to isolate ground track and ignore wall artifacts
            roi_ymin = int(height * 0.75)
            roi = frame[roi_ymin:height, 0:width]
            
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
            
            # Calculate centroid and tracking error
            M = cv2.moments(mask)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                center_of_frame = int(width / 2)
                error = cx - center_of_frame 
            else:
                error = 0
                
            steer = compute_pid(error)
            
            # Dynamic steering boost to react quicker near frame boundaries
            if abs(error) > (width * 0.2):
                steer = steer * 1.5
            
            # Invert steering sign to match correct physical orientation
            set_steering_angle(-steer) 
            
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        print("\n[STOP] Emergency Stop via Ctrl+C")
    finally:
        set_motor_speed(0)
        set_steering_angle(0)
        cap.release()
        pi.stop()

if __name__ == "__main__":
    main()
