import pigpio
import time
import numpy as np
from config import *

PIN_SERVO = 18
PIN1_MOTOR1 = 23
PIN2_MOTOR1 = 24

PIN1_MOTOR2 = 8
PIN2_MOTOR2 = 25


pi = pigpio.pi('127.0.0.1')

if not pi.connected:
    print("ERRO: Não foi possível conectar ao pigpiod. Rode 'sudo pigpiod' no terminal.")

def set_motor_speed(speed=0):
    if not pi.connected: return
    
    motor_speed = np.clip(speed, MIN_SPEED, MAX_SPEED)
    if motor_speed > 0:
        pi.set_PWM_dutycycle(PIN1_MOTOR1, motor_speed)
        pi.set_PWM_dutycycle(PIN2_MOTOR1, 0)
        pi.set_PWM_dutycycle(PIN2_MOTOR2, motor_speed)
        pi.set_PWM_dutycycle(PIN1_MOTOR2, 0)
   
    elif motor_speed < 0:
        pi.set_PWM_dutycycle(PIN2_MOTOR1, -motor_speed)
        pi.set_PWM_dutycycle(PIN1_MOTOR1, 0)
        pi.set_PWM_dutycycle(PIN1_MOTOR2, -motor_speed)
        pi.set_PWM_dutycycle(PIN2_MOTOR2, 0)
    else:
        pi.set_PWM_dutycycle(PIN1_MOTOR1, 0)
        pi.set_PWM_dutycycle(PIN2_MOTOR1, 0)
        pi.set_PWM_dutycycle(PIN1_MOTOR2, 0)
        pi.set_PWM_dutycycle(PIN2_MOTOR2, 0)

def set_servo_angle(angle=NEUTRAL_ANGLE):
    if not pi.connected: return
    
    servo_angle = np.clip(angle, MIN_ANGLE, MAX_ANGLE)
    pulse_width = int(500 + (servo_angle / 180) * 2000)
    pi.set_servo_pulsewidth(PIN_SERVO, pulse_width)