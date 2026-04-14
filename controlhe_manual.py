import os
import cv2
import numpy as np
import time
import socket
from config import *
import sys, select, tty, termios
from atuadores import set_motor_speed, set_servo_angle

# =========================
# TECLADO NON-BLOCKING
# =========================
class NonBlockingKeyboard:
    def __init__(self):
        self.is_windows = (os.name == 'nt')

    def __enter__(self):
        if not self.is_windows:
            # Configuração específica para Linux/macOS
            self.old_settings = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
        return self

    def __exit__(self, type, value, traceback):
        if not self.is_windows:
            # Restaura as configurações originais do terminal no Unix
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)

    def get_char(self):
        if self.is_windows:
            # Lógica para Windows usando msvcrt
            if msvcrt.kbhit():
                # Retorna o caractere decodificado (ex: 'w', 's', 'a', 'd')
                char = msvcrt.getch()
                try:
                    return char.decode('utf-8').lower()
                except UnicodeDecodeError:
                    return None
            return None
        else:
            # Lógica original para Linux/macOS
            if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
                return sys.stdin.read(1).lower()
            return None

# =========================
# FUNÇÃO PRINCIPAL
# =========================
def main():
    cap = cv2.VideoCapture(0)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    serve_address = (UDP_HOST, UDP_PORT)

    if not cap.isOpened():
        print("Erro ao abrir o arquivo de vídeo.")
        return

    # Inicialização de variáveis
    brightness_value = INITIAL_BRIGHTNESS
    servo_angle = NEUTRAL_ANGLE
    motor_speed = MIN_SPEED # Começa na velocidade mínima ou zero
    
    paused = False
    kbd = NonBlockingKeyboard()

    with kbd:
        try:
            while True:
                # --- LÓGICA DE TECLADO ---
                key = kbd.get_char()
                if key:
                    if key == 'q': 
                        break
                    elif key == 'p': 
                        paused = not paused
                    
                    # Controle de Velocidade (Direto no valor da velocidade)
                    elif key == 'w': 
                        motor_speed += SPEED_INCREMENT
                    elif key == 's': 
                        motor_speed -= SPEED_INCREMENT
                    
                    # Controle do Servo
                    elif key == 'a': 
                        servo_angle += SERVO_INCREMENT
                    elif key == 'd': 
                        servo_angle -= SERVO_INCREMENT
                    
                    # Brilho
                    elif key == '+': 
                        brightness_value += 10
                    elif key == '-': 
                        brightness_value -= 10
                    
                    # Reset (Z)
                    elif key == 'z': 
                        motor_speed = 0
                        servo_angle = NEUTRAL_ANGLE

                    # Aplicação de limites e envio de comandos de hardware
                    if motor_speed == 0:
                        set_motor_speed(0)
                    else:
                        motor_speed = np.clip(motor_speed, -MAX_SPEED, MAX_SPEED)
                        set_motor_speed(motor_speed)

                    servo_angle = np.clip(servo_angle, MIN_ANGLE, MAX_ANGLE)
                    set_servo_angle(servo_angle)

                # Limite de brilho
                brightness_value = np.clip(brightness_value, -255, 255)

                if paused:
                    time.sleep(0.1)
                    continue

                # --- PROCESSAMENTO DE VÍDEO ---
                ret, frame = cap.read()
                if not ret: 
                    break

                frame = cv2.resize(frame, (CAM_WIDTH, CAM_HEIGHT))

                # Ajuste de Brilho
                if brightness_value != 0:
                    # alpha=1.5 fixa o contraste, beta=brilho
                    adjusted_frame = cv2.convertScaleAbs(frame, alpha=1.0, beta=brightness_value)
                else:
                    adjusted_frame = frame

                # Feedback no terminal
                print(f"Vel: {motor_speed:.1f} | Angulo: {servo_angle:.1f} | Brilho: {brightness_value}", end='\r')

                # Streaming UDP
                try: 
                    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY] 
                    _, buffer = cv2.imencode('.jpg', adjusted_frame, encode_param) 
                    sock.sendto(buffer, serve_address) 
                except Exception: 
                    pass

        except KeyboardInterrupt:
            print("\nInterrompido pelo usuário.")
        
        finally:
            print("\n[DESLIGANDO] Parando motores e liberando hardware...")
            
            try:
                set_motor_speed(0)
                set_servo_angle(NEUTRAL_ANGLE)
                time.sleep(0.2) 
            except Exception as e:
                print(f"Erro ao parar hardware: {e}")
            
            # Liberação de Recursos
            cap.release()
            cv2.destroyAllWindows()
            sock.close()
            
            # SE estiver usando RPi.GPIO, descomente a linha abaixo:
            # GPIO.cleanup() 
            
            print("Sistema encerrado com segurança.")


if __name__ == '__main__':
    main()