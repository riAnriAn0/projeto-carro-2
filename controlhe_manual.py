import os
import sys
from typing import Optional
import typing
import cv2
import numpy as np
import time
import socket
import config
import atuadores

if os.name == 'nt':
    import msvcrt
else:
    import termios
    import tty
    import select

class NonBlockingKeyboard:
    def __init__(self) -> None:
        self.is_windows: bool = (os.name == 'nt')
        self.old_settings = None

    def __enter__(self):
        if not self.is_windows:
            self.old_settings = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if not self.is_windows and self.old_settings:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)

    def get_char(self) -> Optional[str]:
        """Retorna o caractere pressionado ou None se nenhuma tecla foi tocada."""
        if self.is_windows:
            # Lógica para Windows
            if msvcrt.kbhit():
                try:
                    char = msvcrt.getch()
                    return char.decode('utf-8').lower()
                except (UnicodeDecodeError, AttributeError):
                    return None
            return None
        else:
            if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
                return sys.stdin.read(1).lower()
            return None
        
def control_speed(current_speed: int, increment: int) -> int:
    anterior_speed = current_speed
    current_speed += increment

    current_speed = np.clip(current_speed, -config.MAX_SPEED, config.MAX_SPEED)
    if current_speed == 0:
        return 0
    elif current_speed > 0 and current_speed < config.MIN_SPEED and anterior_speed < current_speed:
        return current_speed + config.MIN_SPEED
    elif current_speed > 0 and current_speed < config.MIN_SPEED and anterior_speed > current_speed:
        return 0
    elif current_speed < 0 and current_speed > -config.MIN_SPEED and anterior_speed > current_speed:
        return current_speed - config.MIN_SPEED
    elif current_speed < 0 and current_speed > -config.MIN_SPEED and anterior_speed < current_speed:
        return 0
    return current_speed

def main() -> None:
    cap = cv2.VideoCapture(0)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    serve_address: tuple[typing.Literal['10.42.0.104'], typing.Literal[9999]] = (config.UDP_HOST, config.UDP_PORT)

    if not cap.isOpened():
        print("Erro ao abrir o arquivo de vídeo.")
        return

    # Inicialização de variáveis
    brightness_value: int = config.INITIAL_BRIGHTNESS
    servo_angle: int = config.NEUTRAL_ANGLE
    motor_speed: int = 0
    incremento: int = 0
    
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
                        paused: bool = not paused
                    
                    elif key == 'w': 
                        incremento = config.SPEED_INCREMENT
                    elif key == 's': 
                        incremento = -1 * config.SPEED_INCREMENT
                    
                    elif key == 'a': 
                        servo_angle += config.SERVO_INCREMENT
                    elif key == 'd': 
                        servo_angle -= config.SERVO_INCREMENT
                        
                    #controlhe de brilho
                    elif key == '+': 
                        brightness_value += 10
                    elif key == '-': 
                        brightness_value -= 10
                    
                    elif key == 'x': 
                        motor_speed = 0
                        servo_angle: int = config.NEUTRAL_ANGLE

                    motor_speed = control_speed(motor_speed, incremento)
                    atuadores.set_motor_speed(motor_speed)

                    servo_angle = np.clip(servo_angle, config.MIN_ANGLE, config.MAX_ANGLE)
                    atuadores.set_servo_angle(servo_angle)

                # Limite de brilho
                brightness_value = np.clip(brightness_value, -255, 255)

                if paused:
                    time.sleep(0.1)
                    continue

                # --- PROCESSAMENTO DE VÍDEO ---
                ret, frame = cap.read()
                if not ret: 
                    break

                frame: cv2.Mat | np.ndarray[typing.Any, np.dtype[np.integer[typing.Any] | np.floating[typing.Any]]] = cv2.resize(frame, (config.CAM_WIDTH, config.CAM_HEIGHT))

                # Ajuste de Brilho
                if brightness_value != 0:
                    # alpha=1.5 fixa o contraste, beta=brilho
                    adjusted_frame: cv2.Mat | np.ndarray[typing.Any, np.dtype[np.integer[typing.Any] | np.floating[typing.Any]]] = cv2.convertScaleAbs(frame, alpha=1.0, beta=brightness_value)
                else:
                    adjusted_frame = frame

                # Feedback no terminal
                print(f"Vel: {motor_speed:.1f} | Angulo: {servo_angle:.1f} | Brilho: {brightness_value}", end='\r')

                # # Streaming UDP
                try: 
                    encode_param: list[int] = [int(cv2.IMWRITE_JPEG_QUALITY), config.JPEG_QUALITY] 
                    _, buffer = cv2.imencode('.jpg', adjusted_frame, encode_param) 
                    sock.sendto(buffer, serve_address) 
                except Exception: 
                    pass

        except KeyboardInterrupt:
            print("\nInterrompido pelo usuário.")
        
        finally:
            print("\n[DESLIGANDO] Parando motores e liberando hardware...")
            
            try:
                atuadores.set_motor_speed(0)
                atuadores.set_servo_angle(config.NEUTRAL_ANGLE)
                time.sleep(0.2) 
            except Exception as e:
                print(f"Erro ao parar hardware: {e}")
            
            cap.release()
            cv2.destroyAllWindows()
            sock.close()

            print("Sistema encerrado com segurança.")


if __name__ == '__main__':
    main()