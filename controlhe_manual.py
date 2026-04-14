import os
import typing
import cv2
import numpy as np
import time
import socket
import config
import sys, select, tty, termios
import atuadores

# =========================
# TECLADO NON-BLOCKING
# =========================
class NonBlockingKeyboard:
    def __init__(self) -> None:
        self.is_windows: bool = (os.name == 'nt')

    def __enter__(self) -> Self:
        if not self.is_windows:
            # Configuração específica para Linux/macOS
            self.old_settings = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
        return self

    def __exit__(self, type, value, traceback) -> None:
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
def main() -> None:
    cap = cv2.VideoCapture(0)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    serve_address: tuple[typing.Literal[' 10.42.0.104'], typing.Literal[9999]] = (config.UDP_HOST, config.UDP_PORT)

    if not cap.isOpened():
        print("Erro ao abrir o arquivo de vídeo.")
        return

    # Inicialização de variáveis
    brightness_value: int = config.INITIAL_BRIGHTNESS
    servo_angle: int = config.NEUTRAL_ANGLE
    motor_speed: int = config.MIN_SPEED # Começa na velocidade mínima ou zero
    
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
                    
                    # Controle de Velocidade (Direto no valor da velocidade)
                    elif key == 'w': 
                        motor_speed += config.SPEED_INCREMENT
                    elif key == 's': 
                        motor_speed -= config.SPEED_INCREMENT
                    
                    # Controle do Servo
                    elif key == 'a': 
                        servo_angle += config.SERVO_INCREMENT
                    elif key == 'd': 
                        servo_angle -= config.SERVO_INCREMENT
                    
                    # Brilho
                    elif key == '+': 
                        brightness_value += 10
                    elif key == '-': 
                        brightness_value -= 10
                    
                    # Reset (Z)
                    elif key == 'z': 
                        motor_speed = 0
                        servo_angle: int = config.NEUTRAL_ANGLE

                    # Aplicação de limites e envio de comandos de hardware
                    if motor_speed == 0:
                        atuadores.set_motor_speed(0)
                    else:
                        motor_speed = np.clip(motor_speed, -config.MAX_SPEED, config.MAX_SPEED)
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

                # Streaming UDP
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
            
            # Liberação de Recursos
            cap.release()
            cv2.destroyAllWindows()
            sock.close()
            
            # SE estiver usando RPi.GPIO, descomente a linha abaixo:
            # GPIO.cleanup() 
            
            print("Sistema encerrado com segurança.")


if __name__ == '__main__':
    main()