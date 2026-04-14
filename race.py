import cv2
import numpy as np
import time
import argparse
import socket
from config import *
import threading
from atuadores import set_motor_speed, set_servo_angle
try:
    import tensorflow.lite as tf
except ImportError:
    import tflite_runtime.interpreter as tf

class InferenceEngine:
    def __init__(self, model_path):
        self.interpreter = tf.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        self.input_height = self.input_details[0]['shape'][1]
        self.input_width = self.input_details[0]['shape'][2]
        
        # Parâmetros de quantização da saída
        out_params = self.output_details[0]['quantization_parameters']
        self.output_scale = out_params['scales'][0]
        self.output_zero_point = out_params['zero_points'][0]

    def detect(self, rgb_frame):
        input_image = cv2.resize(rgb_frame, (self.input_width, self.input_height))
        input_data = np.expand_dims(input_image, axis=0)

        if self.input_details[0]['dtype'] == np.int8:
            input_data = (input_data.astype(np.float32) - 128).astype(np.int8)

        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        return np.squeeze(self.interpreter.get_tensor(self.output_details[0]['index'])).T

class VideoServer:
    def __init__(self, ip, port):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.dest = (ip, port)
        self.frame = None
        self.running = True
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._transmit)
        self.thread.daemon = True
        self.thread.start()

    def update_frame(self, frame):
        with self.lock:
            self.frame = frame.copy()

    def _transmit(self):
        print(f"Transmitindo para {self.dest}...")
        while self.running:
            if self.frame is not None:
                with self.lock:
                    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
                    _, buffer = cv2.imencode('.jpg', self.frame, encode_param)
                
                data = buffer.tobytes()
                if len(data) < 65507:
                    try:
                        self.sock.sendto(data, self.dest)
                    except:
                        pass
            time.sleep(0.04) 
    def stop(self):
        self.running = False
        self.sock.close()

class VideoStream:
    def __init__(self, src=0, width=160, height=160):
        self.stream = cv2.VideoCapture(src)
        if not self.stream.isOpened():
            raise IOError(f"Não foi possível abrir a câmera USB em: {src}")
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False
    def start(self):
        threading.Thread(target=self.update, args=(), daemon=True).start()
        return self
    def update(self):
        while not self.stopped:
            self.grabbed, self.frame = self.stream.read()
    def read(self):
        return self.frame
    def stop(self):
        self.stopped = True
        self.stream.release()

# --- ESTADOS ---
erro_anterior = 0
integral = 0
last_time = time.time()

def pid_servo(x_box):
    global erro_anterior, integral, last_time

    now = time.time()
    dt = now - last_time
    last_time = now

    if dt <= 0:
        return NEUTRAL_ANGLE

    erro = (x_box - CAM_WIDTH / 2) / (CAM_WIDTH / 2)

    # --- PID ---
    integral += erro * dt
    derivada = (erro - erro_anterior) / dt

    controle = KP * erro + KI * integral + KD * derivada

    erro_anterior = erro

    controle = max(-MAX_ANGLE, min(MAX_ANGLE, controle))

    angulo = NEUTRAL_ANGLE + controle

    angulo = max(MIN_ANGLE, min(MAX_ANGLE, angulo))

    return angulo

def main(args):
    engine = InferenceEngine(MODEL_PATH)
    cap = VideoStream(VIDEO_PATH, CAM_WIDTH, CAM_HEIGHT).start()
    #transmissão UDP
    server = VideoServer(UDP_HOST, UDP_PORT)

    #exibir stream
    stream = args.stream

    motor_increment_speed = args.speed
    motor_speed = ((MAX_SPEED - MIN_SPEED) * motor_increment_speed) + MIN_SPEED

    brightness_value = INITIAL_BRIGHTNESS
    servo_angle = NEUTRAL_ANGLE
    servo_angle_antigo = servo_angle

    # Threshold quantizado
    quantized_threshold = int((CONFIDENCE_THRESHOLD / engine.output_scale) + engine.output_zero_point)

    fps, frame_count = 0, 0
    start_time = time.time()

    paused = False

    try:
        while True:
            loop_start_time = time.time()
            if paused:
                time.sleep(0.1)
                continue

            frame = cap.read()
            if frame is None: break

            # Redimensiona para o padrão do projeto
            frame = cv2.resize(frame, (CAM_WIDTH, CAM_HEIGHT))

            motor_speed = np.clip(motor_speed, MIN_SPEED, MAX_SPEED)
            set_motor_speed(motor_speed)
            
            brightness_value = np.clip(brightness_value, -255, 255)


            # Ajuste de Brilho
            if brightness_value != 0:
                adjusted_frame = cv2.convertScaleAbs(frame, alpha=1.5, beta=brightness_value)
            else:
                adjusted_frame = frame

            # --- INFERÊNCIA ---
            best_detection = None
            if frame_count % DETECTION_INTERVAL == 0:
                rgb_frame = cv2.cvtColor(adjusted_frame, cv2.COLOR_BGR2RGB)
                detections = engine.detect(rgb_frame)
                
                max_confidence = quantized_threshold
                for det in detections:
                    if det[4] > max_confidence:
                        max_confidence, best_detection = det[4], det

            # --- DESENHO E CÁLCULO DE CONTROLE ---
            output_frame = adjusted_frame.copy()
            distance = 0
            h, w = output_frame.shape[:2]
            img_center_x, img_center_y = w / 2, h / 2

            if best_detection is not None:
                # Decodificação das coordenadas (cx, cy, w, h)
                cx = (best_detection[0] - engine.output_zero_point) * engine.output_scale
                cy = (best_detection[1] - engine.output_zero_point) * engine.output_scale
                bw = (best_detection[2] - engine.output_zero_point) * engine.output_scale
                bh = (best_detection[3] - engine.output_zero_point) * engine.output_scale

                box_x, box_y = cx * w, cy * h

                servo_angle = pid_servo(box_x)
                set_servo_angle(servo_angle)

                # distance = np.sqrt((box_x - img_center_x)**2 + (box_y - img_center_y)**2)
                
                # # Simulação do ângulo do servo
                # servo_angle = (box_x - img_center_x) * FATOR_GIRO
                # set_servo_angle(NEUTRAL_ANGLE + (-1 * servo_angle))
                                
                print(f"Detecção: Conf={max_confidence:.2f}, Dist={distance:.1f}, Servo Angle={servo_angle:.1f}, Vel={motor_speed:.1f}")

                # Desenhar Bounding Box
                x1, y1 = int(box_x - (bw * w / 2)), int(box_y - (bh * h / 2))
                x2, y2 = int(box_x + (bw * w / 2)), int(box_y + (bh * h / 2))
                cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.line(output_frame, (int(img_center_x), int(img_center_y)), (int(box_x), int(box_y)), (255, 0, 255), 2)
            else:
                set_servo_angle(NEUTRAL_ANGLE)
                set_motor_speed(MIN_SPEED)
                pass

            # --- HUD INFORMATIVO ---
            elapsed = time.time() - start_time
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count, start_time = 0, time.time()

            cv2.putText(output_frame, f"FPS: {fps:.1f} | Vel: {motor_increment_speed*100:.1f}%", (10, 30), 1, 1.3, (0, 0, 255), 2)
            cv2.putText(output_frame, f"Angulo Servo: {servo_angle:.1f}", (10, 60), 1, 1.3, (0,0,255), 2)
            cv2.putText(output_frame, f"Brilho: {brightness_value}", (10, 90), 1, 1.3, (0, 0, 255), 2)

            if stream:
                cv2.imshow("Output", output_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

            try: 
                server.update_frame(output_frame)
            except Exception: 
                pass
            frame_count += 1

    except KeyboardInterrupt:
        pass
    
    finally:
        cap.stop()
        server.stop()
        set_motor_speed(0)
        set_servo_angle(NEUTRAL_ANGLE)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--speed', type=float, default=0.0)
    parser.add_argument('--stream', type=bool, default=False)
    args = parser.parse_args()
    main(args)