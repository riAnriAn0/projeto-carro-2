
# 1. Cole aqui a classe InferenceEngineOBB que criamos anteriormente
# # (Certifique-se de que a classe está no mesmo arquivo ou importada)
import numpy as np
import tensorflow as tf
import cv2
import time

CONF_THRESHOLD = 0.8

class InferenceEngineOBB:
    def __init__(self, model_path):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        self.input_height = self.input_details[0]['shape'][1]
        self.input_width = self.input_details[0]['shape'][2]
        
        # Parâmetros de quantização da saída (Extraídos automaticamente do modelo)
        out_params = self.output_details[0]['quantization_parameters']
        self.output_scale = out_params['scales'][0]
        self.output_zero_point = out_params['zero_points'][0]

    def detect(self, rgb_frame, conf_threshold=CONF_THRESHOLD):
        input_image = cv2.resize(rgb_frame, (self.input_width, self.input_height))
        input_data = np.expand_dims(input_image, axis=0)

        if self.input_details[0]['dtype'] == np.int8:
            input_data = (input_data.astype(np.int16) - 128).astype(np.int8)

        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        
        raw_output = self.interpreter.get_tensor(self.output_details[0]['index'])
        raw_output = np.squeeze(raw_output).T # Agora o shape é [8400, 6]

        predictions = (raw_output.astype(np.float32) - self.output_zero_point) * self.output_scale

        scores = predictions[:, 5]
        mask = scores > conf_threshold
        detections = predictions[mask]

        return detections

def draw_obb(image, detections):
    h_orig, w_orig = image.shape[:2]
    
    for det in detections:
        x, y, w, h, angle, score = det
        x *= (w_orig / 160)
        y *= (h_orig / 160)
        w *= (w_orig / 160)
        h *= (h_orig / 160)

        rect = ((x, y), (w, h), angle * (180 / np.pi)) 
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        
        cv2.drawContours(image, [box], 0, (0, 255, 0), 2)
        cv2.putText(image, f'{score:.2f}', (int(x), int(y)), 
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

def test_video_inference(video_source, model_path):
  
    engine = InferenceEngineOBB(model_path)
    cap = cv2.VideoCapture(video_source)
    
    if not cap.isOpened():
        print(f"Erro: Não foi possível abrir o vídeo {video_source}")
        return

    print("Pressione 'q' para sair.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start_time = time.time()

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        detections = engine.detect(rgb_frame, conf_threshold=CONF_THRESHOLD)
        
        h_orig, w_orig = frame.shape[:2]
        
        for det in detections:
            norm_x = det[0]
            norm_y = det[1]
            norm_w = det[2]
            norm_h = det[3]
            angle_rad = det[4]
            score = det[5]

            x = norm_x * w_orig
            y = norm_y * h_orig
            w = norm_w * w_orig
            h = norm_h * h_orig

            angle_deg = angle_rad * (180 / np.pi)

            rect = ((x, y), (w, h), angle_deg)
            box = cv2.boxPoints(rect)
            box = box.astype(int) 

        # Desenhar
        cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)
        cv2.putText(frame, f"{score:.2f}", (int(x), int(y)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # Calcular e mostrar FPS
        # fps = 1.0 / (time.time() - start_time)
        # cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40), 
        #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Exibir o frame processado
        cv2.imshow("YOLO OBB TFLite - Video Test", frame)

        # Sai se pressionar 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# --- PARA EXECUTAR ---
if __name__ == "__main__":
    video_path = "video/teste.mp4" 
    model_path = "models_obb/best_full_integer_quant-V2.tflite"
    
    test_video_inference(video_path, model_path)