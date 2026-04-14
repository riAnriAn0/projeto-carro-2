import socket
import numpy as np
import cv2
import time
import argparse

UDP_IP = "0.0.0.0"
UDP_PORT = 9999
BUFFER_SIZE = 65535

SAVE_DIR = "videos_treinamentos"

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))
sock.settimeout(2)

print("Aguardando video...")

video_writer = None
fps = 25

def create_video_writer(frame):
    global video_writer
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"{SAVE_DIR}/video_{timestamp}.avi"
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(filename, fourcc, fps, (frame.shape[1], frame.shape[0]))

def main(args):
    gravar = args.stream
    while True:
        try:    
            data, addr = sock.recvfrom(BUFFER_SIZE)

            frame = np.frombuffer(data, dtype=np.uint8)
            img = cv2.imdecode(frame, cv2.IMREAD_COLOR)

            if img is None:
                continue

            if video_writer is None and gravar:
                create_video_writer(img)

            video_writer.write(img)
            cv2.imshow("Stream UDP", img)

        except socket.timeout:
            print("Sem dados...")
            continue

        except Exception as e:
            print("Erro:", e)
            break

        if cv2.waitKey(1) & 0xFF == 27:
            break

    sock.close()
    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--stream', type=bool, default=True, help='Salvar a transmissão em vídeo (True para sim, False para não). Padrão: True')
    args = parser.parse_args()
    main(args)