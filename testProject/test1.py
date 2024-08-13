import cv2
import tkinter as tk
from tkinter import messagebox
from tkinter.filedialog import askopenfilename
import threading
import datetime
import os
from tkinter import ttk
from PIL import Image, ImageTk
from test2 import process_frame_sequence  # 수정된 handle_frame 함수 임포트

import warnings


class CameraApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Camera App")
        self.root.geometry("400x350")
        self.root.configure(bg='#2c3e50')

        # ttk 스타일 설정
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TButton",
                        relief="flat",
                        padding=10,
                        foreground='#ffffff',
                        background='#3498db',
                        borderwidth=0,
                        font=('Helvetica', 12, 'bold'),
                        focuscolor='none')
        style.map("TButton",
                  background=[('active', '#2980b9')],
                  foreground=[('active', '#ecf0f1')])

        # 변수 초기화
        self.is_recording = True  # 카메라가 항상 켜져있음
        self.is_displaying = False  # 화면에 출력 중인지 여부
        self.capture = None
        self.lock = threading.Lock()  # 스레드 간 동기화를 위한 락

        # video 디렉터리 생성
        self.video_dir = os.path.join(os.getcwd(), 'video')
        self.abnormal_dir = os.path.join(self.video_dir, 'abnormal')
        os.makedirs(self.abnormal_dir, exist_ok=True)

        # 버튼 이미지 로드
        self.load_icons()

        # 버튼 생성
        self.start_button = ttk.Button(root, text="매장 확인", command=self.start_display, image=self.start_icon,
                                       compound=tk.LEFT)
        self.start_button.pack(pady=10)

        self.stop_button = ttk.Button(root, text="매장 영상 끄기", command=self.stop_display, image=self.stop_icon,
                                      compound=tk.LEFT)
        self.stop_button.pack(pady=10)

        self.view_button = ttk.Button(root, text="저장된 클립 보기", command=self.view_clip, image=self.view_icon,
                                      compound=tk.LEFT)
        self.view_button.pack(pady=10)

        self.exit_button = ttk.Button(root, text="종료", command=self.exit_app, image=self.stop_icon,
                                      compound=tk.LEFT)  # 종료 버튼 추가
        self.exit_button.pack(pady=10)

        # 초기 카메라 시작 및 처리 스레드 실행
        threading.Thread(target=self.record_video).start()  # 영상 촬영 및 처리 스레드 시작

    def load_icons(self):
        self.start_icon = ImageTk.PhotoImage(Image.open(".venv/icons/start.png").resize((20, 20), Image.LANCZOS))
        self.stop_icon = ImageTk.PhotoImage(Image.open(".venv/icons/stop.png").resize((20, 20), Image.LANCZOS))
        self.view_icon = ImageTk.PhotoImage(Image.open(".venv/icons/view.png").resize((20, 20), Image.LANCZOS))

    def start_display(self):
        if not self.is_displaying:
            self.is_displaying = True
            threading.Thread(target=self.display_video).start()  # 화면 출력 스레드 시작

    def stop_display(self):
        self.is_displaying = False

    def record_video(self):
        self.capture = cv2.VideoCapture(1, cv2.CAP_DSHOW)
        self.capture.set(cv2.CAP_PROP_FPS, 3)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        frame_buffer = []
        processed_frames_buffer = []
        while self.is_recording:
            ret, frame = self.capture.read()
            if ret:
                # 프레임을 버퍼에 추가
                frame_buffer.append(frame)
                if len(frame_buffer) == 30:
                    status, processed_frames = process_frame_sequence(frame_buffer, self.abnormal_dir)
                    frame_buffer = []
                    processed_frames_buffer.append(processed_frames)
                    if len(processed_frames_buffer) >= 10 and status != 'Theft':
                        while len(processed_frames_buffer) < 10:
                            processed_frames_buffer.pop(0)
                    # self.alert_and_save_clip(processed_frames_buffer)  # 이상 행동 발생 시 알림 및 클립 저장
                    # if status == 'Theft':
                    self.alert_and_save_clip(processed_frames_buffer)  # 이상 행동 발생 시 알림 및 클립 저장
            else:
                print("프레임을 가져올 수 없습니다.")
                break

    def alert_and_save_clip(self, frames):
        # 클립 저장을 백그라운드에서 처리
        threading.Thread(target=self.save_clip, args=(frames,)).start()

        # 알림창 띄우기
        threading.Thread(target=self.show_warning_message).start()

    def show_warning_message(self):
        # 알림창 띄우기
        tk.messagebox.showwarning("Warning", "이상 행동이 감지되었습니다!")

    def save_clip(self, processed_frames_buffer):
        # 클립 저장
        start_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        filename = os.path.join(self.abnormal_dir, f"{start_time}_Theft.mp4")
        frame_height, frame_width = processed_frames_buffer[0][0].shape[:2]
        out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), 15.0, (frame_width, frame_height))
        for frames in processed_frames_buffer:
            for frame in frames:
                out.write(frame)

        out.release()
        print(f"Theft video saved at {filename}")

    def display_video(self):
        while self.is_displaying:
            ret, frame = self.capture.read()
            if ret:
                cv2.imshow('Live Camera', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.is_displaying = False
                    break
            else:
                print("프레임을 가져올 수 없습니다.")
                break

        cv2.destroyAllWindows()

    def exit_app(self):
        self.stop_camera()  # 카메라와 관련된 모든 스레드 정지
        self.root.quit()  # Tkinter 애플리케이션 종료
        self.root.destroy()  # 윈도우 창 종료

    def stop_camera(self):
        self.is_recording = False
        self.is_displaying = False
        cv2.destroyAllWindows()  # OpenCV 창 종료
        if self.capture is not None:
            self.capture.release()  # 카메라 해제

    def view_clip(self):
        # 파일 선택 다이얼로그 열기
        filepath = askopenfilename(initialdir=self.video_dir, filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")])
        if filepath:
            # 선택한 파일을 비디오로 재생
            self.play_video(filepath)

    def play_video(self, filepath):
        cap = cv2.VideoCapture(filepath)
        if not cap.isOpened():
            messagebox.showerror("Error", "비디오를 열 수 없습니다.")
            return

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imshow("Saved Video", frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


# 애플리케이션 실행
if __name__ == "__main__":
    # 특정 경고를 무시하도록 설정
    warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
    root = tk.Tk()
    app = CameraApp(root)
    root.mainloop()
