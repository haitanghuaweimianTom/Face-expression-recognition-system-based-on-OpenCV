# -*- coding: utf-8 -*-

import tkinter as tk
from tkinter import Label, Button
import cv2
from PIL import Image, ImageTk
import numpy as np
import sys
import os
from tensorflow.keras.models import load_model

# 获取程序运行时的根目录（兼容打包后环境）
def get_base_path():
    if getattr(sys, 'frozen', False):
        # 打包后环境：获取执行文件所在目录（dist文件夹）
        return os.path.dirname(sys.executable)
    else:
        # 开发环境：获取当前文件所在目录
        return os.path.dirname(os.path.abspath(__file__))

BASE_PATH = get_base_path()

# 模型文件路径（打包后需放在dist文件夹中）
MODEL_PATH = os.path.join(BASE_PATH, "best_emotion_model.h5")
CASCADE_PATH = os.path.join(BASE_PATH, "haarcascade_frontalface_default.xml")

# 表情标签（与模型训练类别顺序一致，确保中文）
EMOTION_DICT = {
    0: "愤怒",
    1: "厌恶",
    2: "恐惧",
    3: "开心",
    4: "中性",
    5: "悲伤",
    6: "惊讶"
}

global last_frame, predicted_emotion, emotion_probabilities, cap
last_frame = np.zeros((500, 600, 3), dtype=np.uint8)
predicted_emotion = "等待识别..."
emotion_probabilities = {}  # 存储各情绪的概率
cap = None  # 全局摄像头对象

def load_emotion_model():
    # 检查模型文件是否存在
    if not os.path.exists(MODEL_PATH):
        print(f"\n模型文件不存在：{MODEL_PATH}")
        print("请确保best_emotion_model.h5文件已放在程序所在目录")
        sys.exit(1)
    
    try:
        model = load_model(MODEL_PATH)
        print(f"成功加载模型：{MODEL_PATH}")
        return model
    except Exception as e:
        print(f"\n模型加载失败：{str(e)}")
        print("请检查：1.模型文件是否完整 2.是否与程序同目录")
        sys.exit(1)

def process_camera(model):
    global cap
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("\n摄像头无法打开，尝试：")
        print("1. 关闭占用摄像头的程序（微信/Zoom等）")
        print("2. 修改代码中cv2.VideoCapture(0)为cv2.VideoCapture(1)")
        return

    # 加载人脸检测模型
    if not os.path.exists(CASCADE_PATH):
        print(f"\n人脸模型不存在：{CASCADE_PATH}")
        print("请确保haarcascade_frontalface_default.xml文件已放在程序所在目录")
        cap.release()
        return
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

    def update_frame():
        global last_frame, predicted_emotion, emotion_probabilities
        ret, frame = cap.read()
        if not ret:
            lmain.after(100, update_frame)
            return

        # 画面预处理（镜像+缩放）
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (600, 500))

        # 人脸检测
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(40, 40))

        # 表情识别
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            # 人脸预处理
            roi = cv2.resize(gray[y:y + h, x:x + w], (48, 48)) / 255.0
            roi_input = np.expand_dims(np.expand_dims(roi, -1), 0)
            # 预测
            pred = model.predict(roi_input, verbose=0)
            # 获取各情绪概率并格式化
            emotion_probabilities = {EMOTION_DICT[i]: round(float(pred[0][i]) * 100, 2) for i in range(len(EMOTION_DICT))}
            # 获取概率最大的情绪
            max_emotion = max(emotion_probabilities, key=emotion_probabilities.get)
            predicted_emotion = max_emotion

        # 更新显示
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_tk = ImageTk.PhotoImage(Image.fromarray(frame_rgb))
        lmain.imgtk = img_tk
        lmain.config(image=img_tk)
        # 显示当前表情和各情绪占比
        emotion_display = f"当前表情：{predicted_emotion}\n\n各情绪占比：\n"
        for emotion, prob in emotion_probabilities.items():
            emotion_display += f"{emotion}: {prob}%\n"
        lresult.config(text=emotion_display, font=("SimHei", 16, "bold"))
        lmain.after(10, update_frame)

    update_frame()

def init_gui(model):
    global root, lmain, lresult
    root = tk.Tk()
    root.title("表情识别系统")
    root.geometry("1000x700+100+50")
    root.config(bg="#2c3e50")

    # 标题
    Label(root, text="实时表情识别", font=("SimHei", 32, "bold"), 
          bg="#2c3e50", fg="white").pack(pady=20)

    # 摄像头显示
    lmain = Label(root, bg="#34495e")
    lmain.place(x=50, y=120, width=600, height=500)

    # 结果显示
    lresult = Label(root, text="等待识别...", 
                    font=("SimHei", 16, "bold"), bg="#34495e", fg="white")
    lresult.place(x=700, y=200, width=250, height=300)

    # 退出按钮
    def on_exit():
        if cap is not None:
            cap.release()
        root.destroy()

    Button(root, text="退出", command=on_exit,
           font=("SimHei", 18, "bold"), bg="#e74c3c", fg="white").pack(side=tk.BOTTOM, pady=30)

    # 启动识别
    process_camera(model)
    root.mainloop()

if __name__ == "__main__":
    model = load_emotion_model()
    global lmain, lresult
    init_gui(model)