"""
Face Recognition Based Attendance Management System
----------------------------------
Features:
- SQLite-based backend (attendance.db)
- Admin login (default: admin / admin123)
- Modern UI using CustomTkinter (dark theme + sidebar)
- Student registration (face capture via webcam)
- Model training using OpenCV LBPH
- Subject-wise attendance (date + time)
- Attendance reports + PDF export

Perfect for academic project demonstration.
"""

import os
import re
import sqlite3
from datetime import datetime
import threading
import queue

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

import customtkinter as ctk
from tkinter import messagebox

from db import get_conn, init_db, check_login, add_student, get_students_dict, get_students_by_db_id, add_attendance, get_subject_attendance, reset_system

# -------------------- PATHS / DIRECTORIES --------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
TRAINING_DIR = os.path.join(DATA_DIR, "training_images")
MODELS_DIR = os.path.join(DATA_DIR, "models")
ATTENDANCE_DIR = os.path.join(DATA_DIR, "attendance")
DB_PATH = os.path.join(BASE_DIR, "attendance.db")

HAAR_PATH = os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml")
MODEL_PATH = os.path.join(MODELS_DIR, "face_model.yml")

os.makedirs(TRAINING_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(ATTENDANCE_DIR, exist_ok=True)

# -------------------- CUSTOMTKINTER THEME --------------------
ctk.set_appearance_mode("dark")          # "light", "dark", "system"
ctk.set_default_color_theme("blue")      # "blue", "green", "dark-blue"



# -------------------- CORE FACE FUNCTIONS & ASYNC CAMERA -----------------------

class VideoCaptureThread:
    """Background thread to capture frames from webcam asynchronously to avoid GUI freezing."""
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.q = queue.Queue(maxsize=3)
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def _capture_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue
            # Keep only the freshest frames in the queue
            if self.q.full():
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    pass
            self.q.put(frame)

    def read(self):
        if not self.q.empty():
            return True, self.q.get()
        return False, None

    def is_opened(self):
        return self.cap.isOpened()

    def release(self):
        self.running = False
        self.thread.join(timeout=1.0)
        self.cap.release()


class CameraWindow(ctk.CTkToplevel):
    """Embedded camera window using CustomTkinter widgets and VideoCaptureThread."""
    def __init__(self, parent, title, mode, **kwargs):
        super().__init__(parent)
        self.title(title)
        self.geometry("680x620")
        self.resizable(False, False)
        
        # Modal setup
        self.transient(parent)
        self.grab_set()
        
        self.mode = mode  # "capture" or "attendance"
        self.kwargs = kwargs
        self.parent = parent
        
        # UI Setup
        self.label_title = ctk.CTkLabel(self, text=title, font=("Arial", 18, "bold"))
        self.label_title.pack(pady=15)
        
        self.video_frame = ctk.CTkFrame(self, width=640, height=480, fg_color="black")
        self.video_frame.pack(padx=20, pady=5)
        self.video_frame.pack_propagate(False)
        
        self.video_label = ctk.CTkLabel(self.video_frame, text="Initializing Camera...", text_color="white")
        self.video_label.pack(expand=True, fill="both")
        
        self.status_label = ctk.CTkLabel(self, text="Status: Ready.", font=("Arial", 13), text_color="#ffcc00")
        self.status_label.pack(pady=10)
        
        if self.mode == "capture":
            self.progress_bar = ctk.CTkProgressBar(self, width=400)
            self.progress_bar.pack(pady=5)
            self.progress_bar.set(0.0)
        else:
            self.progress_bar = None
            
        self.btn_close = ctk.CTkButton(
            self, 
            text="Close Camera" if self.mode == "attendance" else "Cancel Capture",
            fg_color="#aa3333", 
            hover_color="#cc4444", 
            command=self.on_close
        )
        self.btn_close.pack(pady=10)
        
        # Load Haar Cascade
        self.face_cascade = cv2.CascadeClassifier(HAAR_PATH)
        
        # CLAHE for lighting equalization
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        # State
        self.sample_count = 0
        self.max_samples = 60
        self.recognized_ids = set()
        
        if self.mode == "capture":
            self.db_id = self.kwargs.get("db_id")
            self.name = self.kwargs.get("name")
        else:
            self.subject = self.kwargs.get("subject")
            self.students_map = get_students_by_db_id()
            try:
                self.recognizer = cv2.face.LBPHFaceRecognizer_create()
            except AttributeError:
                messagebox.showerror(
                    "Error", "cv2.face module not available.\nInstall 'opencv-contrib-python'."
                )
                self.destroy()
                return
            self.recognizer.read(MODEL_PATH)

        # Start background capture thread
        self.vid_thread = VideoCaptureThread(0)
        if not self.vid_thread.is_opened():
            messagebox.showerror("Error", "Could not open webcam.")
            self.vid_thread.release()
            self.destroy()
            return
            
        self.running = True
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.update_loop()
        
    def update_loop(self):
        if not self.running:
            return
            
        ret, frame = self.vid_thread.read()
        if ret and frame is not None:
            # Mirror frame for natural GUI preview
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = self.clahe.apply(gray)
            
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
            
            if self.mode == "capture" and len(faces) > 1:
                self.status_label.configure(
                    text="Warning: Multiple faces detected! Clean the frame.",
                    text_color="red"
                )
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            else:
                for (x, y, w, h) in faces:
                    # Resize cropped face to standard size
                    face_crop = gray[y : y + h, x : x + w]
                    face_crop_resized = cv2.resize(face_crop, (200, 200), interpolation=cv2.INTER_AREA)
                    
                    if self.mode == "capture":
                        self.sample_count += 1
                        filename = f"{self.db_id}_{self.sample_count}.jpg"
                        path = os.path.join(TRAINING_DIR, filename)
                        cv2.imwrite(path, face_crop_resized)
                        
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                        self.status_label.configure(
                            text=f"Captured {self.sample_count} of {self.max_samples} images...",
                            text_color="#ffcc00"
                        )
                        if self.progress_bar:
                            self.progress_bar.set(self.sample_count / self.max_samples)
                            
                        if self.sample_count >= self.max_samples:
                            self.status_label.configure(
                                text=f"Captured all {self.max_samples} images successfully!",
                                text_color="#00ff99"
                            )
                            self.after(1000, self.on_close)
                            break
                            
                    elif self.mode == "attendance":
                        predicted_id, confidence = self.recognizer.predict(face_crop_resized)
                    
                    # LBPH Confidence threshold
                    if confidence < 58:
                        student_info = self.students_map.get(predicted_id)
                        if student_info:
                            sid, name = student_info
                            label_text = f"{sid} - {name}"
                            color = (0, 255, 0)
                            self.recognized_ids.add(sid)
                        else:
                            label_text = "Unknown"
                            color = (0, 0, 255)
                    else:
                        label_text = "Unknown"
                        color = (0, 0, 255)
                        
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(
                        frame,
                        label_text,
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        color,
                        2,
                    )
            
            # Render frame in CustomTkinter Label
            cv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(cv_img)
            pil_img = pil_img.resize((640, 480), Image.Resampling.LANCEZOS)
            ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(640, 480))
            
            self.video_label.configure(image=ctk_img, text="")
            self.video_label.image = ctk_img
            
        self.after(30, self.update_loop)
        
    def on_close(self):
        self.running = False
        if hasattr(self, "vid_thread"):
            self.vid_thread.release()
            
        # Log attendance on window close if in attendance mode
        if self.mode == "attendance" and self.recognized_ids:
            today_str = datetime.now().strftime("%Y_%m_%d")
            time_str = datetime.now().strftime("%H:%M:%S")
            for sid in self.recognized_ids:
                add_attendance(sid, self.subject, today_str, time_str)
                
            self.parent.status_label.configure(
                text=f"Attendance saved for {len(self.recognized_ids)} student(s) in {self.subject}.",
                text_color="#00ff99",
            )
        elif self.mode == "capture":
            if self.sample_count > 0:
                self.parent.status_label.configure(
                    text=f"Captured {self.sample_count} images for {self.name}.",
                    text_color="#00ff99",
                )
            else:
                self.parent.status_label.configure(
                    text="Registration cancelled. No faces captured.",
                    text_color="red"
                )
                
        self.grab_release()
        self.destroy()


class TrainingProgressWindow(ctk.CTkToplevel):
    """Temporary modal window to show model training progress without freezing dashboard."""
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Training Model")
        self.geometry("340x160")
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()
        
        self.label = ctk.CTkLabel(
            self, 
            text="Training AI Face Recognizer...\nPlease wait, do not close the app.", 
            font=("Arial", 13, "bold"),
            justify="center"
        )
        self.label.pack(pady=20)
        
        self.progress = ctk.CTkProgressBar(self, width=280)
        self.progress.pack(pady=10)
        self.progress.start()
        
        # Disable window closing button to prevent corrupting saved model file
        self.protocol("WM_DELETE_WINDOW", lambda: None)


def capture_faces(student_id: str, name: str, status_label: ctk.CTkLabel, parent_win: ctk.CTk):
    """Prepares registration with regex input validations and opens the camera window."""
    if not os.path.isfile(HAAR_PATH):
        messagebox.showerror("Error", f"Haarcascade file not found:\n{HAAR_PATH}")
        return

    student_id = student_id.strip()
    name = name.strip()

    if student_id == "" or name == "":
        messagebox.showwarning("Input Error", "Please enter both Student ID and Name.")
        return

    # Student ID: Alphanumeric, dashes, underscores only
    if not re.match(r"^[a-zA-Z0-9_-]+$", student_id):
        messagebox.showwarning(
            "Input Error", 
            "Student ID can only contain letters, numbers, dashes (-), and underscores (_)."
        )
        return
        
    if len(student_id) > 30:
        messagebox.showwarning("Input Error", "Student ID must be 30 characters or less.")
        return

    # Name: Letters, spaces, hyphens, apostrophes only
    if not re.match(r"^[a-zA-Z\s'-]+$", name):
        messagebox.showwarning(
            "Input Error", 
            "Student Name can only contain letters, spaces, hyphens (-), and apostrophes (')."
        )
        return

    if len(name) > 50:
        messagebox.showwarning("Input Error", "Student Name must be 50 characters or less.")
        return

    # Add student to DB to fetch auto-incremented primary key
    db_id = add_student(student_id, name)
    if db_id is None:
        messagebox.showerror("Error", "Database error: Could not register student.")
        return

    CameraWindow(parent_win, f"Capture: {name} ({student_id})", "capture", db_id=db_id, name=name)


def train_model(status_label: ctk.CTkLabel, parent_win: ctk.CTk):
    """Train LBPH face recognizer asynchronously on a background thread."""
    image_files = [
        f
        for f in os.listdir(TRAINING_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    if not image_files:
        messagebox.showwarning(
            "No Images", "No training images found. Please capture faces first."
        )
        return

    # Show loading progress overlay dialog
    progress_win = TrainingProgressWindow(parent_win)
    
    def run_training():
        try:
            recognizer = cv2.face.LBPHFaceRecognizer_create()
        except AttributeError:
            def on_error():
                progress_win.destroy()
                messagebox.showerror(
                    "Error",
                    "cv2.face module not available.\nInstall 'opencv-contrib-python'.",
                )
            parent_win.after(0, on_error)
            return

        faces = []
        ids = []

        for img_name in image_files:
            img_path = os.path.join(TRAINING_DIR, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            base = os.path.splitext(img_name)[0]
            parts = base.split("_")

            if len(parts) >= 2 and parts[0].isdigit():
                db_id = int(parts[0])
                faces.append(img)
                ids.append(db_id)

        if not faces:
            def on_no_valid():
                progress_win.destroy()
                messagebox.showwarning("No Valid Images", "No valid images with numeric labels found.")
            parent_win.after(0, on_no_valid)
            return

        try:
            recognizer.train(faces, np.array(ids))
            recognizer.save(MODEL_PATH)
            
            def on_complete():
                progress_win.destroy()
                status_label.configure(
                    text="Model trained successfully.", text_color="#00ff99"
                )
            parent_win.after(0, on_complete)
        except Exception as e:
            def on_exception():
                progress_win.destroy()
                messagebox.showerror("Error", f"Failed to train model:\n{e}")
            parent_win.after(0, on_exception)

    # Launch background thread
    threading.Thread(target=run_training, daemon=True).start()


def mark_attendance(status_label: ctk.CTkLabel, subject: str, parent_win: ctk.CTk):
    """Opens the modern camera window to recognize faces and mark attendance."""
    subject = subject.strip()
    if subject == "":
        messagebox.showwarning("Subject Missing", "Please enter the subject.")
        return

    if not os.path.exists(MODEL_PATH):
        messagebox.showwarning(
            "Model Missing", "No trained model found. Please train the model first."
        )
        return

    if not os.path.isfile(HAAR_PATH):
        messagebox.showerror("Error", f"Haarcascade file not found:\n{HAAR_PATH}")
        return

    students = get_students_by_db_id()
    if not students:
        messagebox.showwarning(
            "No Students", "No registered students in the database."
        )
        return

    CameraWindow(parent_win, f"Attendance: {subject}", "attendance", subject=subject)



# -------------------- REPORTS / ANALYTICS -----------------------
def get_subject_summary(subject: str):
    subject = subject.strip()
    if subject == "":
        messagebox.showwarning("Subject Missing", "Please enter the subject.")
        return None, None

    rows = get_subject_attendance(subject)
    if not rows:
        messagebox.showinfo("No Data", f"No attendance data for '{subject}'.")
        return None, None

    df = pd.DataFrame(rows, columns=["StudentID", "Name", "Date", "Time"])
    total_classes = df["Date"].nunique()

    summary = (
        df.groupby(["StudentID", "Name"])
        .size()
        .reset_index(name="PresentCount")
    )
    summary["TotalClasses"] = total_classes
    summary["Percentage"] = (summary["PresentCount"] / total_classes * 100).round(2)

    return summary, total_classes


def show_subject_report(status_label: ctk.CTkLabel, subject: str):
    summary, total_classes = get_subject_summary(subject)
    if summary is None:
        return

    win = ctk.CTkToplevel()
    win.title(f"Report - {subject}")
    win.geometry("660x420")

    scroll_frame = ctk.CTkScrollableFrame(win, width=630, height=380)
    scroll_frame.pack(expand=True, fill="both", padx=10, pady=10)

    headers = ["StudentID", "Name", "Present", "Total", "Percentage"]

    for j, h in enumerate(headers):
        lbl = ctk.CTkLabel(
            scroll_frame,
            text=h,
            font=("Arial", 13, "bold"),
        )
        lbl.grid(row=0, column=j, sticky="nsew", padx=10, pady=5)
        scroll_frame.grid_columnconfigure(j, weight=1)

    for i, row in summary.iterrows():
        values = [
            row["StudentID"],
            row["Name"],
            row["PresentCount"],
            row["TotalClasses"],
            f'{row["Percentage"]}%',
        ]
        for j, val in enumerate(values):
            lbl = ctk.CTkLabel(
                scroll_frame,
                text=str(val),
                font=("Arial", 12),
            )
            lbl.grid(row=i + 1, column=j, sticky="nsew", padx=10, pady=3)

    status_label.configure(
        text=f"Report generated for subject '{subject}'.", text_color="#00ff99"
    )


def export_subject_report_to_pdf(subject: str):
    summary, total_classes = get_subject_summary(subject)
    if summary is None:
        return

    pdf_path = os.path.join(DATA_DIR, f"{subject}_attendance_report.pdf")
    c = canvas.Canvas(pdf_path, pagesize=A4)
    width, height = A4

    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, f"Attendance Report - {subject}")

    c.setFont("Helvetica", 10)
    y = height - 90
    headers = ["StudentID", "Name", "Present", "Total", "Percentage"]
    line = " | ".join(headers)
    c.drawString(50, y, line)
    y -= 20

    for _, row in summary.iterrows():
        line = f"{row['StudentID']} | {row['Name']} | {row['PresentCount']} | {row['TotalClasses']} | {row['Percentage']}%"
        c.drawString(50, y, line)
        y -= 15
        if y < 50:
            c.showPage()
            y = height - 50

    c.save()
    messagebox.showinfo("PDF Exported", f"Report saved to:\n{pdf_path}")


# -------------------- MAIN UI (DASHBOARD) -----------------------
class MainApp(ctk.CTk):
    def __init__(self, username: str):
        super().__init__()

        self.username = username

        self.title("Face Recognition Attendance System")
        self.geometry("980x560")
        self.resizable(False, False)

        # --- Sidebar ---
        self.sidebar = ctk.CTkFrame(self, width=200, corner_radius=0)
        self.sidebar.pack(side="left", fill="y")

        self.logo_label = ctk.CTkLabel(
            self.sidebar,
            text="Attendance\nSystem",
            font=("Arial", 20, "bold"),
            justify="center",
        )
        self.logo_label.pack(pady=20)

        self.user_label = ctk.CTkLabel(
            self.sidebar,
            text=f"Logged in as:\n{self.username}",
            font=("Arial", 12),
            justify="left",
        )
        self.user_label.pack(pady=5)

        self.sidebar_buttons_frame = ctk.CTkFrame(self.sidebar)
        self.sidebar_buttons_frame.pack(pady=20, fill="x", padx=10)

        self.btn_dashboard = ctk.CTkButton(
            self.sidebar_buttons_frame,
            text="Dashboard",
            command=self.show_dashboard,
        )
        self.btn_dashboard.pack(pady=5, fill="x")

        self.btn_students = ctk.CTkButton(
            self.sidebar_buttons_frame,
            text="Students",
            command=self.show_students,
        )
        self.btn_students.pack(pady=5, fill="x")

        self.btn_attendance = ctk.CTkButton(
            self.sidebar_buttons_frame,
            text="Attendance",
            command=self.show_attendance,
        )
        self.btn_attendance.pack(pady=5, fill="x")

        self.btn_reports = ctk.CTkButton(
            self.sidebar_buttons_frame,
            text="Reports",
            command=self.show_reports,
        )
        self.btn_reports.pack(pady=5, fill="x")

        self.btn_logout = ctk.CTkButton(
            self.sidebar,
            text="Logout",
            fg_color="#aa3333",
            hover_color="#cc4444",
            command=self.logout,
        )
        self.btn_logout.pack(side="bottom", pady=15, fill="x", padx=10)

        # --- Main Content ---
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.pack(side="right", expand=True, fill="both", padx=10, pady=10)

        self.status_label = ctk.CTkLabel(
            self,
            text="Ready.",
            font=("Arial", 11),
            text_color="#00ff99",
        )
        self.status_label.pack(side="bottom", anchor="w", padx=15, pady=8)

        # Keep track of page frames
        self.current_page = None

        # Show dashboard initially
        self.show_dashboard()

    def clear_main(self):
        if self.current_page is not None:
            self.current_page.destroy()
            self.current_page = None

    # ----- PAGES -----
    def show_dashboard(self):
        self.clear_main()
        frame = ctk.CTkFrame(self.main_frame)
        frame.pack(expand=True, fill="both", padx=10, pady=10)

        title = ctk.CTkLabel(
            frame,
            text="Dashboard",
            font=("Arial", 20, "bold"),
        )
        title.pack(anchor="w", pady=10, padx=10)

        subtitle = ctk.CTkLabel(
            frame,
            text="Overview of the Face Recognition Attendance System",
            font=("Arial", 13),
        )
        subtitle.pack(anchor="w", padx=10)

        # Simple info cards
        cards_frame = ctk.CTkFrame(frame)
        cards_frame.pack(pady=20, padx=10, fill="x")

        # total students
        students_dict = get_students_dict()
        total_students = len(students_dict)

        conn = get_conn()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM attendance")
        total_attendance = cur.fetchone()[0]
        conn.close()

        card1 = ctk.CTkFrame(cards_frame, width=200, height=100, corner_radius=15)
        card1.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        ctk.CTkLabel(card1, text="Total Students", font=("Arial", 13, "bold")).pack(
            pady=5
        )
        ctk.CTkLabel(card1, text=str(total_students), font=("Arial", 20, "bold")).pack()

        card2 = ctk.CTkFrame(cards_frame, width=200, height=100, corner_radius=15)
        card2.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        ctk.CTkLabel(card2, text="Attendance Records", font=("Arial", 13, "bold")).pack(
            pady=5
        )
        ctk.CTkLabel(card2, text=str(total_attendance), font=("Arial", 20, "bold")).pack()

        cards_frame.grid_columnconfigure(0, weight=1)
        cards_frame.grid_columnconfigure(1, weight=1)

        # Reset System Button
        btn_reset = ctk.CTkButton(
            frame,
            text="Reset System Data",
            fg_color="#aa3333",
            hover_color="#cc4444",
            command=self.on_reset_system
        )
        btn_reset.pack(pady=20)

        self.current_page = frame
        self.status_label.configure(text="Dashboard loaded.", text_color="#00ff99")

    def on_reset_system(self):
        confirm = messagebox.askyesno(
            "Confirm Reset",
            "Are you sure you want to reset the system?\n\nThis will permanently delete:\n- All registered student records\n- All attendance records\n- All training face photos\n- The trained recognition model\n\nThis action cannot be undone."
        )
        if confirm:
            try:
                reset_system()
                if os.path.exists(TRAINING_DIR):
                    for f in os.listdir(TRAINING_DIR):
                        file_path = os.path.join(TRAINING_DIR, f)
                        if os.path.isfile(file_path):
                            os.unlink(file_path)
                if os.path.exists(MODEL_PATH):
                    os.unlink(MODEL_PATH)
                if os.path.exists(DATA_DIR):
                    for f in os.listdir(DATA_DIR):
                        if f.endswith(".pdf"):
                            file_path = os.path.join(DATA_DIR, f)
                            os.unlink(file_path)
                messagebox.showinfo("Reset Complete", "The system has been successfully reset to its default state.")
                self.show_dashboard()
            except Exception as e:
                messagebox.showerror("Error", f"An error occurred during reset:\n{e}")

    def show_students(self):
        self.clear_main()
        frame = ctk.CTkFrame(self.main_frame)
        frame.pack(expand=True, fill="both", padx=10, pady=10)

        title = ctk.CTkLabel(
            frame,
            text="Student Registration",
            font=("Arial", 20, "bold"),
        )
        title.pack(anchor="w", pady=10, padx=10)

        form = ctk.CTkFrame(frame)
        form.pack(pady=20, padx=20, fill="x")

        lbl_id = ctk.CTkLabel(form, text="Student ID:", font=("Arial", 13))
        lbl_id.grid(row=0, column=0, sticky="w", pady=5, padx=5)
        entry_id = ctk.CTkEntry(form, font=("Arial", 13))
        entry_id.grid(row=0, column=1, sticky="ew", pady=5, padx=5)

        lbl_name = ctk.CTkLabel(form, text="Name:", font=("Arial", 13))
        lbl_name.grid(row=1, column=0, sticky="w", pady=5, padx=5)
        entry_name = ctk.CTkEntry(form, font=("Arial", 13))
        entry_name.grid(row=1, column=1, sticky="ew", pady=5, padx=5)

        form.grid_columnconfigure(1, weight=1)

        buttons = ctk.CTkFrame(frame)
        buttons.pack(pady=10)

        def on_capture():
            sid = entry_id.get().strip()
            name = entry_name.get().strip()
            capture_faces(sid, name, self.status_label, self)

        def on_train():
            train_model(self.status_label, self)

        def on_clear():
            entry_id.delete(0, 'end')
            entry_name.delete(0, 'end')
            self.status_label.configure(text="Fields cleared.", text_color="#00ff99")

        ctk.CTkButton(buttons, text="Capture Faces", command=on_capture).grid(
            row=0, column=0, padx=10, pady=5
        )
        ctk.CTkButton(buttons, text="Train Model", command=on_train).grid(
            row=0, column=1, padx=10, pady=5
        )
        ctk.CTkButton(buttons, text="Clear Fields", command=on_clear, fg_color="#555555", hover_color="#666666").grid(
            row=0, column=2, padx=10, pady=5
        )

        self.current_page = frame
        self.status_label.configure(text="Student registration loaded.", text_color="#00ff99")

    def show_attendance(self):
        self.clear_main()
        frame = ctk.CTkFrame(self.main_frame)
        frame.pack(expand=True, fill="both", padx=10, pady=10)

        title = ctk.CTkLabel(
            frame,
            text="Take Attendance",
            font=("Arial", 20, "bold"),
        )
        title.pack(anchor="w", pady=10, padx=10)

        form = ctk.CTkFrame(frame)
        form.pack(pady=20, padx=20, fill="x")

        lbl_subject = ctk.CTkLabel(form, text="Subject:", font=("Arial", 13))
        lbl_subject.grid(row=0, column=0, sticky="w", pady=5, padx=5)
        entry_subject = ctk.CTkEntry(
            form, font=("Arial", 13), placeholder_text="e.g. OS, DBMS, AI"
        )
        entry_subject.grid(row=0, column=1, sticky="ew", pady=5, padx=5)

        form.grid_columnconfigure(1, weight=1)

        def on_take_attendance():
            subject = entry_subject.get().strip()
            mark_attendance(self.status_label, subject, self)

        ctk.CTkButton(
            frame, text="Start Attendance", command=on_take_attendance
        ).pack(pady=10)

        self.current_page = frame
        self.status_label.configure(text="Attendance page loaded.", text_color="#00ff99")

    def show_reports(self):
        self.clear_main()
        frame = ctk.CTkFrame(self.main_frame)
        frame.pack(expand=True, fill="both", padx=10, pady=10)

        title = ctk.CTkLabel(
            frame,
            text="Reports & Analytics",
            font=("Arial", 20, "bold"),
        )
        title.pack(anchor="w", pady=10, padx=10)

        form = ctk.CTkFrame(frame)
        form.pack(pady=20, padx=20, fill="x")

        lbl_subject = ctk.CTkLabel(form, text="Subject:", font=("Arial", 13))
        lbl_subject.grid(row=0, column=0, sticky="w", pady=5, padx=5)
        entry_subject = ctk.CTkEntry(
            form, font=("Arial", 13), placeholder_text="e.g. OS, DBMS, AI"
        )
        entry_subject.grid(row=0, column=1, sticky="ew", pady=5, padx=5)

        form.grid_columnconfigure(1, weight=1)

        buttons = ctk.CTkFrame(frame)
        buttons.pack(pady=10)

        def on_show():
            subject = entry_subject.get().strip()
            show_subject_report(self.status_label, subject)

        def on_export():
            subject = entry_subject.get().strip()
            export_subject_report_to_pdf(subject)

        ctk.CTkButton(buttons, text="View Report", command=on_show).grid(
            row=0, column=0, padx=10, pady=5
        )
        ctk.CTkButton(buttons, text="Export PDF", command=on_export).grid(
            row=0, column=1, padx=10, pady=5
        )

        self.current_page = frame
        self.status_label.configure(text="Reports page loaded.", text_color="#00ff99")

    def logout(self):
        self.destroy()
        login_window()  # restart login


# -------------------- LOGIN WINDOW -----------------------
def login_window():
    init_db()

    login = ctk.CTk()
    login.title("Login - Attendance System")
    login.geometry("380x260")
    login.resizable(False, False)

    ctk.CTkLabel(
        login,
        text="Face Recognition Attendance",
        font=("Arial", 18, "bold"),
    ).pack(pady=10)

    frame = ctk.CTkFrame(login)
    frame.pack(pady=10, padx=20, fill="x")

    user_entry = ctk.CTkEntry(frame, placeholder_text="Username", font=("Arial", 13))
    user_entry.pack(pady=5, fill="x")

    pass_entry = ctk.CTkEntry(
        frame, placeholder_text="Password", font=("Arial", 13), show="*"
    )
    pass_entry.pack(pady=5, fill="x")

    status = ctk.CTkLabel(frame, text="", font=("Arial", 11))
    status.pack(pady=5)

    def on_login():
        u = user_entry.get().strip()
        p = pass_entry.get().strip()
        if check_login(u, p):
            login.destroy()
            app = MainApp(username=u)
            app.mainloop()
        else:
            status.configure(text="Invalid credentials", text_color="red")

    ctk.CTkButton(login, text="Login", command=on_login).pack(pady=10)

    # Show default creds for demo
    hint = ctk.CTkLabel(
        login,
        text="Demo login: admin / admin123",
        font=("Arial", 10),
        text_color="#aaaaaa",
    )
    hint.pack(side="bottom", pady=5)

    login.mainloop()


if __name__ == "__main__":
    login_window()
