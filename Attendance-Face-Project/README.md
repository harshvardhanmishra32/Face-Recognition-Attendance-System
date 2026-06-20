# Face Recognition Attendance Management System

A modern, secure, and responsive desktop application built with Python, OpenCV, SQLite, and CustomTkinter for automated face-recognition-based attendance tracking. Designed with an asynchronous webcam capture pipeline to ensure a highly responsive user experience.

---

## 🌟 Key Features

- 👤 **Secure Login System**: Salted SHA-256 password hashing for administrative access.
- ⚡ **Asynchronous Webcam Stream**: Contained camera preview embedded directly in the CustomTkinter UI. Non-blocking frame capture ensures the dashboard remains interactive.
- 🎯 **Advanced ML Preprocessing**: Standardizes face crops to `200x200` pixels and applies Contrast Limited Adaptive Histogram Equalization (CLAHE) for robust face detection/recognition under varying lighting conditions.
- 🚨 **Multi-Face Prevention**: Automatic checks alert administrators and halt recording if multiple faces are detected in the capture window, keeping the dataset clean.
- 🎓 **Integer-Mapped ML Pipeline**: Maps face recognition labels to auto-incrementing SQLite database primary key IDs, preventing name collisions or alphanumeric string errors.
- 📊 **Scrollable Reports & Exports**: Generate subject-specific reports inside scrollable grids and export them to formatted A4 PDF reports.
- 🧹 **Reset & Clear Tools**: One-click system reset to wipe database logs, training photos, and model weights for clean testing.

---

## 🛠️ Tech Stack

- **Core Logic**: Python 3
- **GUI Engine**: CustomTkinter (Modernized Tkinter wrapping)
- **Computer Vision & ML**: OpenCV (Haar Cascades + Local Binary Patterns Histograms - LBPH)
- **Database Backend**: SQLite3 (with foreign key enforcement)
- **PDF Generation**: ReportLab
- **Data Structuring**: Pandas

---

## 📁 Project Directory Structure

```text
├── data/
│   ├── training_images/   # Preprocessed face photos (auto-generated)
│   └── models/            # Trained face_model.yml weights (auto-generated)
├── attendance.db          # SQLite3 database (auto-generated)
├── db.py                  # Database helper operations & migrations
├── attendance_app.py      # Main GUI Dashboard, Camera controllers, & report generator
├── requirements.txt       # Project python dependencies
├── .gitignore             # Git ignore list
└── README.md              # Project documentation
```

---

## 🗄️ Database Schema

The database relies on three tables with relational integrity (Foreign Key Constraints enabled):

```mermaid
erDiagram
    users {
        int id PK
        text username UNIQUE
        text password "SHA-256 Hashed"
    }
    students {
        int id PK "Auto-Increment"
        text student_id UNIQUE "User ID e.g. STU101"
        text name
    }
    attendance {
        int id PK
        text student_id FK
        text subject
        text date
        text time
    }
    students ||--o{ attendance : "logs"
```

---

## 🚀 Setup & Installation

### Prerequisites
- Python 3.8 or above installed on your system.
- An integrated or external webcam.

### Installation Steps

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/Attendance-Face-Project.git
   cd Attendance-Face-Project
   ```

2. **Create and activate a virtual environment**:
   ```bash
   # On macOS/Linux:
   python3 -m venv .venv
   source .venv/bin/activate

   # On Windows:
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. **Install the dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Launch the application**:
   ```bash
   python attendance_app.py
   ```

---

## 📖 Usage Guide

### 1. Administrative Login
- Launch the application to open the Login window.
- Enter the default demo credentials:
  - **Username**: `admin`
  - **Password**: `admin123`
- *Note: On the first successful login, the system will automatically upgrade the database to salt and hash the password in SHA-256.*

### 2. Student Registration
- Go to the **Students** tab.
- Enter a unique **Student ID** (e.g. `CS-2026`) and the **Name** (e.g. `John Doe`).
- Click **Capture Faces**. The webcam popup will open. Keep your face centered. It will automatically take 60 samples and close.
- *Note: If another face enters the frame, the camera window will highlight it in red and temporarily halt capturing to prevent data pollution.*

### 3. Model Training
- Once face photos have been recorded, click the **Train Model** button on the registration page.
- A loading overlay will appear while the model compiles on a background thread. The dashboard will remain fully responsive.

### 4. Taking Attendance
- Navigate to the **Attendance** tab.
- Enter the **Subject Name** (e.g. `AI` or `OS`).
- Click **Start Attendance**. Center your face in the webcam window.
- If recognized, a green tracking rectangle with your details will display, logging you into the local session.
- Click **Close Camera** to save the attendance records into the SQLite database.

### 5. Viewing & Exporting Reports
- Navigate to the **Reports** tab.
- Enter the subject name and click **View Report** to scroll through the attendance statistics.
- Click **Export PDF** to generate an A4 report saved under the `data/` folder.
