# db.py
import sqlite3
import os
import hashlib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "attendance.db")


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def hash_password(password: str) -> str:
    salt = "attendance_system_salt_123"
    return hashlib.sha256((password + salt).encode('utf-8')).hexdigest()


def init_db():
    conn = get_conn()
    cur = conn.cursor()

    # Admin / teacher users
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    """)

    # Students
    cur.execute("""
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL
        )
    """)

    # Attendance
    cur.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT NOT NULL,
            subject TEXT NOT NULL,
            date TEXT NOT NULL,
            time TEXT NOT NULL,
            FOREIGN KEY (student_id) REFERENCES students(student_id) ON DELETE CASCADE
        )
    """)

    # Create default admin if not exists, or migrate to hash
    cur.execute("SELECT id, password FROM users WHERE username = ?", ("admin",))
    row = cur.fetchone()
    if row is None:
        hashed = hash_password("admin123")
        cur.execute("INSERT INTO users (username, password) VALUES (?, ?)",
                    ("admin", hashed))
    elif len(row[1]) != 64:
        # Migrate plaintext password to hashed
        hashed = hash_password(row[1])
        cur.execute("UPDATE users SET password = ? WHERE id = ?", (hashed, row[0]))

    conn.commit()
    conn.close()


def add_student(student_id, name):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        INSERT OR IGNORE INTO students (student_id, name)
        VALUES (?, ?)
    """, (student_id, name))
    conn.commit()
    cur.execute("SELECT id FROM students WHERE student_id = ?", (student_id,))
    row = cur.fetchone()
    conn.close()
    return row[0] if row else None


def get_students_dict():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT student_id, name FROM students")
    rows = cur.fetchall()
    conn.close()
    return {r[0]: r[1] for r in rows}


def get_students_by_db_id():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT id, student_id, name FROM students")
    rows = cur.fetchall()
    conn.close()
    return {r[0]: (r[1], r[2]) for r in rows}


def check_login(username, password):
    conn = get_conn()
    cur = conn.cursor()
    hashed = hash_password(password)
    cur.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, hashed))
    row = cur.fetchone()
    conn.close()
    return row is not None


def add_attendance(student_id, subject, date, time):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO attendance (student_id, subject, date, time)
        VALUES (?, ?, ?, ?)
    """, (student_id, subject, date, time))
    conn.commit()
    conn.close()


def get_subject_attendance(subject):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT a.student_id, s.name, a.date, a.time
        FROM attendance a
        LEFT JOIN students s ON a.student_id = s.student_id
        WHERE a.subject = ?
        ORDER BY a.date, a.time
    """, (subject,))
    rows = cur.fetchall()
    conn.close()
    return rows


def reset_system():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS attendance")
    cur.execute("DROP TABLE IF EXISTS students")
    cur.execute("DROP TABLE IF EXISTS users")
    conn.commit()
    conn.close()
    init_db()


