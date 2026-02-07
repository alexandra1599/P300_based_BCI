"""
Project Control Panel (Simplified, no polling)
Requires: pip install PySide6 psutil

Repo layout assumed :
~/Task Code/Gui/
    control_panel.py
    UTIL_marker_stream.py
    config.py
"""

import os, sys, shlex, time, re, tempfile, shutil, socket, subprocess, pathlib
from dataclasses import dataclass, field
from typing import Optional, Dict

from PySide6.QtCore import Qt, QTimer, QProcess, QByteArray, QSize
from PySide6.QtGui import QAction, QClipboard, QTextCursor
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QTabWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QComboBox,
    QCheckBox,
    QGridLayout,
    QLineEdit,
    QTextEdit,
    QGroupBox,
    QFormLayout,
    QMessageBox,
    QSplitter,
    QToolBar,
    QStyle,
)

# ------------------ Paths & Constants -----------------------------
ROOT = os.path.expanduser("/home/alexandra-admin/Documents/PhD/Task Code/GUI")
CONFIG_PY = os.path.join(ROOT, "config.py")
MARKER_PY = os.path.join(ROOT, "UTIL_marker_stream.py")
OFFLINE_TASK = os.path.join(ROOT, "nback.py")
ONLINE_TASK = os.path.join(ROOT, "nback_online.py")
INIT_SH = os.path.join(ROOT, "initialize_devices.sh")

UDP_MARKER = ("127.0.0.1", 12345)

# Driver choices
DRIVERS = ["nback", "nback_online"]


# ------------------ Config read/write helpers -----------------------
SUBJECT_RE = re.compile(r'^(TRAINING_SUBJECT\s*=\s*)([\'"])([^\'"]+)\2\s*$', re.M)


def read_text(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8:") as f:
            return f.read()
    except FileNotFoundError:
        return ""


def write_atomic(path: str, text: str):
    tmp = tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8")
    try:
        tmp.write(text)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp.close()
        # No backup copy
        os.replace(tmp.name, path)
    except Exception:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass
        raise


def read_training_subject(default="P001"):
    txt = read_text(CONFIG_PY)
    m = SUBJECT_RE.search(txt)
    return m.group(3) if m else default


def write_training_subject(val: str):
    txt = read_text(CONFIG_PY)
    if SUBJECT_RE.search(txt):
        # Use \g<1> to avoid \11 ambiguity
        new = SUBJECT_RE.sub(rf'\g<1>"{val}"', txt)
    else:
        sep = "" if (txt.endswith("\n") or txt == "") else "\n"
        new = txt + f'{sep}TRAINING_SUBJECT = "{val}"\n'
    write_atomic(CONFIG_PY, new)


# -------------------- UDP readiness probe -------------------
def _is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.bind((host, port))
        s.close()
        return False  # bind worked -> nobody is listening
    except OSError:
        s.close()
        return True  # address in use -> listener present


# --------------------- Process model ---------------------------
@dataclass
class Proc:
    name: str
    cmd: Optional[str]
    cwd: str
    env: Dict[str, str] = field(default_factory=dict)
    q: Optional[QProcess] = None
    status: str = "stopped"  # stopped|starting|running|error
    pid: Optional[int] = None
    out: bytearray = field(default_factory=bytearray)
    err: bytearray = field(default_factory=bytearray)


# --------------------- Main Window ------------------------------
class ControlPanel(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Nback Control Panel")
        self.resize(1250, 800)

        # State
        self.driver_choice = DRIVERS[0]
        self.training_subject = read_training_subject()

        # Procs (QProcess-managed)
        conda_python = "/home/alexandra-admin/opt/miniconda/envs/mne/bin/python"
        self.driver = Proc("Experimental Driver", None, ROOT)

        # Terminal processes
        self.marker_term: Optional[QProcess] = None
        self.labrec_term: Optional[QProcess] = None
        self.eego_term: Optional[QProcess] = None

        # Logs
        self._log_buffers: Dict[str, str] = {"Marker": "", "Driver": "", "Panel": ""}
        self._current_log_target = "Panel"

        # Build UI (buttons/labels defined here)
        self._build_ui()

        # Configure initial commands
        self._set_cmds_for_driver()

        # Initialize LEDs based on preferences (no polling)
        self._set_led(self.lbl_marker, "stopped")
        self._set_led(self.lbl_driver, "stopped")
        self._set_led(self.lbl_eego, "stopped")
        # Cheap timer: keep LEDs for QProcess-managed procs in sync
        self.ui_timer = QTimer(self)
        self.ui_timer.setInterval(400)
        self.ui_timer.timeout.connect(self._tick)
        self.ui_timer.start()

    # ------------- UI Build ------------------------

    def _build_ui(self):
        self._building_ui = True

        tb = QToolBar("Main")
        tb.setIconSize(QSize(18, 18))
        self.addToolBar(tb)
        act_init = QAction(
            self.style().standardIcon(QStyle.SP_ComputerIcon),
            "Initialize (open script)",
            self,
        )
        act_init.triggered.connect(self.on_initialize)
        tb.addAction(act_init)

        tabs = QTabWidget()
        self.setCentralWidget(tabs)

        # Main tab
        main = QWidget()
        tabs.addTab(main, "Main")
        mv = QVBoxLayout(main)

        # Top row: Driver + Subject + Tools
        top = QHBoxLayout()
        mv.addLayout(top)

        # Driver
        gb_driver = QGroupBox("Driver")
        fd = QHBoxLayout(gb_driver)
        self.cmb_driver = QComboBox()
        self.cmb_driver.addItems(DRIVERS)
        self.cmb_driver.setCurrentText(self.driver_choice)
        self.cmb_driver.currentTextChanged.connect(self.on_driver_choice_changed)
        fd.addWidget(QLabel("Experimental Driver:"))
        fd.addWidget(self.cmb_driver)
        top.addWidget(gb_driver, 2)

        # Subject
        gb_subj = QGroupBox("Subject")
        fs = QHBoxLayout(gb_subj)
        self.cmb_subject = QComboBox()
        self.cmb_subject.setEditable(True)
        self.cmb_subject.addItem(self.training_subject)
        self.cmb_subject.setCurrentText(self.training_subject)
        btn_save_subj = QPushButton("Save")
        btn_copy_subj = QPushButton("Copy")
        btn_save_subj.clicked.connect(self.on_save_subject)
        btn_copy_subj.clicked.connect(self.on_copy_subject)
        fs.addWidget(self.cmb_subject, 1)
        fs.addWidget(btn_save_subj)
        fs.addWidget(btn_copy_subj)
        top.addWidget(gb_subj, 2)

        # Utilities
        gb_utils = QGroupBox("Utilities")
        fu = QHBoxLayout(gb_utils)
        self.btn_mne = QPushButton("Open MNE_LSL Viewer")
        self.btn_mne.clicked.connect(self.on_open_mne_viewer)
        fu.addWidget(self.btn_mne)
        top.addWidget(gb_utils)

        # Middle: Controls + logs
        split = QSplitter()
        mv.addWidget(split, 1)
        controls = QWidget()
        split.addWidget(controls)
        grid = QGridLayout(controls)

        row = 0
        # eegoSports
        self.lbl_eego = QLabel("●")
        self._set_led(self.lbl_eego, "stopped")
        grid.addWidget(QLabel("<b>eegoSports</b>"), row, 0)
        grid.addWidget(self.lbl_eego, row, 1)
        btn_eego = QPushButton("Open eegoSports")
        btn_eego.clicked.connect(self.on_open_eego)
        grid.addWidget(btn_eego, row, 2)
        row += 1

        # Marker
        self.lbl_marker = QLabel("●")
        self._set_led(self.lbl_marker, "stopped")
        grid.addWidget(QLabel("<b>Marker Stream</b>"), row, 0)
        grid.addWidget(self.lbl_marker, row, 1)
        self.btn_marker_start = QPushButton("Start")
        self.btn_marker_stop = QPushButton("Stop")
        self.btn_marker_refresh = QPushButton("Refresh")
        self.btn_marker_start.clicked.connect(self.on_marker_start)
        self.btn_marker_stop.clicked.connect(self.on_marker_stop)
        self.btn_marker_refresh.clicked.connect(self.on_marker_refresh)
        grid.addWidget(self.btn_marker_start, row, 2)
        grid.addWidget(self.btn_marker_stop, row, 3)
        grid.addWidget(self.btn_marker_refresh, row, 4)
        row += 1

        # ===== LabRecorder =====
        self.lbl_labrec = QLabel("●")
        self._set_led(self.lbl_labrec, "stopped")
        grid.addWidget(QLabel("<b>LabRecorder</b>"), row, 0)
        grid.addWidget(self.lbl_labrec, row, 1)
        btn_labrec = QPushButton("Open LabRecorder")
        btn_labrec.clicked.connect(self.on_open_labrec)
        grid.addWidget(btn_labrec, row, 2)
        row += 1

        # ===== Driver =====
        self.lbl_driver = QLabel("●")
        self._set_led(self.lbl_driver, "stopped")
        grid.addWidget(QLabel("<b>Experimental Driver</b>"), row, 0)
        grid.addWidget(self.lbl_driver, row, 1)
        self.btn_driver_start = QPushButton("Start")
        self.btn_driver_stop = QPushButton("Stop")
        self.btn_driver_start.clicked.connect(self.on_driver_start)
        self.btn_driver_stop.clicked.connect(self.on_driver_stop)
        grid.addWidget(self.btn_driver_start, row, 2)
        grid.addWidget(self.btn_driver_stop, row, 3)
        row += 1

        # External apps info
        grid.addWidget(
            QLabel(
                "<i>External Apps:</i> eegoSports, LabRecorder (use Initialize / buttons)"
            ),
            row,
            0,
            1,
            5,
        )
        row += 1

        # ===== Logs Pane =====
        logw = QWidget()
        split.addWidget(logw)
        vl = QVBoxLayout(logw)

        pick_row = QHBoxLayout()
        self.log_title = QLabel("Logs:")
        self.log_selector = QComboBox()
        self.log_selector.addItems(["Marker", "FES", "Driver", "Robot", "Panel"])
        self.log_selector.setCurrentText(self._current_log_target)
        self.log_selector.currentTextChanged.connect(self._on_log_target_changed)
        pick_row.addWidget(self.log_title)
        pick_row.addStretch(1)
        pick_row.addWidget(QLabel("View:"))
        pick_row.addWidget(self.log_selector)

        self.txt_logs = QTextEdit()
        self.txt_logs.setReadOnly(True)
        self.txt_logs.setLineWrapMode(QTextEdit.NoWrap)

        vl.addLayout(pick_row)
        vl.addWidget(self.txt_logs, 1)

        self._building_ui = False
        self._refresh_log_view()

    # ---------- LED helper ----------
    def _set_led(self, label: QLabel, state: str):
        color = {
            "stopped": "#888",
            "starting": "#e6a700",  # yellow
            "running": "#18a558",  # green
            "error": "#c62828",  # red
        }.get(state, "#888")
        label.setText("●")
        label.setStyleSheet(f"color: {color}; font-size: 18px;")

    # ---------- Command wiring ----------
    def _set_cmds_for_driver(self):
        # Driver (local python) based on driver_choice
        conda_python = "/home/alexandra-admin/opt/miniconda/envs/mne/bin/python"

        if self.driver_choice == "nback":
            driver_path = OFFLINE_TASK
        elif self.driver_choice == "nback_online":
            driver_path = ONLINE_TASK
        else:
            driver_path = OFFLINE_TASK

        self.driver.cmd = f'{conda_python} -u "{driver_path}"'

        # Update env on driver only
        self.driver.env["PYTHONUNBUFFERED"] = "1"
        self.driver.env["TRAINING_SUBJECT"] = self.training_subject

    # ---------- Actions ----------
    def on_initialize(self):
        if not os.path.exists(INIT_SH):
            QMessageBox.warning(self, "Missing", f"Not found:\n{INIT_SH}")
            return
        cmd = f'gnome-terminal -- bash -lc "chmod +x \\"{INIT_SH}\\"; \\"{INIT_SH}\\"; exec bash"'
        subprocess.Popen(cmd, shell=True)
        QMessageBox.information(
            self, "Initialize", "Opened initialize_devices.sh in a new terminal."
        )

    def on_driver_choice_changed(self, text: str):
        self.driver_choice = text
        self._set_cmds_for_mode_and_driver()
        self._append_log(
            "Panel", f"[{self._ts()}] Driver selected: {self.driver_choice}\n"
        )

    def on_save_subject(self):
        val = self.cmb_subject.currentText().strip()
        if not val:
            QMessageBox.warning(self, "Subject", "Subject cannot be empty.")
            return
        self.training_subject = val
        write_training_subject(val)
        for p in (self.marker, self.driver):
            p.env["TRAINING_SUBJECT"] = self.training_subject
        self._append_log("Panel", f"[{self._ts()}] TRAINING_SUBJECT saved: {val}\n")

    def on_copy_subject(self):
        val = self.cmb_subject.currentText().strip()
        QApplication.clipboard().setText(val, QClipboard.Clipboard)
        self._append_log("Panel", f"[{self._ts()}] Copied subject: {val}\n")

    def on_open_mne_viewer(self):
        self._spawn_external("mne-lsl viewer")
        self._append_log("Panel", f"[{self._ts()}] Opened mne-lsl viewer\n")

    # ----- Marker -----

    def on_marker_start(self):
        # If already open, do nothing
        if self.marker_term and self.marker_term.state() != QProcess.NotRunning:
            return

        self.marker_term = QProcess(self)

        # When the terminal starts, LED -> green
        self.marker_term.started.connect(
            lambda: (
                self._set_led(self.lbl_marker, "running"),
                self._append_log(
                    "Panel", f"[{self._ts()}] Marker Stream terminal opened\n"
                ),
            )
        )

        # When it closes, LED -> gray
        def _marker_closed(code, status):
            self._set_led(self.lbl_marker, "stopped")
            self._append_log(
                "Panel", f"[{self._ts()}] Marker Stream terminal closed (code={code})\n"
            )
            self.marker_term = None

        self.marker_term.finished.connect(_marker_closed)

        self.marker_term.setProgram("gnome-terminal")
        self.marker_term.setArguments(
            [
                "--wait",
                "--",
                "bash",
                "-lc",
                f'source /home/alexandra-admin/opt/miniconda/etc/profile.d/conda.sh && conda activate mne && cd "{ROOT}" && python3 UTIL_marker_stream.py; exec bash',
            ]
        )
        self.marker_term.start()

    def on_marker_stop(self):
        if not self.marker_term:
            self._set_led(self.lbl_marker, "stopped")
            return
        if self.marker_term.state() != QProcess.NotRunning:
            self.marker_term.terminate()
            if not self.marker_term.waitForFinished(1500):
                self.marker_term.kill()
                self.marker_term.waitForFinished(1500)
        self._set_led(self.lbl_marker, "stopped")
        self._append_log("Panel", f"[{self._ts()}] Marker Stream stopped\n")

    def on_marker_refresh(self):
        self.on_marker_stop()
        time.sleep(0.5)
        self.on_marker_start()
        self._append_log("Panel", f"[{self._ts()}] Refreshed marker stream\n")

    # ----- Driver -----
    def on_driver_start(self):
        # Check if marker terminal is running
        if not self.marker_term or self.marker_term.state() == QProcess.NotRunning:
            QMessageBox.warning(self, "Gating", "Marker not ready. Start Marker first.")
            return

        self._start_proc(self.driver, self.lbl_driver, "Driver")

    def on_driver_stop(self):
        self._stop_proc(self.driver, self.lbl_driver, "Driver")

    # ----- External apps -----
    def on_open_labrec(self):
        # If already open, do nothing
        if self.labrec_term and self.labrec_term.state() != QProcess.NotRunning:
            return

        self.labrec_term = QProcess(self)
        # When the terminal starts, LED -> green
        self.labrec_term.started.connect(
            lambda: (
                self._set_led(self.lbl_labrec, "running"),
                self._append_log(
                    "Panel", f"[{self._ts()}] LabRecorder terminal opened\n"
                ),
            )
        )

        # When it closes, LED -> gray
        def _labrec_closed(code, status):
            self._set_led(self.lbl_labrec, "stopped")
            self._append_log(
                "Panel", f"[{self._ts()}] LabRecorder terminal closed (code={code})\n"
            )
            self.labrec_term = None

        self.labrec_term.finished.connect(_labrec_closed)

        self.labrec_term.setProgram("gnome-terminal")
        # --wait keeps this QProcess alive until the terminal tab/window exits
        self.labrec_term.setArguments(["--wait", "--", "bash", "-lc", "LabRecorder"])
        self.labrec_term.start()

    def on_open_eego(self):
        if self.eego_term and self.eego_term.state() != QProcess.NotRunning:
            return

        self.eego_term = QProcess(self)
        self.eego_term.started.connect(
            lambda: (
                self._set_led(self.lbl_eego, "running"),
                self._append_log(
                    "Panel", f"[{self._ts()}] eegoSports terminal opened\n"
                ),
            )
        )

        def _eego_closed(code, status):
            self._set_led(self.lbl_eego, "stopped")
            self._append_log(
                "Panel", f"[{self._ts()}] eegoSports terminal closed (code={code})\n"
            )
            self.eego_term = None

        self.eego_term.finished.connect(_eego_closed)

        self.eego_term.setProgram("gnome-terminal")
        self.eego_term.setArguments(
            [
                "--wait",
                "--",
                "bash",
                "-lc",
                "/home/alexandra-admin/opt/lsl-eego/eegoSports",
            ]
        )
        self.eego_term.start()

    # ---------- Process helpers ----------
    def _start_proc(self, p: Proc, led: QLabel, title: str):
        if p.cmd is None:
            QMessageBox.information(
                self, "Disabled", f"{p.name} is disabled for this mode."
            )
            return
        if p.q and p.q.state() != QProcess.NotRunning:
            return
        q = QProcess(self)
        parts = shlex.split(p.cmd)
        q.setProgram(parts[0])
        q.setArguments(parts[1:])
        q.setWorkingDirectory(p.cwd)
        # env
        env = os.environ.copy()
        env.update(p.env)
        from PySide6.QtCore import QProcessEnvironment

        qenv = QProcessEnvironment()
        for k, v in env.items():
            qenv.insert(k, v)
        q.setProcessEnvironment(qenv)
        # connect
        q.started.connect(lambda: self._on_started(p, led, title))
        q.finished.connect(
            lambda code, status: self._on_finished(p, led, title, code, status)
        )
        q.readyReadStandardOutput.connect(lambda: self._on_stdout(p, title))
        q.readyReadStandardError.connect(lambda: self._on_stderr(p, title))
        # go
        p.out.clear()
        p.err.clear()
        p.q = q
        p.status = "starting"
        self._set_led(led, "starting")
        q.start()

    def _stop_proc(self, p: Proc, led: QLabel, title: str):
        if not p.q:
            p.status = "stopped"
            self._set_led(led, "stopped")
            return
        if p.q.state() != QProcess.NotRunning:
            p.q.terminate()
            if not p.q.waitForFinished(1500):
                p.q.kill()
                p.q.waitForFinished(1500)
        p.status = "stopped"
        p.pid = None
        self._set_led(led, "stopped")
        self._append_log(title, f"[{self._ts()}] STOPPED\n")

    def _on_started(self, p: Proc, led: QLabel, title: str):
        p.status = "running"
        p.pid = p.q.processId()
        self._set_led(led, "running")
        self._append_log(title, f"[{self._ts()}] STARTED pid={p.pid} cmd={p.cmd}\n")

    def _on_finished(self, p: Proc, led: QLabel, title: str, code: int, status):
        p.pid = None
        p.status = "stopped" if code == 0 else "error"
        self._set_led(led, p.status)
        self._append_log(title, f"[{self._ts()}] FINISHED code={code}\n")

    def _on_stdout(self, p: Proc, title: str):
        data: QByteArray = p.q.readAllStandardOutput()
        p.out.extend(bytes(data))
        self._render_combined_log(title, p)

    def _on_stderr(self, p: Proc, title: str):
        data: QByteArray = p.q.readAllStandardError()
        p.err.extend(bytes(data))
        self._render_combined_log(title, p)

    # ---------- Log helpers ----------
    def _on_log_target_changed(self, target: str):
        self._current_log_target = target
        if getattr(self, "_building_ui", False):
            return
        self._refresh_log_view()

    def _refresh_log_view(self):
        if not hasattr(self, "txt_logs"):
            return
        buf = self._log_buffers.get(self._current_log_target, "")
        self.txt_logs.setPlainText(buf)
        self.txt_logs.moveCursor(QTextCursor.End)
        self.txt_logs.ensureCursorVisible()

    def _spawn_external(self, cmd: str):
        quoted = cmd.replace('"', r"\"")
        full = f'gnome-terminal -- bash -lc "{quoted}; exec bash"'
        subprocess.Popen(full, shell=True)

    def _append_log(self, title: str, text: str):
        key = title if title in self._log_buffers else "Panel"
        self._log_buffers[key] = (self._log_buffers.get(key, "") + text)[-2_000_000:]
        if self._current_log_target == key:
            self.txt_logs.moveCursor(QTextCursor.End)
            self.txt_logs.insertPlainText(text)
            self.txt_logs.moveCursor(QTextCursor.End)
            self.txt_logs.ensureCursorVisible()

    def _render_combined_log(self, title: str, p: Proc):
        combined = p.out + (b"\n[stderr]\n" + p.err if p.err else b"")
        if len(combined) > 2 * 1024 * 1024:
            combined = combined[-2 * 1024 * 1024 :]
        try:
            txt = combined.decode("utf-8", errors="replace")
        except Exception:
            txt = "<binary>\n"
        key = title if title in self._log_buffers else "Panel"
        self._log_buffers[key] = txt
        if self._current_log_target == key:
            self.txt_logs.setPlainText(txt)
            self.txt_logs.moveCursor(QTextCursor.End)
            self.txt_logs.ensureCursorVisible()

    def _append_udp_log(self, line: str):
        self.txt_udp_log.moveCursor(QTextCursor.End)
        self.txt_udp_log.insertPlainText(line + "\n")
        self.txt_udp_log.moveCursor(QTextCursor.End)
        self.txt_udp_log.ensureCursorVisible()

    @staticmethod
    def _ts() -> str:
        return time.strftime("%H:%M:%S")

    # ---------- Cheap LED maintainer for QProcess-procs ----------
    def _tick(self):
        # Keep QProcess-managed LEDs in sync (Marker, FES, Driver). Robot LEDs are manual.
        for p, led in ((self.driver, self.lbl_driver),):
            if p.q and p.q.state() != QProcess.NotRunning and p.status != "error":
                p.status = "running"
            if p.q:
                self._set_led(led, p.status)

    # ---------- Close cleanup ----------
    def closeEvent(self, event):
        # Try to stop local processes
        for p, led, title in ((self.driver, self.lbl_driver, "Driver"),):
            try:
                self._stop_proc(p, led, title)
            except Exception:
                pass
        # Robot terminal just closes with the app; no remote kill
        event.accept()


# ----------------- Entrypoint -----------------
def main():
    os.chdir(ROOT)
    app = QApplication(sys.argv)
    win = ControlPanel()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
