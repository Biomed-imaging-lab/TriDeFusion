import sys
import os
import json
import random
import traceback
from pathlib import Path
from typing import List, Tuple

import numpy as np
import tifffile as tiff
from PyQt5 import QtWidgets, QtCore, QtGui
from .utils import list_3d_tiff_files, to_grayscale


def sample_boxes_for_volume(shape: Tuple[int, int, int], box_size: int, count: int, rng: random.Random) -> List[Tuple[int, int, int]]:
    z_dim, y_dim, x_dim = shape
    if z_dim < box_size or y_dim < box_size or x_dim < box_size:
        return []
    z_max = z_dim - box_size
    y_max = y_dim - box_size
    x_max = x_dim - box_size
    boxes = []
    while len(boxes) < count:
        boxes.append((rng.randint(0, z_max), rng.randint(0, y_max), rng.randint(0, x_max)))
    return boxes


class WorkerSignals(QtCore.QObject):
    finished = QtCore.pyqtSignal()
    progress = QtCore.pyqtSignal(int)
    log = QtCore.pyqtSignal(str)


class ExtractWorker(QtCore.QRunnable):
    def __init__(self, files: List[Path], box_size: int, count: int, seed: int, out_root: Path):
        super().__init__()
        self.files = files
        self.box_size = box_size
        self.count = count
        self.seed = seed
        self.out_root = out_root
        self.signals = WorkerSignals()

    @QtCore.pyqtSlot()
    def run(self):
        try:
            rng = random.Random(self.seed) if self.seed is not None else random.Random()
            total = len(self.files)
            for idx, fpath in enumerate(self.files, start=1):
                try:
                    self.signals.log.emit(f"Reading: {fpath}")
                    arr = tiff.imread(str(fpath))
                    arr = to_grayscale(arr)
                    if arr.ndim == 4 and arr.shape[0] == 1:
                        arr = arr[0]
                    if arr.ndim != 3:
                        self.signals.log.emit(f"Skipping (unsupported shape): {arr.shape}")
                        continue
                    z_dim, y_dim, x_dim = arr.shape
                    boxes = sample_boxes_for_volume((z_dim, y_dim, x_dim), self.box_size, self.count, rng)
                    if not boxes:
                        self.signals.log.emit(f"Too small: {arr.shape}")
                        continue

                    out_dir = self.out_root / fpath.stem
                    out_dir.mkdir(parents=True, exist_ok=True)
                    params = []
                    z0 = z_dim // 2
                    for i, (_, y0, x0) in enumerate(boxes, start=1):
                        z1, y1, x1 = z0 + self.box_size, y0 + self.box_size, x0 + self.box_size
                        seg = arr[z0:z1, y0:y1, x0:x1]
                        mean_int = float(np.mean(seg))
                        fname = out_dir / f"segment_{i:03d}.tif"
                        tiff.imwrite(str(fname), seg)
                        params.append(dict(index=i, z0=z0, y0=y0, x0=x0, size=self.box_size,
                                           mean_intensity_gt=mean_int, out_file=str(fname.name)))
                        self.signals.log.emit(f"Saved {fname.name}")
                    with open(out_dir / "params.json", "w") as f:
                        json.dump(params, f, indent=2)
                except Exception as e:
                    self.signals.log.emit(f"Error: {e}\n{traceback.format_exc()}")
                self.signals.progress.emit(int(idx / total * 100))
        finally:
            self.signals.finished.emit()

class BackgroundExtractWorker(QtCore.QRunnable):
    """
    Extract random background segments with very low mean intensity from 3D volumes.
    """
    def __init__(self, files: List[Path], box_size: int, count: int, seed: int, out_root: Path, threshold: float = 1e-4):
        super().__init__()
        self.files = files
        self.box_size = box_size
        self.count = count
        self.seed = seed
        self.out_root = out_root
        self.threshold = threshold
        self.signals = WorkerSignals()

    @QtCore.pyqtSlot()
    def run(self):
        try:
            rng = random.Random(self.seed) if self.seed is not None else random.Random()
            total = len(self.files)
            for idx, fpath in enumerate(self.files, start=1):
                try:
                    self.signals.log.emit(f"Reading: {fpath}")
                    arr = tiff.imread(str(fpath)).astype(np.float32)
                    if arr.ndim == 4 and arr.shape[0] == 1:
                        arr = arr[0]
                    if arr.ndim != 3:
                        self.signals.log.emit(f"Skipping (unsupported shape): {arr.shape}")
                        continue

                    z_dim, y_dim, x_dim = arr.shape
                    boxes = []
                    attempts = 0
                    max_attempts = self.count * 10  # avoid infinite loop
                    if y_dim < self.box_size or x_dim < self.box_size:
                        raise ValueError(f"Image too small for box size {self.box_size}: {arr.shape}")

                    

                    while len(boxes) < self.count and attempts < max_attempts:
                        y_max = y_dim - self.box_size
                        x_max = x_dim - self.box_size
                        y0 = rng.randint(0, y_max)
                        x0 = rng.randint(0, x_max)
                        patch = arr[
                            y0:y0 + self.box_size,
                            x0:x0 + self.box_size
                        ]
                        # seg = arr[z0:z0+self.box_size, y0:y0+self.box_size, x0:x0+self.box_size]
                        if float(np.mean(patch)) <= self.threshold:
                            boxes.append((z0, y0, x0))
                        attempts += 1

                    if not boxes:
                        self.signals.log.emit("No background segments found")
                        continue

                    out_dir = self.out_root / fpath.stem
                    out_dir.mkdir(parents=True, exist_ok=True)
                    params = []
                    for i, (z0, y0, x0) in enumerate(boxes, start=1):
                        seg = arr[z0:z0+self.box_size, y0:y0+self.box_size, x0:x0+self.box_size]
                        mean_int = float(np.mean(seg))
                        seg_uint8 = ((seg - seg.min()) / (seg.max() - seg.min() + 1e-8) * 255).clip(0, 255).astype(np.uint8)
                        fname = out_dir / f"segment_{i:03d}.tif"
                        tiff.imwrite(str(fname), seg_uint8)
                        params.append(dict(index=i, z0=z0, y0=y0, x0=x0, size=self.box_size,
                                           mean_intensity_gt=mean_int, out_file=str(fname.name)))
                        self.signals.log.emit(f"Saved {fname.name}")

                    with open(out_dir / "params.json", "w") as f:
                        json.dump(params, f, indent=2)

                except Exception as e:
                    self.signals.log.emit(f"Error: {e}\n{traceback.format_exc()}")
                self.signals.progress.emit(int(idx / total * 100))
        finally:
            self.signals.finished.emit()


class VolumeViewer(QtWidgets.QDialog):
    """Dialog to visualize 3D image slices and manually extract a segment."""
    def __init__(self, vol: np.ndarray, out_dir: Path, parent=None):
        super().__init__(parent)
        self.vol = vol
        self.out_dir = out_dir
        self.setWindowTitle("Manual Segment Extractor & Viewer")
        self.resize(700, 600)

        layout = QtWidgets.QVBoxLayout(self)

        self.image_label = QtWidgets.QLabel()
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(self.image_label, stretch=1)

        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setRange(0, vol.shape[0]-1)
        self.slider.valueChanged.connect(self.update_slice)
        layout.addWidget(self.slider)

        coord_layout = QtWidgets.QFormLayout()
        self.z_edit = QtWidgets.QLineEdit("0")
        self.y_edit = QtWidgets.QLineEdit("0")
        self.x_edit = QtWidgets.QLineEdit("0")
        self.size_edit = QtWidgets.QLineEdit("64")
        coord_layout.addRow("z0:", self.z_edit)
        coord_layout.addRow("y0:", self.y_edit)
        coord_layout.addRow("x0:", self.x_edit)
        coord_layout.addRow("size:", self.size_edit)
        layout.addLayout(coord_layout)

        btns = QtWidgets.QHBoxLayout()
        extract_btn = QtWidgets.QPushButton("Extract segment")
        extract_btn.clicked.connect(self.extract_segment)
        btns.addWidget(extract_btn)
        layout.addLayout(btns)

        self.update_slice(0)

    def update_slice(self, idx):
        """Display normalized 2D slice safely for float32 volumes."""
        slice_img = np.asarray(self.vol[idx], dtype=np.float32)
        if np.isnan(slice_img).any():
            slice_img = np.nan_to_num(slice_img, nan=0.0)
        vmin = float(np.min(slice_img))
        vmax = float(np.max(slice_img))
        vrange = vmax - vmin if vmax > vmin else 1e-5
        norm_img = ((slice_img - vmin) / vrange * 255.0).clip(0, 255).astype(np.uint8)
        qimg = QtGui.QImage(
            norm_img.data,
            norm_img.shape[1],
            norm_img.shape[0],
            norm_img.strides[0],
            QtGui.QImage.Format_Grayscale8,
        )
        pix = QtGui.QPixmap.fromImage(qimg).scaled(
            self.image_label.width(),
            self.image_label.height(),
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation,
        )
        self.image_label.setPixmap(pix)

    def extract_segment(self):
        try:
            z0, y0, x0 = int(self.z_edit.text()), int(self.y_edit.text()), int(self.x_edit.text())
            size = int(self.size_edit.text())
            z1, y1, x1 = z0 + size, y0 + size, x0 + size
            seg = self.vol[z0:z1, y0:y1, x0:x1]
            mean_int = float(np.mean(seg))
            seg_min, seg_max = seg.min(), seg.max()
            seg_uint8 = ((seg - seg_min) / (seg_max - seg_min + 1e-8) * 255.0).clip(0, 255).astype(np.uint8)
            fname = self.out_dir / f"manual_segment_z{z0}_y{y0}_x{x0}.tif"
            tiff.imwrite(str(fname), seg_uint8)
            QtWidgets.QMessageBox.information(
                self, "Saved",
                f"Segment saved:\n{fname}\nMean (float32)={mean_int:.4f}"
            )
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Error", str(e))


class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D Segment Extractor")
        self.resize(750, 500)

        layout = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QFormLayout()

        self.input_line = QtWidgets.QLineEdit()
        browse_btn = QtWidgets.QPushButton("Browse...")
        browse_btn.clicked.connect(self.browse_folder)
        h = QtWidgets.QHBoxLayout()
        h.addWidget(self.input_line)
        h.addWidget(browse_btn)
        form.addRow("Input folder:", h)

        self.box_spin = QtWidgets.QSpinBox()
        self.box_spin.setRange(4, 2048)
        self.box_spin.setValue(64)
        form.addRow("Box size:", self.box_spin)

        self.count_spin = QtWidgets.QSpinBox()
        self.count_spin.setRange(1, 1000)
        self.count_spin.setValue(10)
        form.addRow("Segments/image:", self.count_spin)

        self.seed_edit = QtWidgets.QLineEdit()
        form.addRow("Random seed:", self.seed_edit)
        layout.addLayout(form)

        btn_layout = QtWidgets.QHBoxLayout()
        self.start_btn = QtWidgets.QPushButton("Start Random Extraction")
        self.start_btn.clicked.connect(self.start_extraction)
        self.manual_btn = QtWidgets.QPushButton("Manual Extract")
        self.manual_btn.clicked.connect(self.open_manual_viewer)
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.manual_btn)
        layout.addLayout(btn_layout)

        self.bg_btn = QtWidgets.QPushButton("Extract Background Segments")
        self.bg_btn.clicked.connect(self.start_background_extraction)
        btn_layout.addWidget(self.bg_btn)

        self.progress = QtWidgets.QProgressBar()
        layout.addWidget(self.progress)

        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        layout.addWidget(self.log)

        self.threadpool = QtCore.QThreadPool()

    def browse_folder(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select input folder")
        if folder:
            self.input_line.setText(folder)

    def append_log(self, msg): self.log.appendPlainText(msg)
    def set_progress(self, v): self.progress.setValue(v)

    def start_extraction(self):
        folder = Path(self.input_line.text())
        if not folder.exists():
            QtWidgets.QMessageBox.warning(self, "Error", "Invalid folder")
            return
        files = list_3d_tiff_files(folder)
        if not files:
            QtWidgets.QMessageBox.warning(self, "Error", "No TIFF files found")
            return
        seed = int(self.seed_edit.text()) if self.seed_edit.text().strip() else None
        out_root = folder / "extracted_segments"
        out_root.mkdir(exist_ok=True)
        worker = ExtractWorker(files, self.box_spin.value(), self.count_spin.value(), seed, out_root)
        worker.signals.log.connect(self.append_log)
        worker.signals.progress.connect(self.set_progress)
        worker.signals.finished.connect(lambda: self.append_log("Done."))
        self.threadpool.start(worker)
        self.append_log("Started random extraction...")

    def start_background_extraction(self):
        folder = Path(self.input_line.text())
        if not folder.exists():
            QtWidgets.QMessageBox.warning(self, "Error", "Invalid folder")
            return
        files = list_3d_tiff_files(folder)
        if not files:
            QtWidgets.QMessageBox.warning(self, "Error", "No TIFF files found")
            return
        seed = int(self.seed_edit.text()) if self.seed_edit.text().strip() else None
        out_root = folder / "background_segments"
        out_root.mkdir(exist_ok=True)
        worker = BackgroundExtractWorker(files, self.box_spin.value(), self.count_spin.value(), seed, out_root)
        worker.signals.log.connect(self.append_log)
        worker.signals.progress.connect(self.set_progress)
        worker.signals.finished.connect(lambda: self.append_log("Background extraction done."))
        self.threadpool.start(worker)
        self.append_log("Started background extraction...")


    def open_manual_viewer(self):
        folder = Path(self.input_line.text())
        if not folder.exists():
            QtWidgets.QMessageBox.warning(self, "Error", "Select a folder first")
            return
        files = list_3d_tiff_files(folder)
        if not files:
            QtWidgets.QMessageBox.warning(self, "Error", "No TIFF files found")
            return
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select 3D TIFF", str(folder), "TIFF files (*.tif *.tiff)")
        if not fname:
            return
        arr = tiff.imread(fname)
        if arr.ndim == 4 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim != 3:
            QtWidgets.QMessageBox.warning(self, "Error", f"Unsupported shape: {arr.shape}")
            return
        out_dir = folder / "extracted_segments"
        out_dir.mkdir(exist_ok=True)
        viewer = VolumeViewer(arr, out_dir, self)
        viewer.exec_()


def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
