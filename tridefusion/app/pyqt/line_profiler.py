import csv
import json
import sys
from pathlib import Path
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor
from PyQt5 import QtWidgets, QtCore
from .utils import list_3d_tiff_files, to_grayscale

def compute_fwhm(profile):
    """
    Compute Full Width at Half Maximum (FWHM) for 1D intensity profile.
    Returns FWHM width in index units. If cannot compute, returns None.
    """
    if len(profile) < 3:
        return None

    profile = np.array(profile, dtype=float)
    peak_idx = np.argmax(profile)
    peak_val = profile[peak_idx]
    half_val = peak_val / 2.0

    left = None
    for i in range(peak_idx, 0, -1):
        if profile[i] < half_val:
            left = i
            break

    right = None
    for i in range(peak_idx, len(profile)):
        if profile[i] < half_val:
            right = i
            break

    if left is None or right is None:
        return None
    return right - left


class LineProfiler(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D Image Line Profiler")
        self.resize(600, 350)

        layout = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QFormLayout()

        # Folder input
        self.folder_edit = QtWidgets.QLineEdit()
        folder_btn = QtWidgets.QPushButton("Browse...")
        folder_btn.clicked.connect(self.browse_folder)
        h = QtWidgets.QHBoxLayout()
        h.addWidget(self.folder_edit)
        h.addWidget(folder_btn)
        form.addRow("Folder path:", h)

        # Reference file
        self.ref_edit = QtWidgets.QLineEdit()
        ref_btn = QtWidgets.QPushButton("Browse...")
        ref_btn.clicked.connect(self.browse_reference)
        h_ref = QtWidgets.QHBoxLayout()
        h_ref.addWidget(self.ref_edit)
        h_ref.addWidget(ref_btn)
        form.addRow("Noisy path (optional):", h_ref)

        # Layer number
        self.layer_spin = QtWidgets.QSpinBox()
        self.layer_spin.setRange(0, 10000)
        form.addRow("Layer number (z):", self.layer_spin)

        # Refinement
        self.refine_spin = QtWidgets.QSpinBox()
        self.refine_spin.setRange(1, 20)
        self.refine_spin.setValue(1)
        form.addRow("Refine / smooth factor:", self.refine_spin)

        # Mode selection
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItems(["Manual (click 2 points)", "Automatic (middle line)"])
        form.addRow("Mode:", self.mode_combo)

        # Pixel→µm resolution
        self.pixel_size_spin = QtWidgets.QDoubleSpinBox()
        self.pixel_size_spin.setRange(0.01, 10000)
        self.pixel_size_spin.setValue(0.22)  # default example
        self.pixel_size_spin.setSuffix(" µm")
        form.addRow("Pixel size:", self.pixel_size_spin)

        layout.addLayout(form)

        self.plot_btn = QtWidgets.QPushButton("Run Line Profiling")
        self.plot_btn.clicked.connect(self.plot_profiles)
        layout.addWidget(self.plot_btn)

        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        layout.addWidget(self.log)

        self.click_points = []

    def browse_folder(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            self.folder_edit.setText(folder)

    def browse_reference(self):
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Noisy TIFF", "", "TIFF files (*.tif *.tiff)")
        if fname:
            self.ref_edit.setText(fname)

    def append_log(self, msg):
        self.log.appendPlainText(msg)

    def plot_profiles(self):
        folder = Path(self.folder_edit.text())
        if not folder.exists():
            self.append_log("Invalid folder path.")
            return

        files = list_3d_tiff_files(folder)
        if not files:
            self.append_log("No TIFF files found.")
            return

        layer = self.layer_spin.value()
        refine = self.refine_spin.value()
        ref_path = self.ref_edit.text()
        reference = None
        if ref_path:
            reference = tiff.imread(ref_path)
            if reference.ndim == 4 and reference.shape[0] == 1:
                reference = reference[0]

        arr = tiff.imread(str(files[0]))
        if arr.ndim == 4 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim == 4 and arr.shape[-1] in [3, 4]:
            arr = np.stack([to_grayscale(a) for a in arr], axis=0)

        if arr.ndim != 3:
            arr = to_grayscale(arr)
            arr = np.expand_dims(arr, axis=0)

        if layer >= arr.shape[0]:
            self.append_log("Layer index out of bounds.")
            return

        layer_img = to_grayscale(arr[layer])
        mode = self.mode_combo.currentText()

        if mode.startswith("Manual"):
            self.manual_mode(files, layer_img, layer, refine, reference)
        else:
            self.auto_mode(files, layer_img, layer, refine, reference)

    def manual_mode(self, files, layer_img, layer, refine, reference):
        self.append_log("Manual mode: Click two points to define a line.")
        self.click_points = []

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(layer_img, cmap="gray")
        ax.set_title("Click 2 points to define line")

        cursor = Cursor(ax, useblit=True, color='red', linewidth=1)

        def onclick(event):
            if event.inaxes != ax:
                return
            self.click_points.append((int(event.xdata), int(event.ydata)))
            ax.plot(event.xdata, event.ydata, 'ro')
            fig.canvas.draw()

            if len(self.click_points) == 2:
                plt.close(fig)
                self.append_log(f"Selected points: {self.click_points}")
                self.extract_and_plot(files, layer, refine, reference, [self.click_points])

        fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show()

    def auto_mode(self, files, layer_img, layer, refine, reference):
        h, w = layer_img.shape
        points = [(w * 0.25, h // 2), (w * 0.75, h // 2)]
        self.append_log(f"Automatic mode: middle horizontal line: {points}")
        self.extract_and_plot(files, layer, refine, reference, [points])

    def extract_and_plot(self, files, layer, refine, reference, points_list):
        output_dir = Path(self.folder_edit.text()) / "line_profiles_results"
        output_dir.mkdir(exist_ok=True)

        pixel_size_mm = self.pixel_size_spin.value()

        for idx, points in enumerate(points_list):
            x1, y1 = map(float, points[0])
            x2, y2 = map(float, points[1])

            # Store all line profiles here
            all_lines = {}

            plt.figure(figsize=(10, 6))
            plt.title(f"Line profile #{idx+1} (Layer {layer})")

            for fpath in files:
                arr = tiff.imread(str(fpath))
                if arr.ndim == 4 and arr.shape[0] == 1:
                    arr = arr[0]
                if arr.ndim == 4 and arr.shape[-1] in [3, 4]:
                    arr = np.stack([to_grayscale(a) for a in arr], axis=0)

                img = arr[layer] if arr.ndim == 3 else arr
                img = to_grayscale(img)

                N = int(np.hypot(x2 - x1, y2 - y1))
                xs = np.linspace(x1, x2, N)
                ys = np.linspace(y1, y2, N)
                profile = img[ys.astype(int), xs.astype(int)]

                if refine > 1:
                    profile = np.convolve(profile, np.ones(refine)/refine, mode='same')

                # ---- NEW: peak detection ----
                peak_idx = int(np.argmax(profile))
                peak_val = float(profile[peak_idx])

                # ---- FWHM ----
                fwhm_pixels = compute_fwhm(profile)
                fwhm_mm = float(fwhm_pixels * pixel_size_mm) if fwhm_pixels else None

                distances_mm = np.arange(len(profile)) * pixel_size_mm
                plt.plot(distances_mm, profile, label=fpath.name)

                # Log
                self.append_log(
                    f"{fpath.name}: peak={peak_val:.2f}, FWHM={fwhm_mm:.2f} µm" if fwhm_mm else
                    f"{fpath.name}: peak={peak_val:.2f}, FWHM=None"
                )

                # Save
                method_name = fpath.stem  # remove .tif / .tiff
                all_lines[method_name] = {
                    "intensity": profile.tolist(),
                    "peak_value": peak_val,
                    "peak_index": peak_idx,
                    "fwhm_mu_m": fwhm_mm
                }

            # ---------- REFERENCE ----------
            if reference is not None:
                img = to_grayscale(reference[layer])

                N = int(np.hypot(x1 - x2, y1 - y2))
                xs = np.linspace(x1, x2, N)
                ys = np.linspace(y1, y2, N)
                ref_profile = img[ys.astype(int), xs.astype(int)]

                if refine > 1:
                    ref_profile = np.convolve(ref_profile, np.ones(refine)/refine, mode='same')

                peak_idx = int(np.argmax(ref_profile))
                peak_val = float(ref_profile[peak_idx])

                fwhm_pixels = compute_fwhm(ref_profile)
                fwhm_mm = float(fwhm_pixels * pixel_size_mm) if fwhm_pixels else None

                distances_mm = np.arange(len(ref_profile)) * pixel_size_mm
                plt.plot(distances_mm, ref_profile, '--', linewidth=2, label="Noisy")

                # Save reference results
                all_lines["Noisy"] = {
                    "intensity": ref_profile.tolist(),
                    "peak_value": peak_val,
                    "peak_index": peak_idx,
                    "fwhm_mu_m": fwhm_mm
                }

                self.append_log(
                    f"Noisy: peak={peak_val:.2f}, FWHM={fwhm_mm:.2f} µm" if fwhm_mm else
                    f"Noisy: peak={peak_val:.2f}, FWHM=None"
                )

            # Save image figure
            fig_path = output_dir / f"line_profile_{idx+1}.png"
            plt.xlabel("Distance (µm)")
            plt.ylabel("Intensity")
            plt.legend()
            plt.savefig(fig_path)
            plt.close()

            # Save all profiles (NEW: all files saved, not overwritten)
            json_path = output_dir / f"line_profile_{idx+1}.json"
            with open(json_path, "w") as f:
                json.dump(all_lines, f, indent=2)

            summary_csv = output_dir / "line_profiles_summary.csv"
            write_header = not summary_csv.exists()

            with open(summary_csv, "a", newline="") as csvfile:
                writer = csv.writer(csvfile)

                if write_header:
                    writer.writerow([
                        "profile_index",
                        "filename",
                        "peak_value",
                        "peak_index",
                        "fwhm_pixels",
                        "fwhm_um"
                    ])

                for fname, vals in all_lines.items():
                    method_name = Path(fname).stem  # ensure no .tif

                    intens = np.array(vals["intensity"], dtype=float)
                    peak_index = vals["peak_index"]
                    peak_value = vals["peak_value"]

                    fwhm_um = vals["fwhm_mu_m"]
                    fwhm_pixels = fwhm_um / pixel_size_mm if fwhm_um is not None else None

                    writer.writerow([
                        idx + 1,
                        method_name,
                        peak_value,
                        peak_index,
                        fwhm_pixels,
                        fwhm_um
                    ])
            self.append_log(f"Saved profile #{idx+1} to {output_dir}")



def main():
    app = QtWidgets.QApplication(sys.argv)
    w = LineProfiler()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
