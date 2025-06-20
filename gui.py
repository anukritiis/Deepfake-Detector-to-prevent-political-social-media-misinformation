import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path

ModelsLocation = Path(__file__).parent
models = {
    "Emboss":    ModelsLocation / "Models/OneOfTheBestEmboss.keras",
    "Laplacian": ModelsLocation / "Models/OneOfTheBestlaplacian.keras",
    "LEFFT":     ModelsLocation / "Models/OneOfTheBestLEFFT.keras",
    "CNN_Full":  ModelsLocation / "Models/custom_cnn_full.keras",
}

def onlyembossing(img):
    gray   = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.array([[-2,-1,0],[-1,1,1],[0,1,2]], np.float32)
    emb    = cv2.filter2D(gray, ddepth=-1, kernel=kernel)
    return cv2.cvtColor(emb, cv2.COLOR_GRAY2BGR)

def onlylaplacian(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap  = cv2.Laplacian(gray, cv2.CV_64F)
    lap8 = cv2.convertScaleAbs(lap)
    return cv2.cvtColor(lap8, cv2.COLOR_GRAY2BGR)

def onlyfft(img):
    gray   = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    f      = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    mag    = 20 * np.log(np.abs(fshift) + 1)
    norm   = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    mag8   = norm.astype(np.uint8)
    return cv2.cvtColor(mag8, cv2.COLOR_GRAY2BGR)

def alltechniques(img):
    e = onlyembossing(img)
    l = onlylaplacian(e)
    return onlyfft(l)

STAGES = [
    ("Emboss",    models["Emboss"],    onlyembossing),
    ("Laplacian", models["Laplacian"], onlylaplacian),
    ("LEFFT",     models["LEFFT"],     alltechniques),
    ("CNN_Full",  models["CNN_Full"],  alltechniques),
]

def classify_single(ImageLocation: Path):
    StartingImage = cv2.imread(str(ImageLocation))
    if StartingImage is None:
        raise FileNotFoundError(f"Could not load {ImageLocation}")

    coloumns = len(STAGES) + 1
    fig, axes = plt.subplots(1, coloumns, figsize=(4*coloumns, 4))

    axes[0].imshow(cv2.cvtColor(StartingImage, cv2.COLOR_BGR2RGB))
    axes[0].axis("off")
    axes[0].set_title("Original")

    results = []

    for ax, (name, mpath, fn) in zip(axes[1:], STAGES):
        ProcessBGRImage = fn(StartingImage.copy())

        if not mpath.exists():
            raise FileNotFoundError(f"Model file not found: {mpath}")

        model = tf.keras.models.load_model(str(mpath))
        _, H, W, C = model.input_shape
        if C == 1:
            GrayImage = cv2.cvtColor(ProcessBGRImage, cv2.COLOR_BGR2GRAY)
            ProcessForModel = GrayImage[..., None]
        else:
            ProcessForModel = ProcessBGRImage

        x = cv2.resize(ProcessForModel, (W, H)).astype(np.float32) / 255.0
        x = np.expand_dims(x, 0)

        prob = float(model.predict(x, verbose=0)[0][0])
        idx  = int(prob > 0.5)
        label = "Fake" if idx else "Real"
        conf  = prob if idx else 1 - prob

        ax.imshow(cv2.cvtColor(ProcessBGRImage, cv2.COLOR_BGR2RGB))
        ax.axis("off")
        ax.set_title(f"{name}\n{label} ({conf:.1%})",
                     color=("red" if idx else "green"))

        results.append((name, label, conf))

    plt.tight_layout()
    plt.show()
    return results

def select_image():
    file_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.gif")]
    )
    if not file_path:
        return
    image_label.config(text=f"Selected: {file_path}")
    try:
        results = classify_single(Path(file_path))
        result_str = "\n".join([f"{name}: {label} ({conf:.2%})" for name, label, conf in results])
        messagebox.showinfo("Classification Results", result_str)
    except Exception as e:
        messagebox.showerror("Error", str(e))

root = tk.Tk()
root.title("Deepfake Detector GUI")

tk.Label(root, text="Deepfake Detector", font=("Helvetica", 16, "bold")).pack(pady=10)
tk.Button(root, text="Select Image", command=select_image, width=30, height=2).pack(pady=20)
image_label = tk.Label(root, text="No image selected", fg="gray")
image_label.pack(pady=10)

root.mainloop()
