import cv2
import numpy as np
import matplotlib.pyplot as plt


FILE_PATH = 'Image.jpg' 
image_color_bgr = cv2.imread(FILE_PATH)
if image_color_bgr is None:

    raise FileNotFoundError(f"ERROR: File tidak ditemukan di {FILE_PATH} atau tidak dapat dibaca. Pastikan path dan nama file benar.")
image_color_rgb = cv2.cvtColor(image_color_bgr, cv2.COLOR_BGR2RGB)
image_gray = cv2.cvtColor(image_color_bgr, cv2.COLOR_BGR2GRAY)
print("Shape gambar:", image_gray.shape)


blur = cv2.GaussianBlur(image_gray, (5, 5), 0)
ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

kernel = np.ones((3,3), np.uint8) 

opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2) 
sure_bg = cv2.dilate(opening, kernel, iterations=3)
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)

ret, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1 
markers[unknown == 255] = 0 
markers_ws = cv2.watershed(image_color_bgr, markers)
image_color_bgr[markers_ws == -1] = [255, 255, 255] 


plt.figure(figsize=(15, 8))

titles = ["Citra Asli (Grayscale)", "Threshold (Otsu Inv)", "Distance Transform",
          "Sure Foreground (Marker)", "Sure Background", "Hasil Segmentasi Watershed (Batas Putih)"]
images = [image_gray, thresh, dist_transform, sure_fg, sure_bg, cv2.cvtColor(image_color_bgr, cv2.COLOR_BGR2RGB)]
cmaps = ['gray', 'gray', 'jet', 'gray', 'gray', None]

for i in range(len(images)):
    plt.subplot(2, 3, i + 1)
    plt.imshow(images[i], cmap=cmaps[i])
    plt.title(titles[i])
    plt.axis("off")

plt.tight_layout()
plt.show()
