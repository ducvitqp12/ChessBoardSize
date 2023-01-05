import numpy as np
import cv2
from matplotlib import pyplot as plt


def remove_sin_wave_noise(image):
    img_float32 = np.float32(image)  # chuyển ảnh về float để biến đổi Fourier

    # Biến đổi Fourier
    dft = np.fft.fft2(img_float32)
    dft_shift = np.fft.fftshift(dft)

    rows, cols = img.shape
    crow, ccol = rows / 2, cols / 2  # center của ảnh

    # Tạo ra mask để filter trên miền tần số
    mask = np.ones((rows, cols), np.uint8)
    mask[int(crow) - 1:int(crow) + 1, :] = 0  # Xóa phần tương ứng với sóng sin

    # Biến đổi Fourier ngược
    fshift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)

    img_backCopy = np.real(img_back)

    img_backCopy = np.uint8(img_backCopy)  # chuyển từ float thành int

    return img_backCopy


def remove_noise_and_smooth(image):
    median = cv2.medianBlur(image, 5)  # Xóa nhiễu hạt tiêu

    blur = cv2.GaussianBlur(median, (5, 5), 0)  # Làm mượt ảnh

    im = (cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))).apply(blur)  # Cân bằng histogram tăng độ tương phản

    ret, thresh = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # Threshold về binary image

    return thresh


def line_detection(image):
    # Edge detection
    edges = cv2.Canny(image, 50, 150, apertureSize=3)

    # Apply HoughLinesP method to
    # to directly obtain line end points
    lines_list = []
    imm = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)

    # Line Detection
    lines = cv2.HoughLinesP(
        edges,  # Input edge image
        2,  # Distance resolution in pixels
        np.pi / 180,  # Angle resolution in radians
        threshold=100,  # Min number of votes for valid line
        minLineLength=5,  # Min allowed length of line
        maxLineGap=100  # Max allowed gap between line for joining them
    )
    #
    # # Iterate over points
    for points in lines:
        # Extracted points nested in the list
        x1, y1, x2, y2 = points[0]
        # Maintain a simples lookup list for points
        lines_list.append([(x1, y1), (x2, y2)])

    return lines_list


if __name__ == '__main__':
    # Đọc hình ảnh ở dạng xám
    img = cv2.imread('image/Chessboard0631.png', 0)
    # Xóa nhiễu sóng sin
    img = remove_sin_wave_noise(img)
    # Xóa nhiễu hạt tiêu, làm mượt    img = remove_noise_and_smooth(img)
    # Line detect
    lines = line_detection(img)

    # Visualize
    imm = np.zeros((img.shape[0], img.shape[1], 3))
    for line in lines:
        # Extracted points nested in the list
        x1, y1 = line[0]
        x2, y2 = line[1]
        # Draw the lines joing the points
        # On the original image
        cv2.line(imm, (x1, y1), (x2, y2), (0, 255, 255), 2)

    plt.imshow(imm, cmap="gray")

    plt.show()
