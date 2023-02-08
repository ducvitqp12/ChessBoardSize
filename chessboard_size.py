import numpy as np
import cv2
from matplotlib import pyplot as plt
import math
from scipy.ndimage.measurements import label


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

    # cv2.imshow("im",img_backCopy )
    # cv2.waitKey(0)

    return img_backCopy


def remove_noise_and_smooth(image):
    median = cv2.medianBlur(image, 5)  # Xóa nhiễu hạt tiêu

    blur = cv2.GaussianBlur(median, (5, 5), 0)  # Làm mượt ảnh

    im = (cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))).apply(blur)  # Cân bằng histogram tăng độ tương phản

    # cv2.imshow("thres", im)
    #
    # cv2.waitKey(0)

    ret, thresh = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # Threshold về binary image

    print(thresh.shape)

    kernel = np.ones((5, 5), np.uint8)
    #
    img_erosion = cv2.erode(thresh, kernel, iterations=1)
    # img_dilation = cv2.dilate(img_erosion, kernel, iterations=1)
    # img_dilation = cv2.dilate(thresh, kernel, iterations=1)

    labelsAry, nfeatures = label(img_erosion)

    bboxes = []
    weight = []
    index = 0
    for i in range(1, nfeatures + 1):
        arr = labelsAry == i
        weight.append(np.sum(arr))

    # max_value = max(weight)
    # weight.remove(max_value)

    mean = np.mean(weight)
    std = np.std(weight)

    print("%f %f" %(mean, std))
    threshold = std/2

    res = np.zeros(labelsAry.shape, dtype=img_erosion.dtype)

    weight_0 = []

    for i in range(1, nfeatures + 1):
        arr = labelsAry == i
        if abs(np.sum(arr) - mean) <= threshold:
    #         weight_0.append(np.sum(arr))
    #
    # mean = np.mean(weight_0)
    # std = np.std(weight_0)
    # threshold = std
    # for i in range(1, nfeatures + 1):
    #     arr = labelsAry == i
    #     if abs(np.sum(arr) - mean) <= threshold:
            res += arr.astype(np.uint8)*255

    # cv2.imshow("thress", img_erosion)
    #
    # cv2.waitKey(0)

    cv2.imshow("thress", res)

    cv2.waitKey(0)

    return cv2.dilate(res, kernel, iterations=1)


def line_detection(image):
    # Edge detection
    edges = cv2.Canny(image, 50, 150, apertureSize=3)

    # cv2.imshow("thress", edges)
    #
    # cv2.waitKey(0)
    # Apply HoughLinesP method to
    # to directly obtain line end points
    lines_list = []

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


def remove_duplicate(array):
    result = [array[0]]
    for ele in array:
        check = True
        for res in result:
            if np.abs(ele[1] - res[1]) < 10:
                check = False
                break
        if check:
            result.append(ele)

    return result


def remove_outlier(array):
    angle = []
    for ele in array:
        angle.append(ele[0])
    mean = np.mean(angle)
    std = np.std(angle)

    threshold = 2 * std

    outliers_removed = [x for x in array if abs(x[0] - mean) <= threshold]

    return outliers_removed


if __name__ == '__main__':
    # Đọc hình ảnh ở dạng xám
    img = cv2.imread('image/Chessboard_0481.png', 0)
    # Xóa nhiễu sóng sin
    img = remove_sin_wave_noise(img)
    # Xóa nhiễu hạt tiêu, làm mượt    
    img = remove_noise_and_smooth(img)
    # Line detect
    lines = line_detection(img)

    array_object = []

    # Visualize
    imm = np.zeros((img.shape[0], img.shape[1], 3))

    max_angle = 0
    min_angle = 90
    for line in lines:
        # Extracted points nested in the list
        x1, y1 = line[0]
        x2, y2 = line[1]
        # Draw the lines joing the points
        # On the original image
        Ox = [1, 0]
        Oy = [0, 1]
        vector = [x2 - x1, y2 - y1]

        unit_vector_1 = Ox / np.linalg.norm(Ox)
        unit_vector_2 = vector / np.linalg.norm(vector)
        dot_product = np.dot(unit_vector_1, unit_vector_2)
        angle = np.arccos(dot_product) if np.arccos(dot_product) <= math.pi / 2 else math.pi - np.arccos(dot_product)
        angle_deg = angle * 180 / math.pi
        # print(angle_deg)
        if max_angle < angle_deg:
            max_angle = angle_deg
        if min_angle > angle_deg:
            min_angle = angle_deg

        # find distance
        p1 = np.array([x1, y1])
        p2 = np.array([x2, y2])
        p3 = np.array([0, 0])
        distance = np.linalg.norm(np.cross(p2 - p1, p1 - p3)) / np.linalg.norm(p2 - p1)
        array_object.append([angle_deg, distance, line])
        # print(angle * 180 / math.pi, end='\n')

    vertical = []
    horizontal = []
    print(array_object[0])
    for obj in array_object:
        if obj[0] - min_angle < max_angle // 2:
            horizontal.append(obj)
        elif obj[0] - min_angle > max_angle // 2:
            vertical.append(obj)

    # filter duplicate line

    verLine = remove_duplicate(vertical)
    verLine = remove_outlier(verLine)

    for line in verLine:
        x1, y1 = line[2][0]
        x2, y2 = line[2][1]
        cv2.line(imm, (x1, y1), (x2, y2), (0, 255, 255), 2)

    horLine = remove_duplicate(horizontal)
    horLine = remove_outlier(horLine)

    for line in horLine:
        x1, y1 = line[2][0]
        x2, y2 = line[2][1]
        cv2.line(imm, (x1, y1), (x2, y2), (0, 255, 255), 2)

    print(len(verLine), end='\n')
    print(len(horLine))

    plt.imshow(imm, cmap="gray")

    plt.show()
