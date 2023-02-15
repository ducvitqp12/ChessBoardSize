import numpy as np
import cv2
from matplotlib import pyplot as plt
import math
from scipy.ndimage.measurements import label


def denoisePeriodic(img, size_filter=2, diff_from_center_point=50, size_thresh=2):
    h, w = img.shape[:]
    img_float32 = np.fft.fft2(img)
    fshift = np.fft.fftshift(img_float32)

    # show the furier  image transform by log e of fft
    furier_tr = 20 * np.log(np.abs(fshift))

    # plt.subplot(121),plt.imshow(img, cmap = 'gray')
    # plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    # plt.subplot(122),plt.imshow(furier_tr, cmap = 'gray')
    # plt.show()

    # get center point value
    center_fur = furier_tr[int(h / 2)][int(w / 2)]

    # find pick freq point
    new_fur = np.copy(furier_tr)
    kernel = np.ones((2 * size_filter + 1, 2 * size_filter + 1), np.float32) / (
            (2 * size_filter + 1) * (2 * size_filter + 1) - 1)
    kernel[size_filter][size_filter] = -1
    kernel = -kernel
    # print(kernel)
    dst = cv2.filter2D(new_fur, -1, kernel)

    diff_from_center_point = center_fur * diff_from_center_point / 356
    dst[0][:] = dst[1][:] = dst[:][0] = dst[:][1] = 0

    dst[int(h / 2)][int(w / 2)] = 0
    index = np.where(dst > diff_from_center_point)
    # print("index",index)

    # remove point is not the pick one
    index_x = []
    index_y = []

    for i, item in enumerate(index[0]):

        value = furier_tr[index[0][i]][index[1][i]]
        # print("value ", value)
        matrix = np.copy(furier_tr[max(0, index[0][i] - size_filter):min(h, index[0][i] + size_filter + 1),
                         max(0, index[1][i] - size_filter):min(w, index[1][i] + size_filter + 1)])
        # print("new maxtirx", matrix)
        matrix[size_filter][size_filter] = 0

        max_value = np.amax(matrix)
        # print("mean", max_value)

        if (value - max_value < 20):
            continue
        index_y.append(index[0][i])
        index_x.append(index[1][i])

    # set freq value of pick points to 1
    for i, item in enumerate(index_x):
        for j in range(size_thresh):
            for k in range(size_thresh):
                x = max(0, min(int(index_y[i] - int(size_thresh / 2) + j), h - 1))
                y = max(0, min(int(index_x[i] - int(size_thresh / 2) + k), w - 1))
                # print("toa do", x, y)
                furier_tr[x, y] = 1
                fshift[x, y] = 1

    # inverse to image
    # inverse shift
    f_ishift = np.fft.ifftshift(fshift)
    # inverse furier
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back).astype(np.uint8)
    # plt.subplot(121),plt.imshow(img_back, cmap = 'gray')
    # plt.show()

    return img_back


def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img


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

    # cv2.imshow("thres", median)
    #
    # cv2.waitKey(0)

    blur = cv2.GaussianBlur(median, (5, 5), 0)  # Làm mượt ảnh

    # cv2.imshow("thres", blur)
    #
    # cv2.waitKey(0)

    im = (cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))).apply(blur)  # Cân bằng histogram tăng độ tương phản

    cv2.imshow("His", im)

    cv2.waitKey(0)

    ret, thresh = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # Threshold về binary image

    cv2.imshow("thres", thresh)

    cv2.waitKey(0)

    #
    return thresh


def line_detection(image):
    # Edge detection
    edges = cv2.Canny(image, 50, 150, apertureSize=3)

    cv2.imshow("edge", edges)

    cv2.waitKey(0)

    lines_list = []

    # Line Detection
    lines = cv2.HoughLinesP(
        edges,  # Input edge image
        3,  # Distance resolution in pixels
        np.pi / 180,  # Angle resolution in radians
        threshold=100,  # Min number of votes for valid line
        minLineLength=5,  # Min allowed length of line
        maxLineGap=50  # Max allowed gap between line for joining them
    )
    #
    # # Iterate over points
    for points in lines:
        # Extracted points nested in the list
        x1, y1, x2, y2 = points[0]
        # Maintain a simples lookup list for points
        lines_list.append([(x1, y1), (x2, y2)])

    return lines_list


def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        return -1, -1

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div

    return x, y


def remove_duplicate(array, img):
    result = [array[0]]
    for ele in array:
        check = True
        for res in result:
            if abs(ele[0] - res[0]) < 5 and np.abs(ele[1] - res[1]) < 20:
                check = False
                break
            if np.abs(ele[1] - res[1]) < 4 and (ele[0] - 90) * (res[0] - 90) >= 0:
                check = False
                break

            x, y = line_intersection(ele[2], res[2])

            if 0 < x < img.shape[1] and 0 < y < img.shape[0]:
                check = False
                break

        if check:
            result.append(ele)

    return result


def remove_outlier(array):
    angles = []
    for ele in array:
        angle = ele[0] if ele[0] <= 90 else 180 - ele[0]
        angles.append(angle)
    mean = np.mean(angles)
    std = np.std(angles)

    threshold = 2*std

    # print(mean)
    # print(std)

    outliers_removed = []

    for ele in array:
        angle = ele[0] if ele[0] <= 90 else 180 - ele[0]
        if abs(angle - mean) < threshold:
            outliers_removed.append(ele)

    return outliers_removed


if __name__ == '__main__':
    # Đọc hình ảnh ở dạng xám
    img = cv2.imread('image/Chessboard_0511.png', 0)

    # print(img.shape)
    # img = increase_brightness(img)
    #
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Xóa nhiễu sóng sin
    img = denoisePeriodic(img)

    cv2.imshow("result", img)

    cv2.waitKey(0)

    # Xóa nhiễu hạt tiêu, làm mượt    
    img = remove_noise_and_smooth(img)
    # Line detect

    lines = line_detection(img)

    array_object = []

    # Visualize
    imm = np.zeros((img.shape[0], img.shape[1], 3))

    for line in lines:
        # Extracted points nested in the list
        if line[0][1] > line[1][1]:
            x1, y1 = line[0]
            x2, y2 = line[1]
        else:
            x1, y1 = line[1]
            x2, y2 = line[0]
        # Draw the lines joing the points
        # On the original image
        Ox = [1, 0]
        Oy = [0, 1]
        vector = [x2 - x1, y2 - y1]
        # cv2.line(imm, (x1, y1), (x2, y2), (0, 255, 255), 2)

        unit_vector_1 = Ox / np.linalg.norm(Ox)
        unit_vector_2 = vector / np.linalg.norm(vector)
        dot_product = np.dot(unit_vector_1, unit_vector_2)
        angle = np.arccos(dot_product)
        angle_deg = angle * 180 / math.pi

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
        if obj[0] < 45 or obj[0] > 135:
            horizontal.append(obj)
        else:
            vertical.append(obj)

    # filter duplicate line

    verLine = remove_outlier(vertical)
    verLine = remove_duplicate(verLine, imm)
    # verLine = remove_outlier(verLine)

    for line in verLine:
        x1, y1 = line[2][0]
        x2, y2 = line[2][1]
        cv2.line(imm, (x1, y1), (x2, y2), (0, 255, 255), 2)

    horLine = remove_outlier(horizontal)
    horLine = remove_duplicate(horLine, imm)
    # horLine = remove_outlier(horLine)

    for line in horLine:
        x1, y1 = line[2][0]
        x2, y2 = line[2][1]
        cv2.line(imm, (x1, y1), (x2, y2), (0, 255, 255), 2)

    print(len(verLine), end='\n')
    print(len(horLine))

    plt.imshow(imm, cmap="gray")

    plt.show()
