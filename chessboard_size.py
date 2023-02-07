import numpy as np
import cv2
from matplotlib import pyplot as plt
import math


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

def remove_duplicate(array):
    result = [array[0]]
    for ele in array:
        check = True
        for res in result:
            if(np.abs(ele[1] - res[1]) < 30):
                check = False
                break
        if check:
            result.append(ele)

    return result

if __name__ == '__main__':
    # Đọc hình ảnh ở dạng xám
    img = cv2.imread('image/Chessboard0631.png', 0)
    # Xóa nhiễu sóng sin
    img = remove_sin_wave_noise(img)
    # Xóa nhiễu hạt tiêu, làm mượt    
    img = remove_noise_and_smooth(img)
    # Line detect
    lines = line_detection(img)

    array_object = []

    # Visualize
    imm = np.zeros((img.shape[0], img.shape[1], 3))
    for line in lines:
        # Extracted points nested in the list
        x1, y1 = line[0]
        x2, y2 = line[1]
        # Draw the lines joing the points
        # On the original image
        Ox = [1, 0]
        Oy = [0,1]
        vector = [x2 - x1, y2 - y1]
      
        unit_vector_1 = Ox / np.linalg.norm(Ox)
        unit_vector_2 = vector / np.linalg.norm(vector)
        dot_product = np.dot(unit_vector_1, unit_vector_2)
        angle = np.arccos(dot_product) if np.arccos(dot_product) <= math.pi / 2 else math.pi - np.arccos(dot_product) 
        angle_deg = angle * 180 / math.pi 

        # find distance
        p1=np.array([x1,y1])
        p2=np.array([x2,y2])
        p3=np.array([0,0])
        distance = np.linalg.norm(np.cross(p2-p1, p1-p3))/np.linalg.norm(p2-p1)
        array_object.append([angle_deg, distance, line])
        # print(angle * 180 / math.pi, end='\n')
        

    vertical = []
    horizontal = []

    for obj in array_object:
        if(obj[0] < 45):
            horizontal.append(obj)
        else:
            vertical.append(obj)

    # filter duplicate line
    print(len(remove_duplicate(vertical)), end='\n')
    print(len(remove_duplicate(horizontal)))

    verLine = remove_duplicate(vertical)

    for line in verLine:
        x1, y1 = line[2][0]
        x2, y2 = line[2][1]
        cv2.line(imm, (x1, y1), (x2, y2), (0, 255, 255), 2)
    

    horLine = remove_duplicate(horizontal)

    for line in horLine:
        x1, y1 = line[2][0]
        x2, y2 = line[2][1]
        cv2.line(imm, (x1, y1), (x2, y2), (0, 255, 255), 2)




    plt.imshow(imm, cmap="gray")

    plt.show()
