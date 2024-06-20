import numpy as np
import cv2


def smallRegionFilter(img):

    # THRESH_BINARY_INV,将二值图取反，0变为255，因为 cv2.connectedComponentsWithStats 只能检测白色区域，取反后用于消除黑色小区域
    ret, thresh = cv2.threshold(
        img, 127, 255, cv2.THRESH_BINARY_INV)

    # 寻找连通域
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        thresh, connectivity=8)

    # 计算平均面积
    area = 0.1 * img.shape[0] * img.shape[1]

    # 处理连通域，小区域部分取反
    image_filtered = np.zeros_like(img)
    for i in range(1, num_labels):  # 跳过背景
        if stats[i, cv2.CC_STAT_AREA] <= area:
            # 取反操作
            image_filtered[labels == i] = 255 - thresh[labels == i]
        else:
            # 保留原始像素值
            image_filtered[labels == i] = thresh[labels == i]
    return image_filtered
