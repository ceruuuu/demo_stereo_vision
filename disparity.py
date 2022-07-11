import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm
#
# img1 = cv2.imread('images/zed/box_1_left.jpg', 0)  #queryimage # left image
# img2 = cv2.imread('images/zed/box_1_right.jpg', 0) #trainimage # right image
#
# sift = cv2.SIFT_create()
#
# # find the keypoints and descriptors with SIFT
# kp1, des1 = sift.detectAndCompute(img1, None)
# kp2, des2 = sift.detectAndCompute(img2, None)
#
# # orb = cv.ORB_create()![](leftimg.jpg)
# #
# # kp1, des1 = orb.detectAndCompute(img1, None)
# # kp2, des2 = orb.detectAndCompute(img2, None)
#
# # imgSift = cv2.drawKeypoints(img1, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# # cv2.imshow("SIFT Keypoints1", imgSift)
# # cv2.imwrite("SIFT Keypoints1.png", imgSift)
# #
# # imgSift = cv2.drawKeypoints(img2, kp2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# # cv2.imshow("SIFT Keypoints2", imgSift)
# # cv2.imwrite("SIFT Keypoints2.png", imgSift)
#
# # FLANN parameters
# FLANN_INDEX_KDTREE = 1
# index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
# # search_params = dict(checks=50)
# search_params = dict(checks=100)
#
# flann = cv2.FlannBasedMatcher(index_params, search_params)
# matches = flann.knnMatch(des1, des2, k=2)
#
# matchesMask = [[0, 0] for i in range(len(matches))]
# good = []
# pts1 = []
# pts2 = []
#
# # ratio test as per Lowe's paper
# for i, (m, n) in enumerate(matches):
#     if m.distance < 0.3 * n.distance:
#         matchesMask[i] = [1, 0]
#         good.append(m)
#         pts2.append(kp2[m.trainIdx].pt)
#         pts1.append(kp1[m.queryIdx].pt)
#
# draw_params = dict(matchColor=(0, 255, 0), singlePointColor=(255, 0, 0), matchesMask=matchesMask, flags=cv2.DrawMatchesFlags_DEFAULT)
#
# keypoint_matches = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)
# cv2.imshow("Keypoint matches", keypoint_matches)
# cv2.imwrite("Keypoint matching.png", keypoint_matches)
#
# pts1 = np.int32(pts1)
# pts2 = np.int32(pts2)
# F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)
#
# # We select only inlier points
# pts1 = pts1[mask.ravel() == 1]
# pts2 = pts2[mask.ravel() == 1]
#
# def drawlines(img1, img2, lines, pts1, pts2):
#     r, c = img1.shape
#     img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
#     img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
#
#     np.random.seed(0)
#     for r, pt1, pt2 in zip(lines, pts1, pts2):
#         color = tuple(np.random.randint(0, 255, 3).tolist())
#         x0, y0 = map(int, [0, -r[2] / r[1]])
#         x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
#         img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
#         img1 = cv2.circle(img1, tuple(pt1), 5, color, -1)
#         img2 = cv2.circle(img2, tuple(pt2), 5, color, -1)
#
#     return img1, img2
#
# # for i in range(len(pts1)):
# #     if pts1[i][1] == pts2[i][1]:
# #         print(pts1[i][0]-pts2[i][0])
#
# # Find epilines corresponding to points in right image (second image) and
# # drawing its lines on left image
# lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
# lines1 = lines1.reshape(-1, 3)
# img5, img6 = drawlines(img1, img2, lines1, pts1, pts2)
#
# # Find epilines corresponding to points in left image (first image) and
# # drawing its lines on right image
# lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
# lines2 = lines2.reshape(-1, 3)
# img3, img4 = drawlines(img2, img1, lines2, pts2, pts1)
#
# plt.subplot(121), plt.imshow(img5)
# plt.subplot(122), plt.imshow(img3)
# plt.suptitle("Epilines in both images")
# plt.show()
#
# h1, w1 = img1.shape
# h2, w2 = img2.shape
# _, H1, H2 = cv2.stereoRectifyUncalibrated(np.float32(pts1), np.float32(pts2), F, imgSize=(w1, h1))
#
# img1_rectified = cv2.warpPerspective(img1, H1, (w1, h1))
# img2_rectified = cv2.warpPerspective(img2, H2, (w2, h2))
# cv2.imwrite("rectified_1.png", img1_rectified)
# cv2.imwrite("rectified_2.png", img2_rectified)
#
# fig, axes = plt.subplots(1, 2, figsize=(15, 10))
# axes[0].imshow(img1_rectified, cmap="gray")
# axes[1].imshow(img2_rectified, cmap="gray")
# axes[0].axhline(250)
# axes[1].axhline(250)
# axes[0].axhline(450)
# axes[1].axhline(450)
# plt.suptitle("Rectified images")
# plt.savefig("rectified_images.png")
# plt.show()



imgL = cv2.imread('images/zed/box_1_left.jpg', 0)
imgR = cv2.imread('images/zed/box_1_right.jpg', 0)

# stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)

window_size = 3

left_matcher = cv2.StereoSGBM_create(minDisparity=1,
                                     numDisparities=50,
                                     blockSize=5,
                                     P1=8*3*window_size**2,
                                     P2=32*3*window_size**2,
                                     disp12MaxDiff=1,
                                     uniquenessRatio=5,
                                     speckleWindowSize=0,
                                     speckleRange=2,
                                     preFilterCap=63,
                                     mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
                                     # mode=cv2.STEREO_SGBM_MODE_HH)
# disparity = left_matcher.compute(imgL, imgR)

right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)

lmbda = 80000
sigma = 1.2
visual_multiplyer = 1.0

wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
wls_filter.setLambda(lmbda)
wls_filter.setSigmaColor(sigma)

disp_left = left_matcher.compute(imgL,imgR)
disp_right = right_matcher.compute(imgR,imgL)
disp_left = np.int16(disp_left)
disp_right = np.int16(disp_right)

filteredImg = wls_filter.filter(disp_left, imgL, None, disp_right)

filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg,
                            beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
# filteredImg = np.uint8(filteredImg)
filteredImg = np.uint16(filteredImg)

canvas = np.hstack((filteredImg, imgL, imgR))
# canvas = np.hstack((disparity, imgL, imgR))
plt.imshow(canvas, cmap=cm.gray)
plt.imshow(canvas)
plt.show()


cv2.waitKey()
cv2.destroyAllWindows()