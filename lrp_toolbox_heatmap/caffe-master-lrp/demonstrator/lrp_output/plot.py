import cv2

img1 =cv2.imread("/someimages/frame_0229.jpg_as_inputted_into_the_dnn.png")
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

img = cv2.imread("/someimages/frame_0229.jpg_heatmap.png")
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)



heatmap_img = cv2.applyColorMap(gray_img, cv2.COLORMAP_JET)
fin = cv2.addWeighted(heatmap_img, 0.7, img1, 0.3, 0)
cv2.imshow('image', image)
