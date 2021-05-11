import cv2
import matplotlib.pyplot as plt
import numpy as np

save_vid = True
show_plot = True

filename = 'test.mpg'
# filename = 'blackfly_test.avi'
# filename = 'casio.MOV'
# filename = 'CIMG4792.mkv'
# filename = 'casio_closeup.MOV'

print("Running")

cap = cv2.VideoCapture('vids/' + filename)

# Take first frame and find corners in it
ret, old_frame = cap.read()

size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
fps = cap.get(cv2.CAP_PROP_FPS)

old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

plt.imshow(old_gray, cmap='gray', vmin=0, vmax=255)

ret, thresh = cv2.threshold(old_gray, 100, 255, 0)

plt.figure()
plt.imshow(thresh, cmap='gray', vmin=0, vmax=255)

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

img = np.copy(old_gray)
cv2.drawContours(img, contours, -1, (0, 255, 0), 3)

fig = plt.figure
plt.imshow(img)

fig, ax = plt.subplots()
[ax.plot(ct[:, :, 0], ct[:, :, 1], c='k') for ct in contours]
cv2.destroyAllWindows()
cap.release()

# Create a mask image for drawing purposes

while True:
    ret, frame = cap.read()
    break
