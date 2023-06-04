import cv2
import numpy as np
import pandas as pd

# Change numpy's output format
np.set_printoptions(formatter={'float_kind':"{:.2f}".format})


data = pd.read_csv('island_annotation.csv').drop(columns=['image_path'])


data = np.array(data.transpose()).astype(float)



C = np.cov(data)

# Covariance matrix can come out as complex numbers with very small (0) imaginary value
# So, make it real
C = np.abs(C)

print('Covariance Matrix:\n', C)
print('Shape:', C.shape)



vals, vecs = np.linalg.eig(C)

# vals and vecs can come out as complex numbers with very small (0) imaginary value
# So, make it real
vals = np.abs(vals)
vecs = np.abs(vecs)

print('Eigenvalues:')
print(vals)
print('Shape:', vals.shape)
print()
print('Eigenvectors:')
print(vecs)
print('Shape:', vecs.shape)

# Mean face
mean_face = data.mean(axis=1)
print('Mean face:')
print(mean_face)
print('Shape:', mean_face.shape)


mean_features = data.mean(axis=1)
mean_snake = mean_features.reshape((-1,2))

# Create image to hold generated faces
face_image = np.zeros((667, 667, 3))

# Generate a face
b = [0] * 800 # b holds Shape Paramaters - altered in callbacks
# Generative Shape Model
face = mean_face + vecs @ b
face = face.astype(int)


def annoteImage(im, pts):
    # Mark points
    for i in range(0, len(pts), 2):
        cv2.circle(im, (pts[i], pts[i + 1]), 2, (0, 0, 255), -1)

    # Draw island contour
    for i in range(0, len(pts) - 2, 2):
        cv2.line(im, (pts[i], pts[i + 1]), (pts[i + 2], pts[i + 3]), (0, 0, 255), 2)

    # Connect the last point to the first point
    cv2.line(im, (pts[-2], pts[-1]), (pts[0], pts[1]), (0, 0, 255), 2)

########################################
def displayImage():
    # Clear image
    face_image[::] = 0

    # Generative Shape Model
    face = mean_face + vecs @ b
    face = face.astype(int)

    annoteImage(face_image, face)
    cv2.imshow('Generated Image', face_image)


########################################
## Callback function to receive messages from the TrackBar
def changeb0(x):
    b[0] = (x-250) # moving vertically

    displayImage()


########################################
## Callback function to receive messages from the TrackBar
def changeb1(x):
    b[1] = (x-250) # moving vertically

    displayImage()


########################################
## Callback function to receive messages from the TrackBar
def changeb2(x):
    b[2] = (x-250) # moving vertically

    displayImage()


########################################
## Callback function to receive messages from the TrackBar
def changeb3(x):
    b[3] = (x-250) # moving vertically

    displayImage()


########################################
## Callback function to receive messages from the TrackBar
def changeb4(x):
    b[4] = (x-250) # moving vertically

    displayImage()


########################################
## Callback function to receive messages from the TrackBar
def changeb5(x):
    b[5] = (x-250) # moving vertically

    displayImage()



# Set up window
cv2.namedWindow('Generated Image', cv2.WINDOW_NORMAL)
#
# Create TrackBar

default_init = 500
max_value = 1000

cv2.createTrackbar('b[0]', 'Generated Image', 0, max_value, changeb0)
cv2.setTrackbarPos('b[0]', 'Generated Image', default_init)
# #
cv2.createTrackbar('b[1]', 'Generated Image', 0, max_value, changeb1)
cv2.setTrackbarPos('b[1]', 'Generated Image', default_init)
#
cv2.createTrackbar('b[2]', 'Generated Image', 0, max_value, changeb2)
cv2.setTrackbarPos('b[2]', 'Generated Image', default_init)
#
cv2.createTrackbar('b[3]', 'Generated Image', 0, max_value, changeb3)
cv2.setTrackbarPos('b[3]', 'Generated Image', default_init)
#
cv2.createTrackbar('b[4]', 'Generated Image', 0, max_value, changeb4)
cv2.setTrackbarPos('b[4]', 'Generated Image', default_init)
#
cv2.createTrackbar('b[5]', 'Generated Image', 0, max_value, changeb5)
cv2.setTrackbarPos('b[5]', 'Generated Image', default_init)

# Show generated face
annoteImage(face_image, face)
cv2.imshow('Generated Image', face_image)

# Wait for spacebar press before closing,
# otherwise window will close without you seeing it
while True:
   if cv2.waitKey(1) == ord(' '):
       break
