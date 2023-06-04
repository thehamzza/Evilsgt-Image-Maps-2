import cv2
import numpy as np


# Variables
# 00: L. eye L. x
# 01: L. eye L. y
# 02: L. eye R. x
# 03: L. eye R. y
# 04: L. eye high x
# 05: L. eye high y
# 06: L. eye low x
# 07: L. eye low y
# 08: R. eye L. x
# 09: R. eye L. y
# 10: R. eye R. x
# 11: R. eye R. y
# 12: R. eye high x
# 13: R. eye high y
# 14: R. eye low x
# 15: R. eye low y
# 16: Nose L. x
# 17: Nose L. y
# 18: Nose R. x
# 19: Nose R. y
# 20: Nose tip x
# 21: Nose tip y
# 22: L. ear bot. X
# 23: L. ear bot. Y
# 24: R. ear bot. X
# 25: R. ear bot. Y
# 26: Mouth L. x
# 27: Mouth L. y
# 28: Mouth R. x
# 29: Mouth R. y
# 30: Mouth top x
# 31: Mouth top y
# 32: Mouth bot x
# 33: Mouth bot y
# 34: Chin x
# 35: Chin y
# 36: Jaw L. x
# 37: Jaw L. y
# 38: Jaw R. x
# 39: Jaw R. y




# Faces data (f01 - f06)
data = np.array([
[ 92,  95,  91, 101,  60, 135],
[189,  76, 185, 250,  89, 121],
[138, 174, 130, 148, 104, 170],
[187,  90, 191, 236,  94, 121],
[112, 130, 110, 116,  81, 147],
[174,  61, 177, 229,  78, 109],
[114, 137, 111, 127,  83, 154],
[192,  99, 196, 246,  95, 125],
[181, 259, 171, 203, 153, 215],
[188,  92, 190, 227,  92, 108],
[224, 342, 211, 252, 194, 250],
[184,  82, 188, 217,  87,  92],
[203, 305, 190, 223, 172, 225],
[174,  64, 177, 208,  77,  88],
[206, 308, 192, 229, 172, 234],
[192, 100, 196, 225,  93, 104],
[134, 173, 129, 157, 101, 185],
[239, 189, 235, 285, 137, 165],
[184, 258, 171, 215, 154, 224],
[239, 190, 235, 274, 133, 151],
[160, 214, 151, 190, 128, 208],
[250, 210, 246, 282, 146, 169],
[ 74,  63,  66,  96,  35, 128],
[243, 174, 243, 339, 152, 149],
[243, 385, 236, 288, 208, 291],
[242, 189, 243, 301, 147, 116],
[125, 157, 119, 146,  88, 189],
[280, 264, 274, 330, 174, 195],
[194, 281, 181, 239, 157, 244],
[279, 261, 273, 310, 174, 181],
[160, 213, 151, 191, 127, 213],
[272, 239, 267, 311, 159, 183],
[160, 217, 151, 202, 124, 224],
[295, 298, 288, 345, 206, 206],
[161, 217, 152, 208, 121, 233],
[335, 364, 331, 390, 246, 234],
[ 97,  87,  80, 114,  44, 155],
[298, 269, 285, 363, 180, 195],
[224, 360, 222, 283, 193, 285],
[299, 273, 285, 327, 188, 166]
]).astype(float)



########################################
##
def annoteImage(im, pts):
    pts = pts.astype(int)

    # Mark points
    for i in range(0, int(len(pts)), 2):
        cv2.circle(im, (pts[i], pts[i+1]), 2, (0,0,255), -1)

    # Jaw
    cv2.line(im, pts[22:24], pts[36:38], (0,0,255), 2)
    cv2.line(im, pts[36:38], pts[34:36], (0,0,255), 2)
    cv2.line(im, pts[34:36], pts[38:40], (0,0,255), 2)
    cv2.line(im, pts[38:40], pts[24:26], (0,0,255), 2)
    # Mouth
    cv2.line(im, pts[26:28], pts[30:32], (0,0,255), 2)
    cv2.line(im, pts[30:32], pts[28:30], (0,0,255), 2)
    cv2.line(im, pts[28:30], pts[32:34], (0,0,255), 2)
    cv2.line(im, pts[32:34], pts[26:28], (0,0,255), 2)
    cv2.line(im, pts[26:28], pts[28:30], (0,0,255), 2)
    # Left eye
    cv2.line(im, pts[0:2], pts[4:6], (0,0,255), 2)
    cv2.line(im, pts[4:6], pts[2:4], (0,0,255), 2)
    cv2.line(im, pts[2:4], pts[6:8], (0,0,255), 2)
    cv2.line(im, pts[6:8], pts[0:2], (0,0,255), 2)
    # Right eye
    cv2.line(im, pts[8:10], pts[12:14], (0,0,255), 2)
    cv2.line(im, pts[12:14], pts[10:12], (0,0,255), 2)
    cv2.line(im, pts[10:12], pts[14:16], (0,0,255), 2)
    cv2.line(im, pts[14:16], pts[8:10], (0,0,255), 2)
    # Nose
    cv2.line(im, pts[2:4], pts[16:18], (0,0,255), 2)
    cv2.line(im, pts[16:18], pts[20:22], (0,0,255), 2)
    cv2.line(im, pts[20:22], pts[18:20], (0,0,255), 2)
    cv2.line(im, pts[18:20], pts[8:10], (0,0,255), 2)




########################################
##
if __name__ == '__main__':
    im = cv2.imread('f01.jpg')
    annoteImage(im, data[:,0].transpose())
    cv2.imwrite('f01_annot.jpg', im)

    im = cv2.imread('f02.jpg')
    annoteImage(im, data[:,1].transpose())
    cv2.imwrite('f02_annot.jpg', im)

    im = cv2.imread('f03.jpg')
    annoteImage(im, data[:,2].transpose())
    cv2.imwrite('f03_annot.jpg', im)

    im = cv2.imread('f04.jpg')
    annoteImage(im, data[:,3].transpose())
    cv2.imwrite('f04_annot.jpg', im)

    im = cv2.imread('f05.jpg')
    annoteImage(im, data[:,4].transpose())
    cv2.imwrite('f05_annot.jpg', im)

    im = cv2.imread('f06.jpg')
    annoteImage(im, data[:,5].transpose())
    cv2.imwrite('f06_annot.jpg', im)

    # Draw mean face
    im = np.zeros((400, 400, 3))
    mean_face = data.mean(axis=1)
    mean_face = mean_face.astype(int)
    annoteImage(im, mean_face)
    cv2.imwrite('meanface_annot.jpg', im)

    print('\nMean face:')
    print(mean_face)

    # Put mean face on an unseen image
    im = cv2.imread('fn01.jpg')
    annoteImage(im, mean_face)
    cv2.imwrite('fn01_annot_mean.jpg', im)
