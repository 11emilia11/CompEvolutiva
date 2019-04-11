import cv2

img = cv2.imread('C:/Users/Auricelia/Documents/Legendas/polygons.png')
orig = img.copy()

cv2.imshow("a", img)
cv2.waitKey()

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,3,10)
img = cv2.bitwise_not(img)

countours, hierachy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

print(len(countours))
vertices = []
coord = []
for x in countours:
    peri = cv2.arcLength(x, True)
    aproxx = cv2.approxPolyDP(x, 0.04 * peri, True)  # vertices
    coord.append(aproxx)
    vertices.append(len(aproxx))

print(vertices)

cv2.imshow("a", img)
cv2.waitKey()