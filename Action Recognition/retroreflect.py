import cv2
import numpy as np
cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_green = np.array([70,0,198])
    upper_green = np.array([179,255,255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    cv2.imshow('mask',mask)
    cv2.imshow('frame', frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()

#####################################################
# (hMin = 70 , sMin = 0, vMin = 198), (hMax = 179 , sMax = 255, vMax = 255)

# import cv2
# import numpy as np
# vid = cv2.VideoCapture(0)

# while(True):
#     ret, frame = vid.read()
#     # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#     # lower_green = np.array([0,20,0])
#     # upper_green = np.array([255,255,255])
#     # mask = cv2.inRange(hsv, lower_green, upper_green)
#     # res = cv2.bitwise_and(frame,frame,mask=mask)
#     cv2.imshow('orig',frame)
#     # cv2.imshow('fff',res)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# vid.release()
# cv2.destroyAllWindows()

# HSV :  [[[ 97  60 242]]]
# Red:  185
# Green:  229
# Blue:  242
# BRG Format:  [242 229 185]
# Coordinates of pixel: X:  1407 Y:  376

# HSV :  [[[107  40 255]]]
# Red:  215
# Green:  233
# Blue:  255
# BRG Format:  [255 233 215]
# Coordinates of pixel: X:  1420 Y:  344

# HSV :  [[[130  16 144]]]
# Red:  138
# Green:  135
# Blue:  144
# BRG Format:  [144 135 138]
# Coordinates of pixel: X:  1429 Y:  343



# import cv2
# import numpy as np
 

 
# def mouseRGB(event,x,y,flags,param):
#     if event == cv2.EVENT_LBUTTONDOWN: #checks mouse left button down condition
#         colorsB = image[y,x,0]
#         colorsG = image[y,x,1]
#         colorsR = image[y,x,2]
#         colors = image[y,x]
#         hsv_value= np.uint8([[[colorsB ,colorsG,colorsR ]]])
#         hsv = cv2.cvtColor(hsv_value,cv2.COLOR_BGR2HSV)
#         print ("HSV : " ,hsv)
#         print("Red: ",colorsR)
#         print("Green: ",colorsG)
#         print("Blue: ",colorsB)
#         print("BRG Format: ",colors)
#         print("Coordinates of pixel: X: ",x,"Y: ",y)
 
# # Read an image, a window and bind the function to window
# image = cv2.imread(r"C:\Users\there.DESKTOP-CKI66IV\Pictures\Camera Roll\WIN_20220129_20_10_47_Pro.jpg") #name of image
# cv2.namedWindow('mouseRGB')
# cv2.setMouseCallback('mouseRGB',mouseRGB)
 
# #Do until esc pressed
# while(True):
#     cv2.imshow('mouseRGB',image)
#     if cv2.waitKey(20) & 0xFF == 27:
#         break
# #if esc pressed, finish.
# cv2.destroyAllWindows()



# import cv2
# import numpy as np

# def nothing(x):
#     pass

# # Load image
# image = cv2.imread(r"C:\Users\there.DESKTOP-CKI66IV\Pictures\Camera Roll\WIN_20220129_20_10_47_Pro.jpg")

# # Create a window
# cv2.namedWindow('image')

# # Create trackbars for color change
# # Hue is from 0-179 for Opencv
# cv2.createTrackbar('HMin', 'image', 0, 179, nothing)
# cv2.createTrackbar('SMin', 'image', 0, 255, nothing)
# cv2.createTrackbar('VMin', 'image', 0, 255, nothing)
# cv2.createTrackbar('HMax', 'image', 0, 179, nothing)
# cv2.createTrackbar('SMax', 'image', 0, 255, nothing)
# cv2.createTrackbar('VMax', 'image', 0, 255, nothing)

# # Set default value for Max HSV trackbars
# cv2.setTrackbarPos('HMax', 'image', 179)
# cv2.setTrackbarPos('SMax', 'image', 255)
# cv2.setTrackbarPos('VMax', 'image', 255)

# # Initialize HSV min/max values
# hMin = sMin = vMin = hMax = sMax = vMax = 0
# phMin = psMin = pvMin = phMax = psMax = pvMax = 0

# while(1):
#     # Get current positions of all trackbars
#     hMin = cv2.getTrackbarPos('HMin', 'image')
#     sMin = cv2.getTrackbarPos('SMin', 'image')
#     vMin = cv2.getTrackbarPos('VMin', 'image')
#     hMax = cv2.getTrackbarPos('HMax', 'image')
#     sMax = cv2.getTrackbarPos('SMax', 'image')
#     vMax = cv2.getTrackbarPos('VMax', 'image')

#     # Set minimum and maximum HSV values to display
#     lower = np.array([hMin, sMin, vMin])
#     upper = np.array([hMax, sMax, vMax])

#     # Convert to HSV format and color threshold
#     hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#     mask = cv2.inRange(hsv, lower, upper)
#     result = cv2.bitwise_and(image, image, mask=mask)

#     # Print if there is a change in HSV value
#     if((phMin != hMin) | (psMin != sMin) | (pvMin != vMin) | (phMax != hMax) | (psMax != sMax) | (pvMax != vMax) ):
#         print("(hMin = %d , sMin = %d, vMin = %d), (hMax = %d , sMax = %d, vMax = %d)" % (hMin , sMin , vMin, hMax, sMax , vMax))
#         phMin = hMin
#         psMin = sMin
#         pvMin = vMin
#         phMax = hMax
#         psMax = sMax
#         pvMax = vMax

#     # Display result image
#     cv2.imshow('image', result)
#     if cv2.waitKey(10) & 0xFF == ord('q'):
#         break

# cv2.destroyAllWindows()