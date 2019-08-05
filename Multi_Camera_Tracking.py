import numpy as np
import cv2
import imutils
from tkinter import filedialog
global height_roi, width_roi, roi,xa,xb,ya,yb ,min_len
rect = (0, 0, 0, 0)  # tuple
startPoint = False
endPoint = False

def on_mouse(event, x, y, flags, params):
    global rect, startPoint, endPoint
    # get mouse click
    if event == cv2.EVENT_LBUTTONDOWN:
        if startPoint == True and endPoint == True:
            startPoint = False
            endPoint = False
            rect = (0, 0, 0, 0)
        if startPoint == False:
            rect = (x, y, 0, 0)
            startPoint = True
        elif endPoint == False:
            rect = (rect[0], rect[1], x, y)
            endPoint = True
        print(rect)
name1 = filedialog.askopenfilename()
camera1 = cv2.VideoCapture(name1)
frame_count = int(camera1.get(cv2.CAP_PROP_FRAME_COUNT))

def display_img(img, name="image"):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def matches_function(roi, train):
    img1 = roi  # queryImage
    img2 = train  # trainImage
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    bf = cv2.BFMatcher(cv2.NORM_L1,crossCheck=False)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    sum1 = 0
    for j in range(5):
         sum1 += matches[j].distance
    return sum1

screenshot = False
minDist = []
move = 40
counter =0
while (camera1.isOpened()):
    (ret, frame) = camera1.read()
    counter += 1
    frame = imutils.resize(frame, 600, 900)
    blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([38, 86, 0])
    upper_blue = np.array([121, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    c = len(contours)
    cv2.imshow("Mask", mask)
    print('contours',len(contours))

    min_len = 1
    max_template_match = 1
    if (screenshot == True  and len(contours) > 80):
        # same
        h = frame.shape[1] - rect[1]
        w = width_roi
        x1 = rect[0]
        y1 = rect[1]
        x2 = rect[2]
        y2 = rect[3]
        same = frame[y1:y1 + h, x1:x1 + w]
        same_len = matches_function(roi, same)
        #print(same_len)
        min_len = same_len
        xa = x1
        ya = y1
        xb = x2
        yb = y2

        xc = x1
        yc = y1
        xd = x2
        yd = y2
        same_template_match = cv2.matchTemplate(same,roi,cv2.TM_CCOEFF_NORMED)
        #_, max_template_match, _, _ = cv2.minMaxLoc(same_template_match)
        max_template_match = np.max(same_template_match)
        # -------------------------------------------------------------------
        # above
        h = frame.shape[1] - rect[1]
        w = width_roi
        x1 = rect[0]
        y1 = rect[1] - h
        x2 = rect[2]
        y2 = rect[1]
        above = frame[y1:y1 + h, x1:x1 + w]
        if above.any()==None:
            continue
        else:
            above_len = matches_function(roi, above)
            #print(above_len)
            if above_len < min_len:
                min_len = above_len
                xa = rect[0]
                ya = rect[1] + move
                xb = rect[2]
                yb = rect[3] + move
            above_tem_match = np.max(cv2.matchTemplate(above,roi,cv2.TM_CCOEFF_NORMED))
            #_, max_template_match, _, _ = cv2.minMaxLoc(above_tem_match)
            max_above = np.max(above_tem_match)
            if(max_above > max_template_match):
                max_template_match = max_above
                xc = rect[0]
                yc = rect[1] + move
                xd = rect[2]
                yd = rect[3] + move
        # -----------------------------------------------------------
        # below
        h = rect[3]
        w = width_roi
        x1 = rect[0]
        y1 = rect[3]
        x2 = rect[2]
        y2 = 0
        below = frame[y1:y1 + h, x1:x1 + w]
        if below.any() == None:
            continue
        else :
            below_len = matches_function(roi, below)
            if below_len < min_len:
                min_len = below_len
                xa = rect[0]
                ya = rect[1] - move
                xb = rect[2]
                yb = rect[3] - move
            below_tem_match = cv2.matchTemplate(below, roi, cv2.TM_CCOEFF_NORMED)
            max_below = np.max(below_tem_match)
            if (max_below > max_template_match):
                max_template_match = max_below
                xa = rect[0]
                ya = rect[1] - move
                xb = rect[2]
                yb = rect[3] - move
        # ------------------------------------------------------------------------
        # right
        h = height_roi
        w = frame.shape[0] + rect[2]
        x1 = rect[2]
        y1 = rect[1]
        x2 = frame.shape[0]
        y2 = rect[3]
        right = frame[y1:y1 + h, x1:x1 + w]
        if right.any()==None:
            continue
        else :
            right_len = matches_function(roi, right)
            if right_len < min_len:
                min_len = right_len
                xa = rect[0] + move
                ya = rect[1]
                xb = rect[2] + move
                yb = rect[3]
            right_tem_match = cv2.matchTemplate(right, roi, cv2.TM_CCOEFF_NORMED)
            max_right = np.max(right_tem_match)
            if (max_right > max_template_match):
                max_template_match = max_right
                xc = rect[0] + move
                yc = rect[1]
                xd = rect[2] + move
                yd = rect[3]
        #--------------------------------------------------------------------------
        # left
        h = height_roi
        w = rect[0]
        x1 = 0
        y1 = rect[1]
        x2 = rect[0]
        y2 = rect[3]
        left = frame[y1:y1 + h, x1:x1 + w]
        if left.any()==None:
            continue
        else:
            left_len = matches_function(roi, left)
            if left_len < min_len:
                min_len = left_len
                xa = rect[0] - move
                ya = rect[1]
                xb = rect[2] - move
                yb = rect[3]
            left_tem_match = cv2.matchTemplate(left, roi, cv2.TM_CCOEFF_NORMED)
            max_left = np.max(left_tem_match)
            if (max_left > max_template_match):
                max_template_match = max_left
                xc = rect[0] - move
                yc = rect[1]
                xd = rect[2] - move
                yd = rect[3]
                move += 10
        #print("min_sift", min_len)
        #print('max_template_matches',max_template_match)
        cv2.rectangle(frame, (xa+2, ya+2), (xb+2, yb+2), (0, 255, 0), 2)
        cv2.rectangle(frame, (xc, yc), (xd, yd), (255, 0, 0), 2)
        cv2.imwrite('E:/CV_Project/save/scenario3/camera1/'+str(counter)+'.png', frame)
        # print("the minimum distance is ", min(minDist))
    if (min_len == 0 or max_template_match == 0 ):
        cv2.rectangle(frame, (xa+2, ya+2), (xb+2, yb+2), (0, 0, 0), 2)
        cv2.rectangle(frame, (xc, yc), (xd, yd), (0, 0, 0), 2)

        break
    cv2.namedWindow('frame')
    cv2.setMouseCallback('frame', on_mouse)
    if screenshot == False:
      cv2.rectangle(frame, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), 2)
    if frame.size > 0:
        cv2.imshow('frame', frame)
    key = cv2.waitKey(1)
    if key == ord(' '):
        while True:
            screenshot = True
            key2 = cv2.waitKey(1) or 0xff
            cv2.imshow('frame', frame)
            if key2 == ord(' '):
                break
        height_roi = np.abs(rect[1] - rect[3])
        width_roi = np.abs(rect[2] - rect[0])
        roi = frame[rect[1]:rect[1] + height_roi, rect[0]:rect[0] + width_roi]
        cv2.rectangle(frame, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), 2)
        #hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        cv2.imshow("roi", roi)
        # cv2.rectangle(frame, (rect[0], rect[1]), (rect[2], rect[3]), (0, 0, 0), 2)
    if (key == ord("r")):
        break
name2 = filedialog.askopenfilename()
camera2 = cv2.VideoCapture(name2)
minDist = []
counter1 =0
while (camera2.isOpened()):
    counter1 += 1
    #print('counter: ', counter)
    kkk = 0
    (ret, frame) = camera2.read()
    frame = imutils.resize(frame, 600, 900)
    frame=imutils.rotate(frame,-90)
    blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([38, 86, 0])
    upper_blue = np.array([121, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    c = len(contours)
    #print(len(contours))
    if(len(contours) > 20):
        min_len = 1
        max_template_match = 1
        counter1+=1
        # same
        h = frame.shape[1] - rect[1]
        w = width_roi
        x1 = rect[0]
        y1 = rect[1]
        x2 = rect[2]
        y2 = rect[3]
        same = frame[y1:y1 + h, x1:x1 + w]
        same_len = matches_function(roi, same)
        min_len = same_len
        xa = x1
        ya = y1
        xb = x2
        yb = y2

        xc = x1
        yc = y1
        xd = x2
        yd = y2
        same_template_match = cv2.matchTemplate(same, roi, cv2.TM_CCOEFF_NORMED)
        # _, max_template_match, _, _ = cv2.minMaxLoc(same_template_match)
        max_template_match = np.max(same_template_match)
        # -------------------------------------------------------------------
        # above
        h = frame.shape[1] - rect[1]
        w = width_roi
        x1 = rect[0]
        y1 = rect[1] - h
        x2 = rect[2]
        y2 = rect[1]
        above = frame[y1:y1 + h, x1:x1 + w]
        if above.any() ==None:
            continue
        else:
            above_len = matches_function(roi, above)
            if above_len < min_len:
                min_len = above_len
                xa = rect[0]
                ya = rect[1] + move
                xb = rect[2]
                yb = rect[3] + move
            above_tem_match = np.max(cv2.matchTemplate(above, roi, cv2.TM_CCOEFF_NORMED))
            # _, max_template_match, _, _ = cv2.minMaxLoc(above_tem_match)
            max_above = np.max(above_tem_match)
            if (max_above > max_template_match):
                max_template_match = max_above
                xc = rect[0]
                yc = rect[1] + move
                xd = rect[2]
                yd = rect[3] + move
        # -----------------------------------------------------------
        # below
        # h = rect[3]
        # w = width_roi
        # x1 = rect[0]
        # y1 = rect[3]
        # x2 = rect[2]
        # y2 = 0
        # below = frame[y1:y1 + h, x1:x1 + w]
        # if below.any() == None:
        #     continue
        # else:
        #     below_len = matches_function(roi, below)
        #     if below_len < min_len:
        #         min_len = below_len
        #         xa = rect[0]
        #         ya = rect[1] - move
        #         xb = rect[2]
        #         yb = rect[3] - move
        #     below_tem_match = cv2.matchTemplate(below, roi, cv2.TM_CCOEFF_NORMED)
        #     max_below = np.max(below_tem_match)
        #     if (max_below > max_template_match):
        #         max_template_match = max_below
        #         xa = rect[0]
        #         ya = rect[1] - move
        #         xb = rect[2]
        #         yb = rect[3] - move
        # ------------------------------------------------------------------------
        # right
        h = height_roi
        w = frame.shape[0] + rect[2]
        x1 = rect[2]
        y1 = rect[1]
        x2 = frame.shape[0]
        y2 = rect[3]
        right = frame[y1:y1 + h, x1:x1 + w]
        if right.any()==None:
            continue
        else:
            right_len = matches_function(roi, right)
            if right_len < min_len:
                min_len = right_len
                xa = rect[0] + move
                ya = rect[1]
                xb = rect[2] + move
                yb = rect[3]
            right_tem_match = cv2.matchTemplate(right, roi, cv2.TM_CCOEFF_NORMED)
            max_right = np.max(right_tem_match)
            if (max_right > max_template_match):
                max_template_match = max_right
                xc = rect[0] + move
                yc = rect[1]
                xd = rect[2] + move
                yd = rect[3]
        # --------------------------------------------------------------------------
        # left
        h = height_roi
        w = rect[0]
        x1 = 0
        y1 = rect[1]
        x2 = rect[0]
        y2 = rect[3]
        left = frame[y1:y1 + h, x1:x1 + w]
        if left.any()==None :
            continue
        else:
            left_len = matches_function(roi, left)
            # print("left", left_len)
            # minDist.append(left_len)
            if left_len < min_len:
                min_len = left_len
                xa = rect[0] - move
                ya = rect[1]
                xb = rect[2] - move
                yb = rect[3]
            left_tem_match = cv2.matchTemplate(left, roi, cv2.TM_CCOEFF_NORMED)
            max_left = np.max(left_tem_match)
            if (max_left > max_template_match):
                max_template_match = max_left
                xc = rect[0] - move
                yc = rect[1]
                xd = rect[2] - move
                yd = rect[3]
                move += 10
        cv2.rectangle(frame, (xa + 2, ya + 2), (xb + 2, yb + 2), (0, 255, 0), 2)
        cv2.rectangle(frame, (xc, yc), (xd, yd), (255, 0, 0), 2)
        cv2.imwrite('E:/CV_Project/save/scenario3/camera2/' + str(counter1) + '.png', frame)
        # print("the minimum distance is ", min(minDist))
    if (min_len == 0 or max_template_match == 0 ):
        cv2.rectangle(frame, (xa+2, ya+2), (xb+2, yb+2), (0, 0, 0), 2)
        cv2.rectangle(frame, (xc, yc), (xd, yd), (0, 0, 0), 2)
        break
    cv2.namedWindow('frame')
    #cv2.setMouseCallback('frame', on_mouse)
    # drawing rectangle
    # if startPoint == True and endPoint == True:
    #cv2.rectangle(frame, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), 2)
    if frame.size > 0:
        cv2.imshow('frame', frame)
    key = cv2.waitKey(1)
    if (key == ord("r") ):
        break

