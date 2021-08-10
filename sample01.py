import cv2
import numpy as np
import math
from collections import deque
from statistics import  median

FONT = cv2.FONT_HERSHEY_SIMPLEX
RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
ORANGE = (0, 130, 255)
SCORE_AP = (0, 80)
SCORE_FONT_SIZE = 2
SCORE_FONT_WEIGHT = 8

score_queue = deque([])


def get_score_summary(total_arc_length):
    score_queue.append(math.floor(total_arc_length / 100))
    if len(score_queue) > 30:
        score_queue.popleft()

    score = median(score_queue)

    if score > 200:
        return score, '[OBEYA]', RED
    elif score > 150:
        return score, '[MESSY]', ORANGE
    else:
        return score, '[CLEAN]', BLUE


def is_obeya_contour(arc_length):
    return 100 < arc_length < 900


def show_obeya_score(frame):
    score_frame = frame.copy()

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canny_frame = cv2.Canny(gray_frame, 120, 200)
    contours, hierarchy = cv2.findContours(canny_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE )

    total_arc_length = 0
    for contour in contours:
        arc_length = cv2.arcLength(contour, True)
        if is_obeya_contour(arc_length):
            cv2.drawContours(score_frame, contour, -1, GREEN, 3)
            total_arc_length += arc_length

    score, decision, color = get_score_summary(total_arc_length)
    cv2.putText(score_frame, ' '.join([str(score), decision]),
            SCORE_AP, FONT, SCORE_FONT_SIZE, color, SCORE_FONT_WEIGHT, cv2.LINE_AA)

    cv2.imshow('frame: contour', score_frame)


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    while(True):
        ret, frame = cap.read()
        show_obeya_score(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
