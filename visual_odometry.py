import cv2
import numpy as np

class VisualOdometry:
    def __init__(self):
        self.orb = cv2.ORB_create()
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.prev_kp, self.prev_des = None, None
        self.prev_frame = None

    def process_frame(self, frame):
        kp, des = self.orb.detectAndCompute(frame, None)
        if self.prev_kp is None or self.prev_des is None:
            self.prev_kp, self.prev_des = kp, des
            self.prev_frame = frame
            return frame

        matches = self.bf.match(self.prev_des, des)
        matches = sorted(matches, key=lambda x: x.distance)

        img_matches = cv2.drawMatches(self.prev_frame, self.prev_kp, frame, kp, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        self.prev_kp, self.prev_des = kp, des
        self.prev_frame = frame

        return img_matches

if __name__ == "__main__":
    vo = VisualOdometry()
    cap = cv2.VideoCapture("gameplay_video.mp4")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        processed_frame = vo.process_frame(frame)
        cv2.imshow("Visual Odometry", processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()