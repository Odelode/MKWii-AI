import cv2
import numpy as np
from mss import mss

def capture_screen(region = None):
    with mss() as sct:
        screenshot = sct.grab(region)
        img = np.array(screenshot)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img

if __name__ == "__main__":
    region = {"top": 0, "left": 1920, "width": 1920, "height": 1080}
    cv2.namedWindow(("Screen Capture"), cv2.WINDOW_NORMAL)
    while True:
        frame = capture_screen(region)
        cv2.imshow("Screen Capture", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()