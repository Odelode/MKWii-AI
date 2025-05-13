import cv2
import numpy as np
import win32gui
import win32ui
import ctypes
import time

'''
This script captures the screen of a specific window using mss and OpenCV.
This is to be used to feed the AI footage.
'''

def capture_window(window_title):
    hwnd = win32gui.FindWindow(None, window_title)
    if not hwnd:
        raise Exception(f"No window titled `{window_title}`")

    left, top, right, bottom = win32gui.GetWindowRect(hwnd)
    w = right - left
    h = bottom - top

    hwnd_dc = win32gui.GetWindowDC(hwnd)
    mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
    save_dc = mfc_dc.CreateCompatibleDC()
    bmp = win32ui.CreateBitmap()
    bmp.CreateCompatibleBitmap(mfc_dc, w, h)
    save_dc.SelectObject(bmp)

    PW_RENDERFULLCONTENT = 0x00000002
    result = ctypes.windll.user32.PrintWindow(hwnd, save_dc.GetSafeHdc(), PW_RENDERFULLCONTENT)
    if not result:
        raise Exception("PrintWindow failed")

    buf = bmp.GetBitmapBits(True)
    img = np.frombuffer(buf, dtype=np.uint8).reshape((h, w, 4))

    win32gui.DeleteObject(bmp.GetHandle())
    save_dc.DeleteDC()
    mfc_dc.DeleteDC()
    win32gui.ReleaseDC(hwnd, hwnd_dc)

    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # cv2.COLOR_BGRA2BGR for with color

if __name__ == "__main__":
    window_title = "Dolphin 2412 | JIT64 DC | Direct3D 11 | HLE | Mario Kart Wii (RMCE01)" # If you update the emulator you're going to have to update the title.
    target_fps = 24 # Set the target FPS for the capture, lower is better for the AI to quickly process the frames.
    frame_duration = 1.0 / target_fps
    try:
        cv2.namedWindow("Direct Window Capture", cv2.WINDOW_NORMAL)
        while True:
            start_time = time.time()
            frame = capture_window(window_title)
            cv2.imshow("Direct Window Capture", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            elapsed = time.time() - start_time
            time_to_wait = frame_duration - elapsed
            if time_to_wait > 0:
                time.sleep(time_to_wait)
        cv2.destroyAllWindows()
    except Exception as e:
        print(e)