import cv2
import numpy as np
from pyzbar.pyzbar import decode
# pyzbar is the library that allows us to detect and localize barcodes and QR codes.

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3, 640)
cap.set(4, 480)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    code = decode(frame)
    for barcode in code:
        my_data = barcode.data.decode('utf-8')
        barcode_type = barcode.type
        print(barcode_type)
        print(my_data)
        points = np.array([barcode.polygon], np.int32)
        points = points.reshape((-1, 1, 2))
        cv2.polylines(frame, [points], True, (255, 0, 255), 5)
        rectangle_coordinates = barcode.rect
        cv2.putText(frame, my_data, (rectangle_coordinates[0], rectangle_coordinates[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)

    cv2.imshow('Result', frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
