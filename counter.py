import cv2
import pickle
import numpy as np
from typing import Any


def view(content: Any) -> None:
    cv2.imshow('video', content)
    cv2.waitKey(5)


def read_pickle_file(file_pkl: str) -> Any:
    spaces: list
    with open(file_pkl, 'rb') as file:
        spaces = pickle.load(file)
        return spaces


def monitore(video_path: str) -> None:
    try:
        spaces: Any = read_pickle_file('positions.pkl')
        video = cv2.VideoCapture(video_path)
        while True:
            ctn_space: int = 0
            check, image = video.read()
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            threshed = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 16)
            median_blur = cv2.medianBlur(threshed, 5)
            kernel = np.ones((3, 3), np.int8)
            dilate = cv2.dilate(median_blur, kernel)  # if a car get out, the most pixels are black

            # view(gray_image)
            for x, y, width, height in spaces:
                space = dilate[y:y+height, x:x+width]
                count = cv2.countNonZero(space)  # black spaces

                # put text on screen
                cv2.rectangle(image, (x, y), (x+width, y+height), (100, 200, 300), 2)

                if count < 900:
                    cv2.rectangle(image, (x, y), (x + width, y + height), (0, 20, 255), 2)
                    ctn_space += 1
                    print(ctn_space)
                else:
                    cv2.rectangle(image, (x, y), (x + width, y + height), (100, 0, 30), 2)
            cv2.putText(image, f"VAGAS: {str(ctn_space)}", (100, 50), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 255), 2)
            view(image)
    except Exception as e:
        pass


def main(video_path: str) -> None:
    monitore(video_path)


if __name__ == '__main__':
    main('files/video.mp4')
