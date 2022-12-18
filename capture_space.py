from attr import attrs, define, field
import cv2
import pickle


@define
class SpaceRecognition:
    path_to_image: str

    def __get_all_parking(self) -> list:
        """Save the positions in a pkl file"""
        image = cv2.imread(self.path_to_image)
        spaces: list = list()
        for marks in range(69):  # 69 positions at image
            space = cv2.selectROI('mark the spaces', image, False)
            cv2.destroyWindow('mark the spaces')
            spaces.append(space)

            for x, y, width, height in spaces:
                cv2.rectangle(image, (x, y), (x + width, y + height), (0, 0, 255), 3)
        return spaces

    def save_to_pickle_file(self) -> None:
        spaces = self.__get_all_parking()
        with open('positions.pkl', 'wb') as file:
            pickle.dump(spaces, file)


if __name__ == '__main__':
    sr = SpaceRecognition('files/parking.png')
    sr.save_to_pickle_file()
