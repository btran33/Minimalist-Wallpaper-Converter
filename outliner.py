import cv2
import matplotlib.pyplot as plt

class Sketch:
    def __init__(self, img_path: str, dest_path: str, gauss_blur=(27,27)) -> None:   
        self.image = cv2.cvtColor(
                        cv2.imread(img_path),
                        cv2.COLOR_BGR2RGB
                    )
        
        self.image = cv2.fastNlMeansDenoisingColored(self.image, None, 11, 6, 7, 21)
        self.gauss_blur = gauss_blur
        self.dest_path = dest_path

        self.outlineImg()
        
    
    def outlineImg(self) -> None:
        greyImg = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        invert = cv2.bitwise_not(greyImg)
        blur = cv2.GaussianBlur(invert, self.gauss_blur, cv2.BORDER_DEFAULT)
        invertedBlur = cv2.bitwise_not(blur)
        self.outlined = cv2.divide(greyImg, invertedBlur, scale=256.0)

    def show(self):
        plt.figure(figsize = (20,20))
        plt.imshow(self.outlined, cmap='gray')

    def save(self):
        cv2.imwrite(self.dest_path, self.outlined)