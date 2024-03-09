import cv2
from convexity import Convexity

object_a_intensity = 0
object_b_intensity = 255
vector = [1, 0]

img = cv2.imread('input/apple.png', cv2.IMREAD_GRAYSCALE)
convexity_calculator = Convexity(img, verbose=False, logname='output/apple.log', feature_image='output/apple.png')
vals = convexity_calculator.interlacement(object_a_intensity, object_b_intensity, vector)
print(vals)
