from scipy.spatial.distance import euclidean
from imutils import perspective
from imutils import contours
import numpy as np
import imutils
import cv2
import os

def show_images(images):
    for i, img in enumerate(images):
        cv2.imshow("images/image_" + str(i), img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def process_images(image_folder):
    # Get the list of image files in the folder
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for img_file in image_files:
        img_path = os.path.join(image_folder, img_file)
        
        # Read image and preprocess
        image = cv2.imread(img_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (9, 9), 0)
        edged = cv2.Canny(blur, 50, 100)
        edged = cv2.dilate(edged, None, iterations=1)
        edged = cv2.erode(edged, None, iterations=1)

        # Find contours
        cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        (cnts, _) = contours.sort_contours(cnts)
        cnts = [x for x in cnts if cv2.contourArea(x) > 100]

        # Reference object dimensions
        ref_object = cnts[0]
        box = cv2.minAreaRect(ref_object)
        box = cv2.boxPoints(box)
        box = np.array(box, dtype="int")
        box = perspective.order_points(box)
        (tl, tr, br, bl) = box
        dist_in_pixel = euclidean(tl, tr)
        dist_in_cm = 2
        pixel_per_cm = dist_in_pixel / dist_in_cm

        # Draw remaining contours
        for cnt in cnts:
            box = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(box)
            box = np.array(box, dtype="int")
            box = perspective.order_points(box)
            (tl, tr, br, bl) = box
            cv2.drawContours(image, [box.astype("int")], -1, (0, 0, 255), 2)
            mid_pt_horizontal = (tl[0] + int(abs(tr[0] - tl[0]) / 2), tl[1] + int(abs(tr[1] - tl[1]) / 2))
            mid_pt_verticle = (tr[0] + int(abs(tr[0] - br[0]) / 2), tr[1] + int(abs(tr[1] - br[1]) / 2))
            wid = euclidean(tl, tr) / pixel_per_cm
            ht = euclidean(tr, br) / pixel_per_cm

            # Print dimensions to console
            print(f"{img_file}: Width: {wid:.1f}cm, Height: {ht:.1f}cm")

            # Write dimensions to a txt file
            with open("dimensions.txt", "a") as f:
                f.write(f"{img_file}: Width: {wid:.1f}cm, Height: {ht:.1f}cm\n")

        # Save the processed image
        cv2.imwrite(os.path.join("processed_images", img_file), image)

if __name__ == "__main__":
    image_folder_path = "images"  # Change this to the path of your image folder
    if not os.path.exists("processed_images"):
        os.makedirs("processed_images")

    process_images(image_folder_path)
