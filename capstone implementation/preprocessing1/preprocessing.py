import cv2
import os


def preprocess_image(image_path):

    # Create output folder
    os.makedirs("outputs", exist_ok=True)

    # 1. Load original image
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError("Image could not be loaded")

    # 2. Resize
    resized = cv2.resize(image, (640, 640))

    # 3. Convert to grayscale
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # 4. Remove noise
    denoised = cv2.medianBlur(gray, 5)

    # 5. Improve contrast
    enhanced = cv2.equalizeHist(denoised)

    # Save each processing stage
    cv2.imwrite("outputs/original.jpg", resized)
    cv2.imwrite("outputs/grayscale.jpg", gray)
    cv2.imwrite("outputs/denoised.jpg", denoised)
    cv2.imwrite("outputs/enhanced.jpg", enhanced)

    return {
        "original": resized,
        "grayscale": gray,
        "denoised": denoised,
        "enhanced": enhanced
    }


if __name__ == "__main__":

    image_path = "test_images/OIP.webp"

    results = preprocess_image(image_path)

    print("================================")
    print("       METALVISION AI")
    print("     MODULE 1 - PREPROCESSING")
    print("================================")
    print("✓ Image loaded")
    print("✓ Resized to 640 × 640")
    print("✓ Converted to grayscale")
    print("✓ Noise removed")
    print("✓ Contrast enhanced")
    print("--------------------------------")
    print("Processing completed successfully")
    print("================================")