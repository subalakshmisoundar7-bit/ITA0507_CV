import cv2
import os


def get_component_roi(image):
    """
    Extract the main metal component region.
    ROI coordinates can be adjusted for the dataset.
    """

    height, width = image.shape

    # Current image: focus on the central/lower component
    x1 = 70
    y1 = 230
    x2 = 570
    y2 = 635

    roi = image[y1:y2, x1:x2]

    return roi


def segment_defects(image_path):

    os.makedirs("outputs", exist_ok=True)

    # 1. Load enhanced image
    image = cv2.imread(
        image_path,
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:
        raise ValueError(
            "Enhanced image could not be loaded"
        )

    # 2. Extract component ROI
    roi = get_component_roi(image)

    # Save ROI
    cv2.imwrite(
        "outputs/component_roi.jpg",
        roi
    )

    # 3. Smooth image
    blurred = cv2.GaussianBlur(
        roi,
        (5, 5),
        0
    )

    # 4. Edge detection
    edges = cv2.Canny(
        blurred,
        50,
        150
    )

    # 5. Threshold
    threshold = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        7
    )

    # 6. Morphological cleaning
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5)
    )

    cleaned = cv2.morphologyEx(
        threshold,
        cv2.MORPH_OPEN,
        kernel
    )

    cleaned = cv2.morphologyEx(
        cleaned,
        cv2.MORPH_CLOSE,
        kernel
    )

    # 7. Find contours
    contours, _ = cv2.findContours(
        cleaned,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    result = cv2.cvtColor(
        roi,
        cv2.COLOR_GRAY2BGR
    )

    features = []

    # 8. Extract candidate defects
    for contour in contours:

        area = cv2.contourArea(contour)

        # Filter very small regions
        if area < 100:
            continue

        # Filter very large regions
        if area > 5000:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        if w < 8 or h < 8:
            continue

        aspect_ratio = w / h

        if aspect_ratio > 5 or aspect_ratio < 0.2:
            continue

        perimeter = cv2.arcLength(
            contour,
            True
        )

        if perimeter == 0:
            continue

        circularity = (
            4 * 3.14159 * area
        ) / (perimeter * perimeter)

        defect_id = len(features) + 1

        # Draw candidate
        cv2.rectangle(
            result,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        cv2.putText(
            result,
            f"Candidate {defect_id}",
            (x, max(y - 5, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1
        )

        features.append({
            "id": defect_id,
            "area": area,
            "perimeter": perimeter,
            "width": w,
            "height": h,
            "aspect_ratio": aspect_ratio,
            "circularity": circularity
        })

    # 9. Save processing stages
    cv2.imwrite(
        "outputs/roi_edges.jpg",
        edges
    )

    cv2.imwrite(
        "outputs/roi_threshold.jpg",
        threshold
    )

    cv2.imwrite(
        "outputs/roi_segmented.jpg",
        cleaned
    )

    cv2.imwrite(
        "outputs/defect_detection.jpg",
        result
    )

    return features


if __name__ == "__main__":

    features = segment_defects(
        "outputs/enhanced.jpg"
    )

    print("================================")
    print("       METALVISION AI")
    print("    MODULE 2 - DEFECT ANALYSIS")
    print("================================")

    print("✓ Component ROI extracted")
    print("✓ Edge detection completed")
    print("✓ Thresholding completed")
    print("✓ Morphological cleaning completed")
    print("✓ Feature extraction completed")
    print("--------------------------------")

    print(
        "Potential candidates:",
        len(features)
    )

    for feature in features:

        print(
            f"Candidate {feature['id']}: "
            f"Area={feature['area']:.1f}, "
            f"Perimeter={feature['perimeter']:.1f}, "
            f"Aspect={feature['aspect_ratio']:.2f}, "
            f"Circularity={feature['circularity']:.2f}"
        )

    print("--------------------------------")
    print("Module 2 completed")
    print("================================")