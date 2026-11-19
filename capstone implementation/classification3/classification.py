import os
import json
from segmentation2.segmentation import segment_defects


def classify_candidate(feature):

    area = feature["area"]
    aspect_ratio = feature["aspect_ratio"]
    circularity = feature["circularity"]

    # Prototype classification
    if aspect_ratio > 2.5 or aspect_ratio < 0.4:
        defect_type = "Scratch"

    elif circularity < 0.25 and area > 500:
        defect_type = "Surface Irregularity"

    elif area < 300:
        defect_type = "Minor Anomaly"

    else:
        defect_type = "Potential Defect"

    # Severity
    if area > 2000:
        severity = "High"
    elif area > 800:
        severity = "Medium"
    else:
        severity = "Low"

    return defect_type, severity


def generate_report(features):

    results = []

    for feature in features:

        defect_type, severity = classify_candidate(feature)

        result = {
            "candidate_id": feature["id"],
            "classification": defect_type,
            "severity": severity,
            "area": round(feature["area"], 2),
            "perimeter": round(feature["perimeter"], 2),
            "width": feature["width"],
            "height": feature["height"],
            "aspect_ratio": round(
                feature["aspect_ratio"], 2
            ),
            "circularity": round(
                feature["circularity"], 3
            )
        }

        results.append(result)

    return results


if __name__ == "__main__":

    print("================================")
    print("       METALVISION AI")
    print(" MODULE 3 - CLASSIFICATION")
    print("================================")

    # Module 2 automatically provides features
    features = segment_defects(
        "outputs/enhanced.jpg"
    )

    print("Candidates received:", len(features))

    # Classify all candidates
    results = generate_report(features)

    # Create reports folder
    os.makedirs("reports", exist_ok=True)

    # Save JSON report
    report_path = "reports/inspection_report.json"

    with open(
        report_path,
        "w"
    ) as file:

        json.dump(
            results,
            file,
            indent=4
        )

    print("--------------------------------")
    print("Classification completed")

    for result in results:

        print(
            f"Candidate {result['candidate_id']} | "
            f"{result['classification']} | "
            f"Severity: {result['severity']}"
        )

    print("--------------------------------")
    print("Report saved:")
    print(report_path)
    print("================================")