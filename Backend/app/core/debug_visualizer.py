import cv2

CLASS_COLORS = {
    "1509": (0, 255, 0),
    "IRRI-6": (255, 0, 0),
    "Super_White": (0, 0, 255),
    "Unknown": (0, 255, 255)
}


def draw_predictions(image, boxes, predictions, save_path):

    img = image.copy()

    for box, pred in zip(boxes, predictions):

        x1, y1, x2, y2 = box

        label = pred["class"]
        conf = pred["confidence"]

        color = CLASS_COLORS.get(label, (255, 255, 255))

        # draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        text = f"{label} {conf:.2f}"

        # prevent text going outside image
        text_y = max(15, y1 - 5)

        cv2.putText(
            img,
            text,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )

    cv2.imwrite(save_path, img)