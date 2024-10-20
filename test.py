import os
import cv2
import typer
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, Colors


def main(
    model: str,
    image: str,
    line_width: int = 2,
    font_size: int = 8,
    show: bool = False,
    output_dir: str = None,
    save_format: str = "jpg"
):
    if not os.path.isfile(image):
        raise ValueError(f"Image file not found: {image}")

    # Validate save format
    valid_formats = ['jpg', 'png']
    if save_format not in valid_formats:
        raise ValueError(f"Invalid save format '{save_format}'. Supported formats: {valid_formats}")

    try:
        img = cv2.imread(image, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Error loading image. Check the file format and path.")
        
        model = YOLO(model)
        result = model.predict(img)[0]
        
        height, width = result.orig_shape[:2]
        colors = Colors()

        # Annotate the image
        annotator = Annotator(img, line_width=line_width, font_size=font_size)
        for label, box in zip(result.boxes.cls.tolist(), result.boxes.xyxyn.tolist()):
            label = int(label)
            annotator.box_label(
                [box[0] * width, box[1] * height, box[2] * width, box[3] * height],
                result.names[label],
                color=colors(label, bgr=True),
            )

        # Determine the output path
        if output_dir is None:
            output_dir = os.path.dirname(image)
        os.makedirs(output_dir, exist_ok=True)

        output_path = os.path.join(
            output_dir, f"annotated-{os.path.splitext(os.path.basename(image))[0]}.{save_format}"
        )
        annotator.save(output_path)

        print(f"Annotated image saved at {output_path}")

        if show:
            cv2.imshow("Annotated Image", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    except Exception as e:
        print(f"Error during processing: {e}")


if __name__ == "__main__":
    typer.run(main)
