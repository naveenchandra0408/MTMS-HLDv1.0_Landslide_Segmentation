import os
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

import argparse
import numpy as np
import cv2
import tensorflow as tf
import tensorflow_advanced_segmentation_models as tasm


def get_args():
    parser = argparse.ArgumentParser(
        description="Generate a landslide segmentation mask for an input image."
    )
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--img_size", type=int, default=320)
    parser.add_argument("--model_name", type=str, default="UNet")
    parser.add_argument("--backbone", type=str, default="ResNet50")
    parser.add_argument("--num_classes", type=int, default=2)
    return parser.parse_args()


def build_model(model_name, backbone, img_size, num_classes):
    base_model, layers, _ = tasm.create_base_model(
        name=backbone,
        weights="imagenet",
        height=img_size,
        width=img_size
    )

    ModelClass = getattr(tasm, model_name)

    if model_name == "HRNetOCR":
        return ModelClass(
            n_classes=num_classes,
            height=img_size,
            width=img_size
        ).model()

    return ModelClass(
        n_classes=num_classes,
        base_model=base_model,
        output_layers=layers,
        backbone_trainable=True
    ).model()


def main():
    args = get_args()

    img = cv2.imread(args.input)
    if img is None:
        raise FileNotFoundError(
            f"Input image not found or could not be read: {args.input}"
        )

    original_size = (img.shape[1], img.shape[0])

    resized = cv2.resize(
        img,
        (args.img_size, args.img_size),
        interpolation=cv2.INTER_LINEAR
    )
    resized = resized.astype(np.float32) / 255.0

    model = build_model(
        args.model_name, args.backbone, args.img_size, args.num_classes
    )

    model.load_weights(args.model)

    prediction = model.predict(
        np.expand_dims(resized, axis=0),
        verbose=0
    )[0]

    pred_mask = np.argmax(prediction, axis=-1).astype(np.uint8)

    # Resize the binary prediction back to the original image dimensions.
    pred_mask = cv2.resize(
        pred_mask,
        original_size,
        interpolation=cv2.INTER_NEAREST
    )

    pred_mask = pred_mask * 255

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    cv2.imwrite(args.output, pred_mask)

    print("Prediction completed.")
    print(f"Input      : {args.input}")
    print(f"Model      : {args.model}")
    print(f"Output     : {args.output}")


if __name__ == "__main__":
    main()
