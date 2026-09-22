import os
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

import time
import argparse
import numpy as np
import cv2
from glob import glob
import tensorflow as tf
import tensorflow_advanced_segmentation_models as tasm


def get_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained semantic segmentation model."
    )
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--img_size", type=int, default=320)
    parser.add_argument("--model_name", type=str, default="UNet")
    parser.add_argument("--backbone", type=str, default="ResNet50")
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--output_dir", type=str, default="results/test_predictions")
    return parser.parse_args()


def load_image_mask(img_path, mask_path, size, num_classes):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found or could not be read: {img_path}")
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR) / 255.0

    mask = cv2.imread(mask_path, 0)
    if mask is None:
        raise FileNotFoundError(f"Mask not found or could not be read: {mask_path}")
    mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
    mask = (mask > 0).astype(np.uint8)
    mask = tf.keras.utils.to_categorical(mask, num_classes)

    return img.astype(np.float32), mask.astype(np.float32)


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

    val_imgs = sorted(glob(os.path.join(args.dataset, "val/images/*")))
    val_masks = sorted(glob(os.path.join(args.dataset, "val/masks/*")))

    if not val_imgs or not val_masks:
        raise RuntimeError("No validation images/masks were found.")

    if len(val_imgs) != len(val_masks):
        raise ValueError(
            f"Number of validation images ({len(val_imgs)}) and masks "
            f"({len(val_masks)}) do not match."
        )

    print("Validation samples:", len(val_imgs))

    model = build_model(
        args.model_name, args.backbone, args.img_size, args.num_classes
    )

    # The loss/metrics are compiled so that custom objects used by the
    # segmentation library are registered with the model.
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss=(
            tasm.losses.CategoricalFocalLoss()
            + tasm.losses.DiceLoss()
        ),
        metrics=[
            tasm.metrics.IOUScore(threshold=0.5),
            tasm.metrics.FScore(threshold=0.5),
            tasm.metrics.Precision(),
            tasm.metrics.Recall()
        ]
    )

    model.load_weights(args.model)

    iou_metric = tasm.metrics.IOUScore(threshold=0.5)
    f1_metric = tasm.metrics.FScore(threshold=0.5)
    precision_metric = tasm.metrics.Precision()
    recall_metric = tasm.metrics.Recall()

    iou_total = f1_total = precision_total = recall_total = 0.0

    os.makedirs(args.output_dir, exist_ok=True)

    start_inf = time.time()

    for img_path, mask_path in zip(val_imgs, val_masks):
        img, mask = load_image_mask(
            img_path, mask_path, args.img_size, args.num_classes
        )

        pred = model.predict(np.expand_dims(img, axis=0), verbose=0)[0]

        y_true = np.expand_dims(mask, axis=0)
        y_pred = np.expand_dims(pred, axis=0)

        iou_total += float(iou_metric(y_true, y_pred).numpy())
        f1_total += float(f1_metric(y_true, y_pred).numpy())
        precision_total += float(precision_metric(y_true, y_pred).numpy())
        recall_total += float(recall_metric(y_true, y_pred).numpy())

        pred_mask = np.argmax(pred, axis=-1).astype(np.uint8)
        save_path = os.path.join(args.output_dir, os.path.basename(img_path))
        cv2.imwrite(save_path, pred_mask * 255)

    inference_time = (time.time() - start_inf) / len(val_imgs)
    count = len(val_imgs)

    iou = iou_total / count
    f1 = f1_total / count
    precision = precision_total / count
    recall = recall_total / count

    print("\n===== TEST / VALIDATION RESULTS =====")
    print(f"Precision   : {precision:.4f}")
    print(f"Recall      : {recall:.4f}")
    print(f"F1-score    : {f1:.4f}")
    print(f"IoU         : {iou:.4f}")
    print(f"Inference   : {inference_time * 1000:.2f} ms/image")
    print(f"Predictions : {args.output_dir}")


if __name__ == "__main__":
    main()
