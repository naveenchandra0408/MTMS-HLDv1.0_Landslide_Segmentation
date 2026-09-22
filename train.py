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
        description="Train a semantic segmentation model on the MTMS-HLD dataset."
    )
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--img_size", type=int, default=320)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--model_name", type=str, default="UNet")
    parser.add_argument("--backbone", type=str, default="ResNet50")
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--result_dir", type=str, default="results")
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


def data_generator(img_paths, mask_paths, batch_size, size, num_classes):
    if len(img_paths) != len(mask_paths):
        raise ValueError(
            f"Number of images ({len(img_paths)}) and masks ({len(mask_paths)}) do not match."
        )

    while True:
        idxs = np.arange(len(img_paths))
        np.random.shuffle(idxs)

        for i in range(0, len(idxs), batch_size):
            batch_idx = idxs[i:i + batch_size]
            imgs, masks = [], []

            for j in batch_idx:
                img, mask = load_image_mask(
                    img_paths[j], mask_paths[j], size, num_classes
                )
                imgs.append(img)
                masks.append(mask)

            yield np.asarray(imgs), np.asarray(masks)


def build_model(model_name, backbone, img_size, num_classes):
    base_model, layers, _ = tasm.create_base_model(
        name=backbone,
        weights="imagenet",
        height=img_size,
        width=img_size
    )

    ModelClass = getattr(tasm, model_name)

    if model_name == "HRNetOCR":
        model = ModelClass(
            n_classes=num_classes,
            height=img_size,
            width=img_size
        ).model()
    else:
        model = ModelClass(
            n_classes=num_classes,
            base_model=base_model,
            output_layers=layers,
            backbone_trainable=True
        ).model()

    return model


def main():
    args = get_args()

    os.makedirs(args.result_dir, exist_ok=True)

    train_imgs = sorted(glob(os.path.join(args.dataset, "train/images/*")))
    train_masks = sorted(glob(os.path.join(args.dataset, "train/masks/*")))
    val_imgs = sorted(glob(os.path.join(args.dataset, "val/images/*")))
    val_masks = sorted(glob(os.path.join(args.dataset, "val/masks/*")))

    if not train_imgs or not train_masks:
        raise RuntimeError("No training images/masks were found.")
    if not val_imgs or not val_masks:
        raise RuntimeError("No validation images/masks were found.")

    print("Train size:", len(train_imgs))
    print("Val size:", len(val_imgs))

    train_gen = data_generator(
        train_imgs, train_masks, args.batch_size, args.img_size, args.num_classes
    )
    val_gen = data_generator(
        val_imgs, val_masks, args.batch_size, args.img_size, args.num_classes
    )

    steps_per_epoch = max(1, int(np.ceil(len(train_imgs) / args.batch_size)))
    val_steps = max(1, int(np.ceil(len(val_imgs) / args.batch_size)))

    model = build_model(
        args.model_name, args.backbone, args.img_size, args.num_classes
    )

    num_params = model.count_params()
    num_layers = len(model.layers)

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

    best_model_path = os.path.join(args.result_dir, "best.keras")
    last_model_path = os.path.join(args.result_dir, "last.keras")

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            best_model_path,
            monitor="val_iou_score",
            mode="max",
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_iou_score",
            mode="max",
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
    ]

    print("Model created successfully")
    print("Starting training...")

    x, y = next(train_gen)
    print("Batch shape:", x.shape, y.shape)

    start_time = time.time()

    model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_gen,
        validation_steps=val_steps,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1
    )

    training_time = time.time() - start_time

    model.save(last_model_path)

    print("\n===== MODEL PERFORMANCE SUMMARY =====")
    print(f"Model            : {args.model_name}")
    print(f"Backbone         : {args.backbone}")
    print(f"Image Size       : {args.img_size} x {args.img_size}")
    print(f"Layers           : {num_layers}")
    print(f"Parameters       : {num_params:,}")
    print(f"Training Time    : {training_time / 60:.2f} minutes")
    print(f"Best Model       : {best_model_path}")
    print(f"Last Model       : {last_model_path}")


if __name__ == "__main__":
    main()
