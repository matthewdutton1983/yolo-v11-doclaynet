import typer
from ultralytics import YOLO


def main(
    base_model: str,
    datasets: str = "./datasets/data.yaml",
    epochs: int = 50,
    imgsz: int = 1024,
    batch: int = 16,
    dropout: float = 0.1,
    seed: int = 0,
    resume: bool = False,
    learning_rate: float = 1e-5,
    weight_decay: float = 0.01,
):
    try:
        from clearml import Task

        task = Task.init(
            project_name="yolo-doclaynet",
            task_name=f"fine-tuned-model-{base_model}-epochs-{epochs}-imgsz-{imgsz}-batch-{batch}",
            output_uri=True  # Track output artifacts
        )
    except ImportError:
        print("clearml not installed")

    model = YOLO(base_model)

    results = model.train(
        data=datasets,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        dropout=dropout,
        seed=seed,
        resume=resume,
        lr0=learning_rate,
        optimizer='AdamW',
        weight_decay=weight_decay,
        scheduler='cosine',
        callbacks=[task.get_logger()]
    )

    print(results)


if __name__ == "__main__":
    typer.run(main)
  
