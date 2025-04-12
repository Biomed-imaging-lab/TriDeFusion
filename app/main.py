import os
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from app.worker import process_image_task
from app.utils import save_to_s3
import uuid

app = FastAPI()

UPLOAD_DIR = "/tmp/uploaded_images"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.post("/denoise/")
def denoise_image(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        buffer.write(file.file.read())

    task = process_image_task.delay(file_path)
    
    # Publish event to Kafka
    producer.send("image_processing_started", {"task_id": task.id, "file_path": file_path})
    
    return {"status": "processing", "task_id": task.id}

@app.get("/result/{task_id}")
def get_result(task_id: str):
    result = AsyncResult(task_id)
    if result.state == "SUCCESS":
        return {"status": "completed", "processed_image_path": result.result}
    return {"status": result.state}

@app.post("/process/")
async def process_image(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    file_id = str(uuid.uuid4())  # Unique file ID
    file_path = f"/tmp/{file_id}.png"

    with open(file_path, "wb") as buffer:
        buffer.write(file.file.read())

    # Upload original image to S3
    save_to_s3(file_path, f"original/{file_id}.png")

    # Send processing task to Celery
    task = process_image_task.delay(file_path, file_id)

    return {"status": "processing", "task_id": task.id, "file_id": file_id}

@app.get("/result/{task_id}")
async def get_result(task_id: str):
    from celery.result import AsyncResult
    result = AsyncResult(task_id)

    if result.state == "SUCCESS":
        return {"status": "completed", "processed_image_url": result.result}
    return {"status": result.state}
