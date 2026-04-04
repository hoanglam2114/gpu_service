"""
services/training.py — LoRA fine-tuning với unsloth + SFTTrainer.
"""
import os
import shutil
import threading
import time

from datasets import load_dataset
from huggingface_hub import HfApi, login, snapshot_download
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import FastLanguageModel, is_bfloat16_supported

import app.state as state
from app.config import config
from app.callbacks import FlaskProgressCallback, EnhancedWatchdogCallback
from app.services.gpu import release_gpu_memory
from app.services.evaluation import formatting_prompts_func


def background_train_task(job_id: str, train_config: dict, filepath: str | None, hf_token: str | None) -> None:
    state.jobs_db[job_id]["status"] = "TRAINING"
    local_job_dir = os.path.join(config.LOCAL_CHECKPOINT_BASE, job_id)
    hf_repo_id = train_config.get("hf_repo_id")

    try:
        if hf_token:
            print(f"🔑 Logging into Hugging Face for job {job_id}...")
            login(token=hf_token)

            if hf_repo_id and "/" not in hf_repo_id:
                api = HfApi()
                user_info = api.whoami(token=hf_token)
                username = user_info["name"]
                hf_repo_id = f"{username}/{hf_repo_id}"
                print(f"📝 Updated repo ID to: {hf_repo_id}")

            if hf_repo_id and train_config.get("push_to_hub"):
                from huggingface_hub import create_repo
                try:
                    create_repo(repo_id=hf_repo_id, token=hf_token, repo_type="model", exist_ok=True)
                    print(f"✅ Repository {hf_repo_id} is ready.")
                except Exception as e:
                    print(f"⚠️ Warning creating repo: {e}")

        os.makedirs(local_job_dir, exist_ok=True)
        release_gpu_memory()

        # 1. Load model
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=train_config["model_name"],
            max_seq_length=train_config["modelMaxLength"],
            load_in_4bit=True,
        )
        if getattr(tokenizer, "pad_token", None) is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        # 2. Apply LoRA
        model = FastLanguageModel.get_peft_model(
            model,
            r=train_config["r"],
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_alpha=train_config["lora_alpha"],
            lora_dropout=train_config["lora_dropout"],
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=train_config["random_state"],
        )

        # 3. Load dataset
        if filepath:
            ext = os.path.splitext(filepath)[1]
            dataset = load_dataset("json" if "json" in ext else "csv", data_files=filepath, split="train")
        elif train_config.get("dataset_hf_id"):
            dataset = load_dataset(train_config["dataset_hf_id"], split="train")
        else:
            raise ValueError("No dataset source provided.")

        dataset = dataset.train_test_split(test_size=0.1, seed=42)
        dataset_train = dataset["train"].map(formatting_prompts_func, batched=True)
        dataset_eval = dataset["test"].map(formatting_prompts_func, batched=True)

        # 4. Resume from checkpoint on HF Hub
        resume_from = None
        if hf_repo_id:
            api = HfApi()
            try:
                repo_files = api.list_repo_files(repo_id=hf_repo_id, token=hf_token)
                if any("checkpoint-" in f for f in repo_files):
                    print(f"[🔄] Found checkpoints on HF Hub. Downloading...")
                    snapshot_download(
                        repo_id=hf_repo_id,
                        local_dir=local_job_dir,
                        token=hf_token,
                        allow_patterns=["checkpoint-*/*", "*.json"],
                    )
                    checkpoints = [d for d in os.listdir(local_job_dir) if d.startswith("checkpoint-")]
                    if checkpoints:
                        latest = max(checkpoints, key=lambda x: int(x.split("-")[1]))
                        resume_from = os.path.join(local_job_dir, latest)
                        print(f"[✅] Resuming from: {resume_from}")
            except Exception as e:
                print(f"[*] No checkpoints or error: {e}")

        # 5. Trainer
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset_train,
            dataset_text_field="text",
            eval_dataset=dataset_eval,
            max_seq_length=train_config["modelMaxLength"],
            args=TrainingArguments(
                per_device_train_batch_size=train_config["batchSize"],
                gradient_accumulation_steps=train_config["gradient_accumulation_steps"],
                warmup_steps=train_config["warmup_steps"],
                num_train_epochs=train_config["epochs"],
                learning_rate=train_config["learningRate"],
                fp16=not is_bfloat16_supported(),
                bf16=is_bfloat16_supported(),
                logging_steps=1,
                optim=train_config["optim"],
                weight_decay=train_config["weight_decay"],
                lr_scheduler_type=train_config["lr_scheduler_type"],
                seed=train_config["seed"],
                output_dir=local_job_dir,
                save_strategy="steps",
                save_steps=50,
                save_total_limit=1,
                report_to="none",
                push_to_hub=train_config.get("push_to_hub", False),
                hub_model_id=hf_repo_id,
                hub_token=hf_token,
                hub_strategy="checkpoint",
            ),
            callbacks=[FlaskProgressCallback(job_id), EnhancedWatchdogCallback(job_id)],
        )

        # 6. Train
        trainer.train(resume_from_checkpoint=resume_from)

        # 7. Final push
        if train_config.get("push_to_hub") and hf_repo_id:
            print(f"🎉 Pushing final model to {hf_repo_id}...")
            model.push_to_hub(hf_repo_id, token=hf_token, commit_message="End of training push")
            tokenizer.push_to_hub(hf_repo_id, token=hf_token, commit_message="End of training push")
            state.jobs_db[job_id].update({"status": "COMPLETED", "progress": 100, "final_path": f"hf://{hf_repo_id}"})
        else:
            state.jobs_db[job_id].update({"status": "COMPLETED", "progress": 100})

    except Exception as e:
        print(f"[❌] Training error: {e}")
        state.jobs_db[job_id].update({"status": "ERROR", "error": str(e)})

    finally:
        if os.path.exists(local_job_dir):
            shutil.rmtree(local_job_dir)
        state.active_training_jobs.discard(job_id)
        print(f"[INFO] Job {job_id} finished. Worker free.")
        release_gpu_memory()


def job_manager_thread() -> None:
    """Vòng lặp worker xử lý job queue."""
    while True:
        if state.job_queue and len(state.active_training_jobs) < config.MAX_CONCURRENT_JOBS:
            job_id, train_config, file_path, hf_token = state.job_queue.popleft()

            if job_id in state.active_training_jobs:
                continue

            state.active_training_jobs.add(job_id)
            state.jobs_db[job_id] = {"status": "PENDING", "progress": 0, "logs": []}
            print(f"[INFO] Starting Train Job {job_id}.")

            t = threading.Thread(target=background_train_task, args=(job_id, train_config, file_path, hf_token))
            t.daemon = True
            t.start()

        time.sleep(3)
