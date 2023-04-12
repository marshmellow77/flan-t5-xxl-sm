import os
import argparse
import numpy as np
from transformers import (
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    AutoTokenizer,
    set_seed,
)
from datasets import load_from_disk
import torch
import evaluate
import nltk
import numpy as np
import time
import wandb

from huggingface_hub import HfFolder
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
import boto3

nltk.download("punkt", quiet=True)

# Metric
metric = evaluate.load("rouge")

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels


def parse_arge():
    """Parse the arguments."""
    parser = argparse.ArgumentParser()
    # add model id and dataset path argument
    parser.add_argument("--model_id", type=str, default="google/flan-t5-xl", help="Model id to use for training.")
    parser.add_argument("--train_dataset_path", type=str, help="Path to processed dataset stored by sageamker.")
    parser.add_argument("--test_dataset_path", type=str, help="Path to processed dataset stored by sageamker.")
    parser.add_argument(
        "--repository_id", type=str, default=None, help="Hugging Face Repository id for uploading models"
    )
    # add training hyperparameters for epochs, batch size, learning rate, and seed
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs to train for.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8, help="Batch size to use for training.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8, help="Batch size to use for testing.")
    parser.add_argument("--generation_max_length", type=int, default=140, help="Maximum length to use for generation")
    parser.add_argument("--generation_num_beams", type=int, default=4, help="Number of beams to use for generation.")
    parser.add_argument("--lr", type=float, default=3e-3, help="Learning rate to use for training.")
    parser.add_argument("--seed", type=int, default=42, help="Seed to use for training.")
    parser.add_argument("--deepspeed", type=str, default=None, help="Path to deepspeed config file.")
    parser.add_argument("--gradient_checkpointing", type=bool, default=True, help="Whether to use gradient checkpointing.")
    parser.add_argument(
        "--bf16",
        type=bool,
        default=True if torch.cuda.get_device_capability()[0] == 8 else False,
        help="Whether to use bf16.",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=HfFolder.get_token(),
        help="Token to use for uploading models to Hugging Face Hub.",
    )
    args = parser.parse_known_args()
    return args


def upload_directory_to_s3(bucket_name, s3_directory, local_directory):
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(bucket_name)
    
    for root, dirs, files in os.walk(local_directory):
        for file in files:
            local_file = os.path.join(root, file)
            s3_key = os.path.join(s3_directory, os.path.relpath(local_file, local_directory))
            
            print(f"Uploading {local_file} to {s3_key}")
            bucket.upload_file(local_file, s3_key)


def training_function(args):
    # set seed
    set_seed(args.seed)
    
    
    timestamp = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
    project_name = f"hf-sagemaker-flan-t5-base-{timestamp}"
    job_type='Training'

    wandb.init(name=f"hf-sm-ds-{os.environ['RANK']}", project=project_name, job_type=job_type, group="hf-sm-dist")
    wandb.run._label('sagemaker-hf')
    os.environ["WANDB_LOG_MODEL"] = "TRUE"  # Hugging Face Trainer will use this to log model weights to W&B
    

    # load dataset from disk and tokenizer
    train_dataset = load_from_disk(args.train_dataset_path)
    eval_dataset = load_from_disk(args.test_dataset_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    # load model from the hub
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_id,
        use_cache=False if args.gradient_checkpointing else True,  # this is needed for gradient checkpointing
        cache_dir = "/opt/ml/input/" # changed for SM to have enough storage space
    )

    # we want to ignore tokenizer pad token in the loss
    label_pad_token_id = -100
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer, model=model, label_pad_token_id=label_pad_token_id, pad_to_multiple_of=8
    )

    # Define compute metrics function
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        result = {k: round(v * 100, 4) for k, v in result.items()}
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        return result

    # Define training args
    # output_dir = os.environ["SM_OUTPUT_DATA_DIR"]
    output_dir = '/tmp'
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        predict_with_generate=True,
        generation_max_length=args.generation_max_length,
        generation_num_beams=args.generation_num_beams,
        fp16=False,  # T5 overflows with fp16
        bf16=args.bf16,  # Use BF16 if available
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        deepspeed=args.deepspeed,
        gradient_checkpointing=args.gradient_checkpointing,
        # logging & evaluation strategies
        logging_dir=f"{output_dir}/logs",
        logging_strategy="steps",
        logging_steps=500,
        evaluation_strategy="epoch",
        save_strategy="no",
        save_total_limit=2,
        load_best_model_at_end=False,
        # push to hub parameters
        # push_to_hub=True if args.repository_id else False,
        # hub_strategy="every_save",
        # hub_model_id=args.repository_id if args.repository_id else None,
        # hub_token=args.hf_token,
    )

    # Create Trainer instance
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Start training
    trainer.train()

    # Save our tokenizer and create model card
#     tokenizer.save_pretrained(output_dir)
#     trainer.create_model_card()
#     # Push the results to the hub
#     if args.repository_id:
#         trainer.push_to_hub()

#     # Saves the model to s3 uses os.environ["SM_MODEL_DIR"] to make sure checkpointing works
#     trainer.save_model(os.environ["SM_MODEL_DIR"])
#     tokenizer.save_pretrained(os.environ["SM_MODEL_DIR"])

    save_model_dir = '/tmp/output/asset/'
    tokenizer.save_pretrained(save_model_dir)
    trainer.save_model(save_model_dir)
    
    # os.system("chmod +x ./scripts/s5cmd")
    # os.system("./scripts/s5cmd sync {0} {1}".format(save_model_dir, os.environ['MODEL_S3_PATH']))
    
    # Usage
    bucket_name = os.environ['MODEL_S3_BUCKET']
    s3_directory = os.environ['MODEL_S3_DIR']
    local_directory = save_model_dir

    upload_directory_to_s3(bucket_name, s3_directory, local_directory)

def main():
    args, _ = parse_arge()
    training_function(args)


if __name__ == "__main__":
    main()
