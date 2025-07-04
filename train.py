from datasets import load_dataset, Audio
from transformers import AutoProcessor, DiaForConditionalGeneration
from transformers.training_args import TrainingArguments
from transformers.trainer import Trainer
import torch

class DiaDataCollator:
    """Pads & tensorises text (+ speaker tags) and audio, sets -100 where labels are PAD."""
    def __init__(self, processor, text_column="text"):
        self.processor = processor
        self.text_column = text_column
        self.pad_id = processor.tokenizer.pad_token_id

    def __call__(self, features):
        audios = []
        texts = []
        for f in features:
            grouped_text = ""
            for id, text in zip(f['speaker_ids'], f['texts']):
                grouped_text += f"[S{id + 1}] {text} "
            texts.append(grouped_text)
            audios.append(f["audio"]["array"])
        
        proc_out = self.processor(
            audio=audios,
            text=texts,
            generation=False,
            output_labels=True,
            padding=True,
            return_tensors="pt",
        )
        proc_out["labels"][proc_out["labels"] == self.pad_id] = -100
        return proc_out

def main():
    torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_checkpoint = "nari-labs/Dia-1.6B-0626"
    dataset_name = "eustlb/dailytalk-conversations-grouped"

    ds = load_dataset(dataset_name, split="train")

    ds = ds.cast_column("audio", Audio(sampling_rate=44_100))

    processor = AutoProcessor.from_pretrained(model_checkpoint)

    data_collator = DiaDataCollator(processor)

    model = DiaForConditionalGeneration.from_pretrained(model_checkpoint)
    model = model.to(torch_device)

    training_args = TrainingArguments(
        output_dir="./dia-finetuned-elise",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=1e-5,
        lr_scheduler_type="cosine",
        warmup_steps=200,
        num_train_epochs=3,
        bf16=True,
        logging_steps=10,
        save_steps=100,
        remove_unused_columns=False,
        report_to="none",
        optim="adamw_bnb_8bit",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        data_collator=data_collator,
    )

    trainer.train()


if __name__ == "__main__":
    main()
