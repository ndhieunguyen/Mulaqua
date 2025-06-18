from transformers import set_seed
import os
from transformers import TrainerCallback
import warnings

warnings.filterwarnings("ignore")


class FileLoggerCallback(TrainerCallback):
    def __init__(self, args, file_path, output_dir):
        """
        Initialize the callback with a specified file path for logging.
        """
        self.file_path = file_path
        self.last_logged_epoch = -1
        self.args = args
        # if not os.path.exists(os.path.dirname(file_path)):
        #     os.makedirs(os.path.dirname(file_path))

        with open(self.file_path, "a") as f:
            f.write(f"Save to output dir: {output_dir}\n")

        with open(self.file_path, "a") as f:
            f.write(
                f"dropout: {args.dropout}, learning_rate: {args.learning_rate}, batch_size: {args.per_device_train_batch_size}, num_train_epochs: {args.num_train_epochs}, weight_decay: {args.weight_decay}, warmup_ratio: {args.warmup_ratio}, weight_decay: {args.weight_decay}\n"
            )

    def on_log(self, args, state, control, logs=None, **kwargs):
        """
        Called when logging occurs in the Trainer.
        """
        if (
            logs
            and self.last_logged_epoch < float(logs["epoch"])
            and float(logs["epoch"]) - int(logs["epoch"]) < 0.1
        ):
            with open(self.file_path, "a") as f:
                f.write(f"{logs}\n")
            self.last_logged_epoch = float(logs["epoch"])


def main(args):
    set_seed(args.seed)
    print(f"Seed set to {args.seed}")

    os.environ["WANDB_PROJECT"] = f"Bitter_{os.path.basename(args.data_folder)}"

    if args.prepare_dataset:
        from src.data import prepare_dataset

        prepare_dataset(args)
        print("Dataset prepared")

    else:
        from src.model import prepare_model, prepare_tokenizer
        from datasets import load_from_disk
        from src.utils import compute_metrics
        from transformers import DataCollatorWithPadding, TrainingArguments, Trainer
        import json

        run_name = f"{args.chosen_feature}_{args.fold}_of_{args.k_folds}_{args.dropout}"
        output_dir = os.path.join(
            args.output_dir,
            args.chosen_feature
            + f"_{args.pretrained_name.split('/')[-1].replace('-', '_')}",
            "dropout_" + str(args.dropout),
            "fold_" + str(args.fold),
        )
        print(f"Save to output dir: {output_dir}")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(os.path.join(output_dir, "args.json"), "w") as f:
            json.dump(args.__dict__, f)

        tokenizer = prepare_tokenizer(args)
        model = prepare_model(args)

        if args.k_folds == args.fold:
            data_path = os.path.join(
                args.data_folder,
                f"dataset_{args.swin_used}_{args.chosen_feature}",
            )
        else:
            data_path = os.path.join(
                args.data_folder,
                f"fold_{args.fold}",
                f"dataset_{args.swin_used}_{args.chosen_feature}",
            )
        dataset = load_from_disk(data_path)
        tokenized_dataset = dataset.map(
            lambda x: tokenizer(x["text"], truncation=True, padding=True),
            batched=True,
        ).shuffle(args.seed)
        data_collator = DataCollatorWithPadding(tokenizer)

        training_args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=args.learning_rate,
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            num_train_epochs=args.num_train_epochs,
            weight_decay=args.weight_decay,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=False,
            push_to_hub=args.push_to_hub,
            warmup_ratio=args.warmup_ratio,
            run_name=run_name,
            report_to=args.report_to,
            save_total_limit=args.save_total_limit,
            seed=args.seed,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=(
                tokenized_dataset["test"]
                if args.k_folds == args.fold
                else tokenized_dataset["val"]
            ),
            processing_class=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )
        print(f"Logging to {args.log_file}")
        trainer.add_callback(FileLoggerCallback(args, args.log_file, output_dir))
        trainer.train()
        trainer.save_model(output_dir)
        metrics = trainer.evaluate()
        metrics_path = os.path.join(output_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f)


if __name__ == "__main__":
    import argparse
    from config import config

    parser = argparse.ArgumentParser()
    for k, v in config.__dict__.items():
        if type(v) in [str, int, float]:
            parser.add_argument(f"--{k}", type=type(v), default=v)
        elif type(v) == bool:
            parser.add_argument(f"--{k}", action="store_false" if v else "store_true")
        elif type(v) == list:
            parser.add_argument(f"--{k}", nargs="*", type=type(v[0]), default=v)

    args = parser.parse_args()
    main(args)
