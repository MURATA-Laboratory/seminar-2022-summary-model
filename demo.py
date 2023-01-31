import time

import numpy as np
import pytorch_lightning as pl
import torch
from box import Box
from transformers import T5ForConditionalGeneration, T5Tokenizer

MODEL_PATH = "./epoch=9.ckpt"


config = dict(
    wandb_project_name="summary",
    pretrained_model_name="sonoisa/t5-base-japanese",
    seed=40,
    data_module=dict(
        batch_size=2,
        text_max_length=30,  # データセットの入力テキストは21~25字
        summary_max_length=17,  # 20字を超えないようにxトークンとする
    ),
)

config = Box(config)


tokenizer = T5Tokenizer.from_pretrained(config.pretrained_model_name)


class MyModel(pl.LightningModule):
    def __init__(self, tokenizer, config):
        super().__init__()
        self.tokenizer = tokenizer
        self.config = config

        self.model = T5ForConditionalGeneration.from_pretrained(
            config.pretrained_model_name,
            return_dict=True,
        )


trained_model = MyModel(
    tokenizer,
    config=config,
)
trained_model.load_state_dict(
    torch.load(MODEL_PATH, map_location=torch.device("cpu"))["state_dict"]
)
trained_model.eval()
trained_model.freeze()

while True:
    text = input("Text (exit): ")
    if text == "exit":
        break

    t0 = time.time()
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=config.data_module.text_max_length,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    generated_ids = trained_model.model.generate(
        input_ids=encoding["input_ids"],
        attention_mask=encoding["attention_mask"],
        max_length=config.data_module.summary_max_length,
        num_beams=4,
        repetition_penalty=2.5,
        length_penalty=1.0,
        # early_stopping=True,
    )
    print("    Time: ", time.time() - t0)
    print(f"    {tokenizer.batch_decode(generated_ids, skip_special_tokens=True)}")
