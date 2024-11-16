import pandas as pd
from transformers import AutoTokenizer,AutoModelForSequenceClassification,AutoConfig,DataCollatorWithPadding
from datasets import Dataset
import os
from transformers import TrainingArguments, Trainer
import torch
import gc
import shutil
from sklearn.model_selection import KFold, GroupKFold
import random
import numpy as np
from typing import List


def seed_everything(seed:int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(seed=42)


class CFG:
    model_name="microsoft-deberta-v3-base"
    output_dir="microsoft-deberta-v3-base-finetuing"
    learning_rate=1e-5
    weight_decay=1e-8
    hidden_dropout_prob=0.
    attention_probs_dropout_prob=0.
    num_train_epochs=2
    n_splits=4
    batch_size=8
    random_seed=42
    save_steps=200
    max_length=1600
    folds=[0,1,2,3]


# train test数据
prompts_train = pd.read_csv('./prompts_train.csv')
prompts_test = pd.read_csv('./prompts_test.csv')
summaries_train = pd.read_csv('./summaries_train.csv')
summaries_test = pd.read_csv('./summaries_test.csv')

train = summaries_train.merge(prompts_train, how="left", on="prompt_id")
test = summaries_test.merge(prompts_test, how="left", on="prompt_id")

# fold
gkf = GroupKFold(n_splits=len(CFG.folds))
for i, (_, val_index) in enumerate(gkf.split(train, groups=train["prompt_id"])):
    train.loc[val_index, "fold"] = i


train.head()

def compute_mcrmse(eval_pred):
    preds, labels = eval_pred
    col_rmse = np.sqrt(np.mean((preds-labels)**2, axis=0))
    mcrmse = np.mean(col_rmse)

    return {
        "content_rmse": col_rmse[0],
        "wording_rmse": col_rmse[1],
        "mcrmse":mcrmse,
    }

# 预处理
# class Preprocessor:
#     def __init__(self, model_name:str):
#         self.model_name = model_name
#         self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

#     def run(self, prompts:pd.DataFrame, summaries:pd.DataFrame) -> pd.DataFrame:



eval
# 模型：train,predict
class ScoreRegressor:
    def __init__(self, 
                model_name:str,
                output_dir:str,
                inputs:List[str],
                target_cols:List[str],
                hidden_dropout_prob:float,
                attention_probs_dropout_prob:float,
                max_length:int,
                ):
                self.input_col = "input"
                self.input_text_cols = inputs
                self.target_cols = target_cols
                self.model_name = model_name
                self.output_dir = output_dir
                self.max_length = max_length

                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model_config = AutoConfig.from_pretrained(self.model_name)

                self.model_config.update({
                    "hidden_dropout_prob":hidden_dropout_prob,
                    "attention_probs_dropout_prob":attention_probs_dropout_prob,
                    "num_labels":2,
                    "problem_type":"regression",
                })

                self.data_collator = DataCollatorWithPadding(
                    tokenizer=self.tokenizer
                )

    def concatenate_with_sep_token(self, row):
        sep = " " + self.tokenizer.sep_token + " "
        return sep.join(row[self.input_text_cols])

    def tokenize_function(self, examples:pd.DataFrame):
        labels = [examples["content"], examples["wording"]]
        tokenized = self.tokenizer(examples[self.input_col],
                                    padding="longest",
                                    truncation=True,
                                    max_length=self.max_length)

        return {
            **tokenized,
            "labels": labels,
        }                            


    def tokenize_function_test(self, examples: pd.DataFrame):
        tokenized = self.tokenizer(examples[self.input_col],
                        padding="longest",
                        truncation=True,
                        max_length=self.max_length)
        return tokenized


    def train(self,
            fold:int,
            train_df:pd.DataFrame,
            valid_df:pd.DataFrame,
            batch_size:int,
            learning_rate:float,
            weight_decay: float,
            num_train_epochs: float,
            save_steps: int,
            ):
        train_df[self.input_col] = train_df.apply(self.concatenate_with_sep_token, axis=1)
        valid_df[self.input_col] = valid_df.apply(self.concatenate_with_sep_token, axis=1)

        train_df = train_df[[self.input_col] + self.target_cols]
        valid_df = valid_df[[self.input_col] + self.target_cols]

        model = AutoModelForSequenceClassification.from_pretrained(self.model_name)

        train_dataset = Dataset.from_pandas(train_df, preserve_index=False)
        val_dataset = Dataset.from_pandas(valid_df, preserve_index=False)

        train_tokenized_datasets = train_dataset.map(self.tokenize_function, batched=False)
        val_tokenized_datasets = val_dataset.map(self.tokenize_function, batched=False)

        # output_dir = os.path.join(self.model_dir, str(fold)) ##

        training_args = TrainingArguments(
            output_dir=self.output_dir,
            load_best_model_at_end=True,
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=num_train_epochs,
            weight_decay=weight_decay,
            gradient_checkpointing=True,
            report_to='none',
            greater_is_better=False,
            save_strategy="steps",
            evaluation_strategy="steps",
            eval_steps=save_steps,
            save_steps=save_steps,
            metric_for_best_model="mcrmse",
            save_total_limit=1,
            fp16=True,
            auto_find_batch_size=True,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_tokenized_datasets,
            eval_dataset=val_tokenized_datasets,
            tokenizer=self.tokenizer,
            compute_metrics=compute_mcrmse,
            data_collator=self.data_collator
        )

        trainer.train()

        model.save_pretrained(self.output_dir) ##
        self.tokenizer.save_pretrained(self.output_dir)

        model.cpu() ##
        del model
        gc.collect()
        torch.cuda.empty_cache()

    def predict(self,
                test_df:pd.DataFrame,
                batch_size:int,
                fold:int):
        test_df[self.input_col] = test_df.apply(self.concatenate_with_sep_token, axis=1)
        test_dataset = Dataset.from_pandas(test_df[[self.input_col]], preserve_index=False)
        test_tokenized_dataset = test_dataset.map(self.tokenize_function_test, batched=False)

        model = AutoModelForSequenceClassification.from_pretrained(self.output_dir)
        model.eval()

        test_args = TrainingArguments(
            output_dir=self.output_dir,##
            do_train=False,
            do_predict=True,
            per_device_eval_batch_size=batch_size
            dataloader_drop_last=False,
            fp16=True,
            auto_find_batch_size=True,
        )

        test = Trainer(
            model=model,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            args=test_args
        )

        preds = test.predict(test_tokenized_dataset)[0] ##
        pred_df = pd.DataFrame(
            preds,
            columns=["content_pred", "wording_pred"]
        )

        model.cpu()
        del model
        gc.collect()
        torch.cuda.empty_cache()

        return pred_df


# train,predict
def train_by_fold(
    train_df:pd.DataFrame,
    model_name:str,
    targets:List[str],
    inputs:List[str],
    batch_size: int,
    learning_rate: int,
    hidden_dropout_prob: float,
    attention_probs_dropout_prob: float,
    weight_decay: float,
    num_train_epochs: int,
    save_steps: int,
    max_length:int

):
#     if os.path.exists(model_name):
#         #shutil.rmtree(model_name)
#         print("please delete {model_name}")

#     os.mkdir(model_name)

    for fold in CFG.folds:
        print("fold:", fold)
        train_data = train_df[train_df["fold"] != fold]
        valid_data = train_df[train_df["fold"] == fold] 

        output_dir = f"{CFG.output_dir}/flod_{fold}"

        csr = ScoreRegressor(
            model_name=model_name,
            target_cols=targets,
            inputs=inputs,
            output_dir=output_dir,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            max_length=max_length,
        )   
        csr.train(
            fold=fold,
            train_df=train_data,
            valid_df=valid_data, 
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            num_train_epochs=num_train_epochs,
            save_steps=save_steps,
            )


def predict(
    test_df: pd.DataFrame,
    targets:List[str],
    inputs: List[str],
    batch_size: int,
    model_name: str,
    hidden_dropout_prob: float,
    attention_probs_dropout_prob: float,
    max_length : int
):
    columns = list(test_df.columns.values)

    for fold in CFG.folds:
        print("predict fold:", fold)
        output_dir = f"{CFG.output_dir}/fold_{fold}"
        csr = ScoreRegressor(
            model_name=model_name,
            target_cols=targets,
            inputs= inputs,
            output_dir = output_dir, 
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            max_length=max_length,
        )

        pred_df = csr.predict(
            test_df=test_df,
            batch_size=batch_size,
            fold=fold
        )

        test_df[f"content_pred_{fold}"] = pred_df["content_pred"].values
        test_df[f"wording_pred_{fold}"] = pred_df["wording_pred"].values

    test_df[f"content_pred"] = test_df[[f"content_pred_{fold}" for fold in range(len(CFG.folds))]].mean(axis=1)
    test_df[f"wording_pred"] = test_df[[f"wording_pred_{fold}" for fold in range(len(CFG.folds))]].mean(axis=1)
        

    return test_df[columns + [f"content_pred", f"wording_pred"]]    


# call
targets = ["wording", "content"]
input_cols = ["text", "prompt_question", "prompt_text"]
# mode = "multi"

train_by_fold(
    train_df=train,
    model_name=CFG.model_name,
    targets=targets,
    inputs= input_cols,
    batch_size=CFG.batch_size,
    learning_rate=CFG.learning_rate,
    hidden_dropout_prob=CFG.hidden_dropout_prob,
    attention_probs_dropout_prob=CFG.attention_probs_dropout_prob,
    weight_decay=CFG.weight_decay,
    num_train_epochs=CFG.num_train_epochs,
    save_steps=CFG.save_steps,
    max_length=CFG.max_length
)


pred = predict(
    test,
    targets=targets,
    inputs=input_cols,
    batch_size=CFG.batch_size,
    model_name=CFG.model_name,
    hidden_dropout_prob=CFG.hidden_dropout_prob,
    attention_probs_dropout_prob=CFG.attention_probs_dropout_prob,
    max_length=CFG.max_length
)


pred[targets] = pred[["content_pred", "wording_pred"]].values
pred[['student_id'] + targets].to_csv('submission.csv', index=False)