import os

from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    Trainer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from src.common.config import ModelArguments, DataArguments, TrainingArguments


def predict(model, tokenizer, model_inputs):
    """
    Predicts the output for the given input.
    :param model_inputs: input to the model
    :return: output of the model
    """
    print("Predicting...")
    # print("Model inputs: ", model_inputs)
    model.eval()
    # Generate output
    generated_ids = model.generate(
        input_ids=model_inputs["input_ids"],
        attention_mask=model_inputs["attention_mask"],
        max_length=512,
        num_beams=4,
        early_stopping=True,
    )
    preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False)
             for g in generated_ids]
    print("Predictions: ", preds)
    return preds


if __name__ == '__main__':
    model_name = "t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(os.path.join(TrainingArguments.output_dir,"checkpoint-2626"), config=config)
    # prefix = "Select facts:"
    # question = "Melinda learned that days in some seasons have more daylight hours than in other seasons. Which season receives the most hours of sunlight in the Northern Hemisphere?"
    # answer = "summer"
    # hypothesis = "northern hemisphere will have the most sunlight in summer"
    # facts = "if a place is in summer, then it will have the most sunlight; the northern hemisphere is a kind of hemisphere of earth; a hemisphere of earth is a kind of place"
    # input = f"{prefix} question: {question.lstrip()} answer: {answer.lstrip()} hypothesis: {hypothesis.lstrip()} facts: {facts.lstrip()}"
    # model_inputs = tokenizer(
    #     input,
    #     max_length=ModelArguments.max_src_length,
    #     padding=ModelArguments.padding,
    #     truncation=True,
    #     return_tensors="pt",
    # )
    # predict(model, tokenizer, model_inputs)

    prefix = "Predict intermediate conclusion:"
    question = "Melinda learned that days in some seasons have more daylight hours than in other seasons. Which season receives the most hours of sunlight in the Northern Hemisphere?"
    answer = "summer"
    hypothesis = "northern hemisphere will have the most sunlight in summer"
    facts = "the northern hemisphere is a kind of hemisphere of earth; a hemisphere of earth is a kind of place"
    input = f"{prefix} question: {question.lstrip()} answer: {answer.lstrip()} hypothesis: {hypothesis.lstrip()} facts: {facts.lstrip()}"
    model_inputs = tokenizer(
        input,
        max_length=ModelArguments.max_src_length,
        padding=ModelArguments.padding,
        truncation=True,
        return_tensors="pt",
    )
    predict(model, tokenizer, model_inputs)
