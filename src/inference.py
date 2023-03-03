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


def mtl_inference(model, tokenizer):
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
    facts = 'if a place is in summer, then it will have the most sunlight; a hemisphere of earth is a kind of place'
    input = f"{prefix} question: {question.lstrip()} answer: {answer.lstrip()} hypothesis: {hypothesis.lstrip()} facts: {facts.lstrip()}"
    model_inputs = tokenizer(
        input,
        max_length=ModelArguments.max_src_length,
        padding=ModelArguments.padding,
        truncation=True,
        return_tensors="pt",
    )
    predict(model, tokenizer, model_inputs)


def selection_inference(model, tokenizer):
    prefix = "Select facts:"
    question = "Melinda learned that days in some seasons have more daylight hours than in other seasons. " \
               "Which season receives the most hours of sunlight in the Northern Hemisphere?"
    facts = ["if a place is in summer, then it will have the most sunlight",
             "the northern hemisphere is a kind of hemisphere of earth", "a hemisphere of earth is a kind of place"]
    factoids = [
        f"sent {i}: {factoid}" for i, factoid in enumerate(facts)
    ]
    # append the prefix at start and question at the end
    prompt = [prefix] + factoids + [f"question: {question}"]
    # join the factoids with new line
    prompt = "\n".join(prompt)
    print(prompt)
    model_inputs = tokenizer(
        prompt,
        max_length=ModelArguments.max_src_length,
        padding=ModelArguments.padding,
        truncation=True,
        return_tensors="pt",
    )
    prediction = predict(model, tokenizer, model_inputs)[0]

    context_dict = {}
    for i, sent in enumerate(facts):
        context_dict[f"sent {i}"] = sent

    # replace key with its value in prediction
    for k,v in context_dict.items():
        if k in prediction:
            prediction = prediction.replace(k, v)

    print(prediction)


def inf_module(model, tokenizer):
    prefix = "Predict conclusion:"
    selected_facts = "If a place is in summer, then it will have the most sunlight. " \
            "We know that the northern hemisphere is a kind of hemisphere of earth."
    prompt = f"{prefix} {selected_facts}"
    model_inputs = tokenizer(
        prompt,
        max_length=ModelArguments.max_src_length,
        padding=ModelArguments.padding,
        truncation=True,
        return_tensors="pt",
    )
    prediction = predict(model, tokenizer, model_inputs)[0]
    print(prediction)


if __name__ == '__main__':
    model_name = "t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name)
    output_dir = "/storage/ukp/work/sachdeva/research_projects/chain-of-thought/t5-base-inference-generation"
    model = AutoModelForSeq2SeqLM.from_pretrained(os.path.join(output_dir), config=config)
    inf_module(model, tokenizer)
