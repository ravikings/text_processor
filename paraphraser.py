from transformers import T5Tokenizer, T5Config, T5ForConditionalGeneration

def paraphrasingT5(input_text):
    # Instantiating the model and tokenizer 
    my_model = T5ForConditionalGeneration.from_pretrained('t5-small')
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    input_ids=tokenizer.encode(input_text, return_tensors='pt', max_length=512)
    summary_ids = my_model.generate(input_ids)
    t5_summary = tokenizer.decode(summary_ids[0])

    return t5_summary




