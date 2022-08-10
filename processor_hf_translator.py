from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from langdetect import detect


class ProcessorHFTranslator:
    """ Uses pretrained transformers to translate the text sentence by sentence
        En->Ru or Ru->En by default
    """

    def __init__(self, model_name='facebook/nllb-200-distilled-1.3B', lang1='ru', lang2='en'):
        """
        Args:
            model_name (str): model name in Hugging Face Hub
        """
        self.model_name = model_name
        self._lang_codes = {
            'ru': 'rus_Cyrl',
            'en': 'eng_Latn',
        }
        self._language_options = {
            lang1: self._lang_codes[lang2],
            lang2: self._lang_codes[lang1],
        }

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def __call__(self, text, tokens, sentences):
        sentences_text = [text[tokens[sent.begin].begin:tokens[sent.end - 1].end] for sent in sentences]
        language = detect(text)
        lang_params = self._language_options.get(language)
        if not lang_params:
            # Unknown source language
            return {'text_translated': []}

        inputs = self.tokenizer.batch_encode_plus(sentences_text, padding=True, return_tensors="pt")
        result = self.model.generate(**inputs,
                                     forced_bos_token_id=self.tokenizer.lang_code_to_id[lang_params])

        return {'text_translated': self.tokenizer.batch_decode(result, skip_special_tokens=True)}
