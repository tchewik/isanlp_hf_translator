from processor_hf_translator import ProcessorHFTranslator
from isanlp import PipelineCommon


def create_pipeline(delay_init=False):
    return PipelineCommon([(ProcessorHFTranslator(),
                            ['text', 'tokens', 'sentences'],
                            {'text_translated': 'text_translated'})],
                          name='default')
