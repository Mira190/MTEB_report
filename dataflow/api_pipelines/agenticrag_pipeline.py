from dataflow.operators.generate.AgenticRAG import (
    AutoPromptGenerator,
    QAGenerator,
    QAScorer
)

from dataflow.operators.process.AgenticRAG import (
    ContentChooser
)

from dataflow.utils.storage import FileStorage
from dataflow.llmserving import APILLMServing_request, LocalModelLLMServing

import os

class AgenticRAGPipeline():
    def __init__(self):
        self.storage = FileStorage(
            first_entry_file_name="1.json",
            cache_path="./cache_local",
            file_name_prefix="dataflow_cache_step",
            cache_type="json",
        )

        # use DashScope API as LLM serving (no extra_generation_parameters here)
        llm_serving = APILLMServing_request(
            api_url="https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation",
            model_name="qwen-max",
            max_workers=1
        )

        self.content_chooser_step1 = ContentChooser(embedding_model_path="your embedding model path")
        self.prompt_generator_step2 = AutoPromptGenerator(llm_serving)
        self.qa_generator_step3    = QAGenerator(llm_serving)
        self.qa_scorer_step4       = QAScorer(llm_serving)
        
    def forward(self):
        self.content_chooser_step1.run(
            storage=self.storage.step(),
            input_key="content",
            num_samples=5,
            method="random"
        )

        self.prompt_generator_step2.run(
            storage=self.storage.step(),
            input_key="content"
        )

        self.qa_generator_step3.run(
            storage=self.storage.step(),
            input_key="content",
            prompt_key="generated_prompt",
            output_quesion_key="generated_question",
            output_answer_key="generated_answer"
        )

        self.qa_scorer_step4.run(
            storage=self.storage.step(),
            input_question_key="generated_question",
            input_answer_key="generated_answer",
            output_question_quality_key="question_quality_grades",
            output_question_quality_feedback_key="question_quality_feedbacks",
            output_answer_alignment_key="answer_alignment_grades",
            output_answer_alignment_feedback_key="answer_alignment_feedbacks",
            output_answer_verifiability_key="answer_verifiability_grades",
        )
        
if __name__ == "__main__":
    model = AgenticRAGPipeline()
    model.forward()
