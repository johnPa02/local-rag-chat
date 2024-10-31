import gradio as gr
from gradio_pdf import PDF
from pages.theme import CSS
from pipeline import RAGPipeline
from settings import CHAT_MSG_PLACEHOLDER

DEFAULT_ASSISTANT_DICT = {'role': 'assistant', 'content': CHAT_MSG_PLACEHOLDER}

class App:
    def __init__(self, pipeline: RAGPipeline):
        self._pipeline = pipeline

    def _get_response(self, query: str, history: list[dict[str, str]]):
        message = {'role': 'user', 'content': query}
        text = ""
        yield history + [message, DEFAULT_ASSISTANT_DICT]
        streaming_response = self._pipeline.stream(query)
        for token in streaming_response.response_gen:
            text += token
            assistant_message = {'role': 'assistant', 'content': text}
            yield history + [message, assistant_message or DEFAULT_ASSISTANT_DICT]


    def build(self):
        with gr.Blocks(css=CSS) as app:
            with gr.Row():
                with gr.Column(scale=40):
                    model_name = gr.Dropdown(
                        choices=["llm3.2:1b", "llm2", "gpt-3.5"],
                        label="Model",
                    )
                    file_box = PDF(
                        label='Document',
                        height=700,
                        min_width=500,
                        interactive=True
                    )
                    file_box.upload(
                        fn=self._pipeline.process_documents,
                        inputs=[file_box],
                        outputs=[gr.Textbox(visible=False)],
                        show_progress='full'
                    )
                with gr.Column(scale=60):
                    chatbot = gr.Chatbot(height=650, type='messages')
                    text_box = gr.Textbox(
                        lines=2,
                        label="Chat message",
                        show_label=False,
                        container=False,
                        placeholder="Type here..."
                    )
                    with gr.Row():
                        clear_btn = gr.ClearButton(
                            [text_box, chatbot], variant="secondary", size="sm"
                        )
                        submit_btn = gr.Button("Submit", variant="primary", size="sm")
                        submit = submit_btn.click(
                            fn=self._get_response,
                            inputs=[text_box, chatbot],
                            outputs=chatbot
                        )
        return app
