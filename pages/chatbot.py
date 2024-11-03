import gradio as gr
from gradio_pdf import PDF
from pages.theme import CSS
from pipeline import RAGPipeline
from settings import CHAT_MSG_PLACEHOLDER, DEFAULT_MODEL_LIST

DEFAULT_ASSISTANT_DICT = {'role': 'assistant', 'content': CHAT_MSG_PLACEHOLDER}

class App:
    def __init__(self, pipeline: RAGPipeline):
        self._pipeline = pipeline
        self.document_processed = False

    def _get_response(self, query: str, history: list[dict[str, str]]):
        if not self.document_processed:
            yield ["## Please upload a document first!", history, query]
            return
        if not query:
            yield ["", history, ""]
            return

        query = query.strip()
        message = {'role': 'user', 'content': query}
        text = ""
        yield ["", history + [message, DEFAULT_ASSISTANT_DICT], ""]
        streaming_response = self._pipeline.stream(query)
        for token in streaming_response.response_gen:
            text += token
            assistant_message = {'role': 'assistant', 'content': text}
            yield ["", history + [message, assistant_message or DEFAULT_ASSISTANT_DICT], ""]

    def upload_file(self, files):
        yield "## Processing documents, please wait..."
        self._pipeline.process_documents(files)
        self.document_processed = True
        yield "## Documents processed successfully!"

    def change_llm(self, model: str):
        yield "## Changing model, please wait..."
        self._pipeline.change_llm(model)
        yield "## Model changed successfully!"

    def build(self):
        with gr.Blocks(css=CSS) as app:
            status = gr.Markdown("")
            with gr.Row():
                with gr.Column(scale=40):
                    model_name = gr.Dropdown(
                        choices=DEFAULT_MODEL_LIST,
                        label="Model",
                    )
                    file_box = PDF(
                        label='Document',
                        height=700,
                        min_width=500,
                        interactive=True
                    )
                    file_box.upload(
                        fn=self.upload_file,
                        inputs=[file_box],
                        outputs=[status],
                    )
                    model_name.change(
                        fn=self.change_llm,
                        inputs=[model_name],
                        outputs=[status]
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
                            [chatbot], variant="secondary", size="sm"
                        )
                        submit_btn = gr.Button("Submit", variant="primary", size="sm")
                        submit_btn.click(
                            fn=self._get_response,
                            inputs=[text_box, chatbot],
                            outputs=[status, chatbot, text_box]
                        )
        return app
