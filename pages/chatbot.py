import gradio as gr
from gradio_pdf import PDF
from pages.theme import CSS
from pipeline import RAGPipeline
from settings import CHAT_MSG_PLACEHOLDER


class App:
    def __init__(self, pipeline: RAGPipeline):
        self._pipeline = pipeline

    def _get_response(self, query, history):
        text = ""
        yield history + [(query, CHAT_MSG_PLACEHOLDER)]
        for response in self._pipeline.stream(query):
            text += response
            yield history + [(query, text or CHAT_MSG_PLACEHOLDER)]


    def build(self):
        with gr.Blocks(css=CSS) as app:
            with gr.Row():
                with gr.Column(scale=40):
                    model_name = gr.Dropdown(
                        choices=["llm3", "llm2", "gpt-3.5"],
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
                        outputs=[gr.Textbox(visible=False)]
                    )
                with gr.Column(scale=60):
                    chatbot = gr.Chatbot(height=650)
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
