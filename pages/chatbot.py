import gradio as gr
from gradio_pdf import PDF
from pages.theme import CSS
from settings import CHAT_MSG_PLACEHOLDER


class App:
    def __init__(self, pipeline=None):
        self._pipeline = pipeline

    def _get_response(self, query, history, file_box):
        text = ""
        yield history + [(query, CHAT_MSG_PLACEHOLDER)]
        for response in self._pipeline.stream(query, history, file_box):
            text += response
            yield history + [(query, text or CHAT_MSG_PLACEHOLDER)]


    def build(self):
        with gr.Blocks(css=CSS) as app:
            with gr.Row():
                with gr.Column(scale=40):
                    model_name = gr.Dropdown(
                        choices=["bert-base-uncased", "gpt2"],
                        label="Model",
                    )
                    file_box = PDF(label='Document', height=700, min_width=500)
                with gr.Column(scale=60):
                    chatbot = gr.Chatbot(height=650)
                    text_box = gr.Textbox(
                        lines=2,
                        label="Chat message",
                        show_label=False,
                        container=False,
                        placeholder="Type here...",
                    )
                    with gr.Row():
                        clear_btn = gr.ClearButton(
                            [text_box, chatbot, file_box], variant="secondary", size="sm"
                        )
                        submit_btn = gr.Button("Submit", variant="primary", size="sm")
                        submit = submit_btn.click(
                            fn=self._get_response,
                            inputs=[text_box, chatbot, file_box],
                            outputs=chatbot
                        )
        return app
