import gradio as gr
from task1.pipeline import process_ticket

def predict(ticket_text):
    issue_type, urgency_level, entities = process_ticket(ticket_text)
    return issue_type, urgency_level, entities

demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(lines=5, label="Ticket Text"),
    outputs=[
        gr.Text(label="Predicted Issue Type"),
        gr.Text(label="Predicted Urgency Level"),
        gr.JSON(label="Extracted Entities")
    ],
    title="Ticket Classification & Entity Extraction",
    description="Enter a customer support ticket to predict issue type, urgency, and extract useful entities."
)

if __name__ == "__main__":
    demo.launch()
