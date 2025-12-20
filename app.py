import gradio as gr
from project_chimera.p5_agent.caelonyx_agent import CaelonyxAgent

# Initialize your AI model
agent = CaelonyxAgent()

# Define function for interface
def chat(prompt):
    try:
        response = agent.run(prompt)
        return response
    except Exception as e:
        return f"⚠️ Error: {str(e)}"

# Create Gradio interface
demo = gr.Interface(
    fn=chat,
    inputs=gr.Textbox(lines=3, label="Ask Caelonyx Anything"),
    outputs=gr.Textbox(label="Caelonyx Response"),
    title="🧠 Caelonyx Cognitive AI",
    description="An advanced cognitive model capable of reasoning, symbolic processing, and multimodal understanding."
)

if __name__ == "__main__":
    demo.launch()
