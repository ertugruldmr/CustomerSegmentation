import pickle
import gradio as gr
import numpy as np
import sklearn


# File Paths
model_path = 'cluster.sav'

# Loading the files
model = pickle.load(open(model_path, 'rb'))



# Example Cases to test
examples = [
    [39, 61, 31],
    [71, 35, 43],
    [69, 91, 39],
    [15, 39, 19],
    [38, 35, 65],
    [15, 81, 21],
]

# Util Functions
def cluster(*args):

  # preparing the input
  features = np.array([*args]).reshape(1,3)

  # prediction
  pred = model.predict(features)
  return pred


# creating the components
with gr.Blocks() as demo_app:

    # input components
    annual_income  = gr.inputs.Slider(minimum=15, maximum=137, default=15, label="Annual Income (k$)")
    score = gr.inputs.Slider(minimum=1, maximum=99, default=39, label="Spending Score (1-100)")
    age = gr.inputs.Slider(minimum=18, maximum=70, default=19, label="Age")
    inputs = [annual_income, score, age]

    # other components
    cluster_btn = gr.Button("Cluster")
    output = gr.Number(label="Customer Segment No")
    
    # connecting function
    cluster_btn.click(fn=cluster, inputs=inputs, outputs=output) #examples=examples
    gr.Examples(examples, inputs)

# Launching the demo
if __name__ == "__main__":
    demo_app.launch()