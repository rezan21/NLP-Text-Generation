# NLP-Text-Generation
AI Generates Shakespeare-alike poem. This is possible thorugh training the Recurrent Neural Network (RNN) on 5,000,000 text characters of Shakespeare's writings. Details of the technical implementation and model creation can be found in the `nlp-text-generation.ipynb` jupyter notebook.

### Demo:
The trained model is deployed online, through Heroku, available at: [ai-generated-shakespeare.herokuapp.com](https://ai-generated-shakespeare.herokuapp.com/)

### Dataset
5,000,000 characters of Shakespeare's work. Dataset available at:  [MIT Website](https://ocw.mit.edu/ans7870/6/6.006/s08/lecturenotes/files/t8.shakespeare.txt "MIT Dataset").

### Used Libraries:
- Tensorflow-gpu
- Scikit-Learn
- Pandas
- NumPy

### ML Model: Recurrent Neural Network (RNN)
> A recurrent neural network (RNN) is a class of artificial neural networks where connections between nodes form a directed graph along a temporal sequence. This allows it to exhibit temporal dynamic behavior. Unlike feedforward neural networks, RNNs can use their internal state (memory) to process variable length sequences of inputs. This makes them applicable to tasks such as unsegmented, connected handwriting recognition or speech recognition. [-Wikipedia](https://en.wikipedia.org/wiki/Recurrent_neural_network "reference source")

### To run locally:
1. __`git clone https://github.com/rezan21/NLP-Text-Generation.git` and `cd NLP-Text-Generation`__

2. __Install required libraries__
    - `pip install -r requirements.txt`
    
3. __Run on localhost `streamlit run app.py`__

<br />
<br />
<br />

Partial Credit to [Jose Portilla](https://www.linkedin.com/in/jmportilla/).

