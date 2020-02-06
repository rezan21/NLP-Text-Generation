import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import streamlit as st
import altair as alt

from PIL import Image
import tensorflow as tf
print('--- done --- ')
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense,Embedding,Dropout,GRU
from tensorflow.keras.losses import sparse_categorical_crossentropy
import pickle
with open('ind_char.pickle', 'rb') as handle:
    ind_to_char = pickle.load(handle)
with open('char_ind.pickle', 'rb') as handle:
    char_to_ind = pickle.load(handle) 
vocab_size = 84
embed_dim = 64
rnn_neurons = 1026
def main():
        
    # style css
    st.markdown('''
        <style>
            @import url('https://fonts.googleapis.com/css?family=Open+Sans+Condensed:300&display=swap');

            * {
                font-size: 1.2rem;
                margin: 0;
                padding: 0;
                -webkit-box-sizing: border-box;
                box-sizing: border-box;
                font-family: 'Open Sans Condensed', sans-serif;
                text-align:center;
            }

            .my-top-text{
                line-height:90%;
            }

            #general{
                font-size: 19.2px;
                font-family: 'Open Sans Condensed', sans-serif;
                
            }

            .reportview-container .main .block-container {
                padding-top:40px;
            }

            .reportview-container h3{
                text-align: center;
                margin:0px 0px 10px 0px;
                padding: 0px 0px 30px 0px; 
                border-bottom: 1px solid #909090;
            }
            .reportview-container h1{
                padding:0;
                margin:0;
            }

            .reportview-container .main footer{
                display: none !important;
            }






            
        </style>
        ''',unsafe_allow_html=True)

    st.title("AI-Generated Shakespeare")
    st.subheader("Generate Shakespeare-alike poems using the power of AI")

    img = Image.open('undraw.png')
    st.image(img, width=330)

    st.markdown('''
        <div class="my-top-text"> 
            <p><b id="general">Algorithm:</b> The app uses Natural Language Processing (NLP) along with Recurrent Neural Networks (RNN) to generate text based on Shakespeare's work. For technical details, please <a id="general" href="github.com">visit github repository</a>.</p>
            <p> <b id="general">How does it work?</b> Enter a word in the input field below. The generated text will depend on this, as the algorithm uses it as a starting seed. You can also specify a temperature. It's used to aeffect the probability of next characters.</p>
        </div>
    ''', unsafe_allow_html=True)

    # functions
    def sparse_cat_loss(y_true,y_pred):
        return sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)

    def create_model(vocab_size, embed_dim, rnn_neurons, batch_size):
        model = Sequential()
        model.add(Embedding(vocab_size, embed_dim,batch_input_shape=[batch_size, None]))
        model.add(GRU(rnn_neurons,return_sequences=True,stateful=True,recurrent_initializer='glorot_uniform'))
        # Final Dense Layer to Predict
        model.add(Dense(vocab_size))
        model.compile(optimizer='adam', loss=sparse_cat_loss) 
        return model

    def generate_text(model, start_seed,gen_size=800,temp=1.0):
        num_generate = gen_size
        input_eval = [char_to_ind[s] for s in start_seed]
        input_eval = tf.expand_dims(input_eval, 0)
        text_generated = []
        temperature = temp
        model.reset_states()

        for i in range(num_generate):
            predictions = model(input_eval)
            predictions = tf.squeeze(predictions, 0)
            predictions = predictions / temperature
            predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()
            input_eval = tf.expand_dims([predicted_id], 0)
            text_generated.append(ind_to_char[predicted_id])
        return (start_seed + ''.join(text_generated))

    model = create_model(vocab_size, embed_dim, rnn_neurons, batch_size=1)
    model.load_weights('shakespeare_gen.h5')
    model.build(tf.TensorShape([1, None]))

    def isValid(x):
        checked = True
        for c in x:
            if c not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz '":
                checked = False
                break
            else:
                checked = True
        return checked

    # User inputs
    start_seed = st.text_input("Enter Words: (Min 3, Max 20 characters)")
    temp = st.slider("Temperature", 0.1,2.0,1.0) # min, default, max values
    submit = st.button("Generate!")

    error = False 
    loading = False

    # Text input validation
    start_seed = start_seed.strip()
    if len(start_seed) == 0:
        error = True
    elif len(start_seed)>20 or len(start_seed)<3:
        error = True
        st.error("Text must be between 3 to 20 characters.")
    if not isValid(start_seed):
        error = True
        st.error("Text must be only English characters.")

    # On submit
    if submit: 
        try:
            if not error:
                loading = True
                while loading:
                    with st.spinner("Generating Text. This might take up to 1 minute."):
                        gen_text = generate_text(model,start_seed,gen_size=500,temp=temp)
                        if gen_text:
                            loading = False
                st.text(gen_text)

            else: # if there is some error
                if len(start_seed) == 0:
                    st.warning("You Must Fill the Text Field!")
                else:
                    st.warning("Failed! Fix the Errors")
        except:
            st.error('Something went wrong!')

# to keep the space for a later component
# placeHolder = st.empty() 
if __name__ == "__main__":
    main()