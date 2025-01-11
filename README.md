INSTALLATION GUIDE

git clone https://github.com/Somrajkarki/Pdf-summarizer.git

Install ollama from the web 
https://ollama.com/download

In terminal enter the following commands

.\myenv\Scripts\activate 
pip install -r requirements.txt

pip install streamlit

ollama run llama2:7b

//dont forget to change model name to the one you selected above in app.py. (2-3 times)

streamlit run app.py

//You are good to go.
