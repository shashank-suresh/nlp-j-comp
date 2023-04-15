
from question.pipelines import pipeline
import pandas as pd
import en_core_web_sm
from spacy.matcher import PhraseMatcher
import PyPDF2

CORPORA={"Python":"""Python is a high-level, interpreted programming language that was first released in 1991. It has a simple syntax and is easy to learn, which makes it a popular choice for beginners. Python is used for a wide range of tasks, from web development and data analysis to artificial intelligence and scientific computing. \
One of the main features of Python is its extensive standard library, which includes modules for working with regular expressions, network protocols, GUI development, and more. Python also has a large community of developers who have created many third-party libraries and tools for working with specific tasks or domains. \
Python supports multiple programming paradigms, including procedural, object-oriented, and functional programming. It also has a dynamic type system and automatic memory management, which makes it easy to write and debug code. \
Python can be used on a variety of platforms, including Windows, macOS, and Linux. There are several popular integrated development environments (IDEs) available for Python, including PyCharm, Visual Studio Code, and Spyder. \
Python is often used in scientific computing and data analysis, thanks to its support for libraries like NumPy, Pandas, and Matplotlib. It's also commonly used in web development, with popular frameworks like Django and Flask. \
Overall, Python is a versatile and powerful programming language that is widely used in industry, academia, and research.""",

"Stats":"""Statistics is the practice of collecting, analyzing, and interpreting data. It is used in a wide range of fields, including business, science, and social sciences. There are two main branches of statistics: descriptive statistics and inferential statistics. \
Descriptive statistics involve the collection and analysis of data in order to summarize and describe its characteristics. This includes measures like mean, median, mode, and standard deviation. Descriptive statistics can be used to understand the distribution of data and to identify patterns and trends. \
Inferential statistics, on the other hand, involve making inferences or predictions about a population based on a sample of data. This includes methods like hypothesis testing and confidence intervals. Inferential statistics can be used to draw conclusions about the population based on the sample data. \
There are many statistical techniques and methods that can be used to analyze data. Some common techniques include regression analysis, time series analysis, and factor analysis. Statistical software like R and SAS can be used to perform complex statistical analyses. \
Statistics is also used in machine learning and data science, where it is used to build predictive models and make data-driven decisions. Techniques like linear regression, logistic regression, and decision trees are commonly used in machine learning. \
Overall, statistics is a powerful tool for understanding and interpreting data. It allows researchers and practitioners to draw conclusions, make predictions, and make data-driven decisions based on empirical evidence.""" ,

'NLP':"""Natural Language Processing (NLP) is a field of study that focuses on the interaction between human language and computers. NLP techniques are used to analyze, understand, and generate natural language text. The goal of NLP is to enable computers to understand human language and to communicate with humans in a natural way. \
NLP involves several tasks, including language modeling, part-of-speech tagging, named entity recognition, sentiment analysis, and machine translation. These tasks are often performed using machine learning algorithms, such as neural networks, decision trees, and support vector machines. \
One of the main challenges in NLP is the ambiguity and complexity of human language. Words can have multiple meanings, and sentences can have different interpretations depending on context. To address this challenge, NLP researchers use techniques such as semantic analysis and context modeling to better understand the meaning of language. \
NLP has many applications, including chatbots, virtual assistants, and sentiment analysis. Chatbots are computer programs that can communicate with humans through natural language text or speech. Virtual assistants, like Siri and Alexa, are examples of chatbots that can help users with a wide range of tasks. Sentiment analysis is the process of analyzing text to determine the sentiment or emotion expressed in it. This can be used for applications like social media monitoring and customer feedback analysis. \
Overall, NLP is a rapidly growing field with many exciting applications. As computers become more intelligent and more capable of understanding human language, NLP will continue to play an important role in enabling human-computer communication""",

'ML':"""Machine learning is a type of artificial intelligence that allows machines to learn from data, without being explicitly programmed. It involves the use of statistical models and algorithms to identify patterns in data and make predictions or decisions based on those patterns. The goal of machine learning is to enable machines to automatically improve their performance on a task through experience \
There are three main types of machine learning: supervised learning, unsupervised learning, and reinforcement learning. In supervised learning, the machine is trained on labeled data, meaning the input data is accompanied by the correct output. In unsupervised learning, the machine is trained on unlabeled data, meaning the input data is not accompanied by the correct output. In reinforcement learning, the machine is trained by interacting with an environment and learning from feedback in the form of rewards or punishments \
Machine learning algorithms can be used for a wide range of tasks, including image recognition, natural language processing, and anomaly detection. Some popular machine learning algorithms include decision trees, random forests, support vector machines, and neural networks \
There are many challenges to using machine learning, including overfitting, bias, and interpretability. Overfitting occurs when the machine learns to fit the training data too closely, resulting in poor performance on new data. Bias can occur when the machine learns from data that is not representative of the population it will be used on. Interpretability refers to the ability to understand how the machine arrived at its predictions or decisions \
Overall, machine learning is a powerful tool for solving complex problems and making predictions based on data. As data becomes more abundant and computational power increases, machine""",

'DL':"""Deep learning is a subfield of machine learning that involves the use of artificial neural networks to model and solve complex problems. Deep learning algorithms are capable of learning from data with little to no human intervention and can be used for tasks such as image recognition, speech recognition, natural language processing, and autonomous driving\
The main building block of a deep learning algorithm is the artificial neural network, which is inspired by the structure of the human brain. An artificial neural network consists of layers of interconnected nodes, called neurons, that process information and make predictions. The input to the network is fed through the layers of neurons, with each layer learning increasingly abstract features of the data. The output of the network is the prediction or decision made by the algorithm\
Some popular deep learning architectures include convolutional neural networks (CNNs) and recurrent neural networks (RNNs). CNNs are commonly used for image recognition tasks, while RNNs are used for sequential data, such as natural language processing\
There are many challenges to using deep learning, including overfitting, vanishing gradients, and explainability. Overfitting occurs when the algorithm learns to fit the training data too closely, resulting in poor performance on new data. Vanishing gradients occur when the gradients used to update the neural network weights become very small, making it difficult for the network to learn. Explainability refers to the ability to understand how the deep learning algorithm arrived at its predictions or decisions\
Overall, deep learning is a powerful tool for solving complex problems and making predictions based on data. As data becomes more abundant and computational power increases, deep learning will continue to play an important role in industry, research, and everyday life.""",

'R':"""R is a programming language and software environment for statistical computing and graphics. It is widely used in academia, industry, and research for data analysis, statistical modeling, and visualization \
One of the main strengths of R is its large and active user community, which has developed thousands of packages for a wide range of tasks. Some popular packages include ggplot2 for data visualization, dplyr for data manipulation, and caret for machine learning \
R is also known for its powerful graphics capabilities, which allow users to create high-quality plots and visualizations with ease. The graphics system in R is highly customizable and can be used to create a wide range of plots, from basic line charts to complex heatmaps and 3D visualizations \
R is a versatile language that can be used for many different types of data analysis, including statistical analysis, machine learning, and data visualization. It also has strong support for data import and export, with built-in functions for reading and writing data in a variety of formats, including CSV, Excel, and SQL \
One potential challenge with R is its learning curve, particularly for users who are new to programming or statistical analysis. However, there are many resources available to help users get started with R, including online tutorials, documentation, and user forums \
Overall, R is a powerful and flexible programming language that is well-suited for data analysis and visualization. Its large and active user community, powerful graphics capabilities, and extensive library of packages make it a popular choice for researchers, analysts, and data scientists.""",

'Data Engineering':"""Data engineering is the process of designing, building, and maintaining the infrastructure and systems that are used to collect, store, process, and analyze large volumes of data. It involves a variety of technologies and tools, including databases, data warehouses, data lakes, ETL (extract, transform, load) tools, and big data frameworks \
One of the key challenges in data engineering is managing and processing large volumes of data. This requires careful consideration of factors such as data storage, processing speed, and scalability. Data engineers must also ensure that the data they are working with is accurate, consistent, and secure \
Another important aspect of data engineering is data integration. This involves bringing together data from a variety of sources, including databases, APIs, and third-party services, and transforming it into a format that can be used for analysis. This process can be complex, requiring data engineers to have a deep understanding of data structures, algorithms, and programming languages \
Data engineers must also be familiar with data governance and compliance regulations, particularly if they are working with sensitive or regulated data. They must ensure that data is collected, stored, and processed in compliance with legal and ethical guidelines, and that it is accessible only to authorized users \
Overall, data engineering is a critical component of the data analysis pipeline, enabling organizations to collect, store, and process large volumes of data for analysis and decision-making. Data engineers play a crucial role in designing and maintaining the systems that enable this process, and must be skilled in a wide range of technologies and tools.""",
}

def generate(location):
    nlp = en_core_web_sm.load()
    pdfFileObj = open(location, 'rb')
    pdfReader = PyPDF2.PdfReader(pdfFileObj)
    # print(len(pdfReader.pages))
    pageObj = pdfReader.pages[0]
    # print(pageObj.extract_text ())
    text=pageObj.extract_text ()
    pdfFileObj.close()
    keyword_dict = pd.read_csv('Skills_Keywords.csv')   
    keyword_dict.head()
    stats_words = [nlp(text) for text in keyword_dict['Statistics'].dropna(axis = 0)]
    nlp_words = [nlp(text) for text in keyword_dict['NLP'].dropna(axis = 0)]
    ml_words = [nlp(text) for text in keyword_dict['Machine Learning'].dropna(axis = 0)]
    dl_words = [nlp(text) for text in keyword_dict['Deep Learning'].dropna(axis = 0)]
    r_words = [nlp(text) for text in keyword_dict['R Language'].dropna(axis = 0)]
    python_words = [nlp(text) for text in keyword_dict['Python Language'].dropna(axis = 0)]
    data_eng_words = [nlp(text) for text in keyword_dict['Data Engineering'].dropna(axis = 0)]
    matcher = PhraseMatcher(nlp.vocab)
    matcher.add('Stats', None, *stats_words)
    matcher.add('NLP', None, *nlp_words)
    matcher.add('ML', None, *ml_words)
    matcher.add('DL', None, *dl_words)
    matcher.add('R', None, *r_words)
    matcher.add('Python', None, *python_words)
    matcher.add('Data Engineering', None, *data_eng_words)
    d = {}
    doc = nlp(text)
    print("____________________THE DOCUMENT________________")
    print(doc)
    # Find matches in the doc
    matches = matcher(doc)
    # print(matches)
    # For each of the matches
    idx=0
    for match_id, start, end in matches:
        # Get the general word and the matched phrase
        gen_word = nlp.vocab.strings[match_id]
        match = doc[start:end]

        # Append all the keywords specific to a resume ID
        d.setdefault(gen_word, []).append(match.text)
        idx+=1
    print("_____MATCHES______")
    print(d)

    nlp = pipeline("multitask-qa-qg", model="valhalla/t5-base-qa-qg-hl")
    print("Corpus used and questions generated")
    for i in d.keys():
        print(nlp(CORPORA[i]))