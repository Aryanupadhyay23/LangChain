from langchain_classic.text_splitter import RecursiveCharacterTextSplitter

text = """
Artificial Intelligence (AI) is transforming the way humans interact with technology. From recommendation systems on streaming platforms to self-driving cars and virtual assistants, AI is becoming an essential part of daily life. Machine learning, a subset of AI, enables systems to learn patterns from data and make predictions without being explicitly programmed for every task.

Natural Language Processing (NLP) is one of the most popular applications of AI. NLP allows machines to understand, process, and generate human language. Chatbots, translation systems, text summarizers, and sentiment analysis tools are all powered by NLP techniques. Large Language Models (LLMs) such as GPT-based systems are trained on massive datasets to generate human-like responses and assist users in various tasks.

Deep learning is another important branch of AI that uses neural networks with multiple layers to process complex data. Convolutional Neural Networks (CNNs) are widely used in image classification and object detection tasks, while Recurrent Neural Networks (RNNs), LSTMs, and Transformers are commonly used for sequence and language-related tasks. Transformers have revolutionized AI by enabling models to process large amounts of information efficiently.

MLOps is the practice of deploying, monitoring, and maintaining machine learning models in production environments. It combines machine learning, DevOps, and data engineering principles to automate workflows and improve scalability. Tools such as Docker, Kubernetes, MLflow, and DVC are commonly used in MLOps pipelines to manage experiments, version data, and deploy models consistently.

Data preprocessing is a critical step in any machine learning workflow. Raw data often contains missing values, duplicate entries, inconsistent formats, and outliers. Data scientists clean and transform the data before feeding it into machine learning algorithms. Feature engineering, normalization, encoding categorical variables, and handling imbalanced datasets are some common preprocessing techniques.

Generative AI focuses on creating new content such as text, images, audio, and videos. Models like GPT, diffusion models, and GANs can generate realistic outputs based on training data. Generative AI is being used in education, healthcare, entertainment, marketing, and software development. However, ethical concerns such as misinformation, bias, and copyright issues are also important topics in this field.

Cloud computing platforms like AWS, Google Cloud, and Microsoft Azure provide scalable infrastructure for AI applications. These platforms offer GPU resources, storage systems, APIs, and managed services for training and deploying machine learning models. Cloud-native AI solutions help organizations build intelligent systems without investing heavily in physical hardware.

The future of AI is expected to include advancements in autonomous systems, robotics, healthcare diagnostics, and personalized education. Researchers are also exploring Artificial General Intelligence (AGI), where machines could perform tasks with human-like reasoning abilities. Despite rapid progress, challenges such as data privacy, security, explainability, and responsible AI development remain significant areas of focus.

"""

splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=0)
chunks = splitter.split_text(text)
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}:\n{chunk}\n")