6/05/2025

*For 6eme homme* - thank you Hamzah for already doing something like this in the past and not gatekeeping. ❤️

## Task
- 6eme is looking into implementing AI to match resumes to gigs
- either they outsource it to some already existing product
- or they do it in house (might not be worth)
- seems like an easy (relatively) task so I though I would try

found [Resume Matcher](https://github.com/srbhr/Resume-Matcher) which seems like a good tool, I need to understand how to use it though 
- plan is to send this to Reda

there is [this tool](https://github.com/kirudang/CV-Job-matching/tree/main) as well even though it has a lot less stars and is less polished
- looks simples
[this article](https://kartikmadan11.medium.com/building-a-job-description-to-resume-matcher-using-natural-language-processing-5a4f5181cfe4) seems very helpful as well - about methods of varying difficulty to accomplish this task + implementations - talks about the BERT thing that hamza uses

### What is BERT?
It is an open-source NLP (natural language processing) machine learning framework (made by GOOGLE?????). It is made? for understanding the context of text by considering all words in a sentence - and it is **great for fine-tuning** because it was trained on 2 tasks:
1. Masked LM(Language Modelling)![[Screenshot 2025-05-07 at 12.06.13.png]]
2. Next sentence prediciton
![[Screenshot 2025-05-07 at 12.07.26.png]]
Cool stuff:![[Screenshot 2025-05-07 at 12.12.00.png]]
**Pre-training:**
	BERT has been pre-trained on loads of data using unsupervised learning techniques.
**Fine-tuning:** [this video details some of that](https://www.youtube.com/watch?v=4QHg8Ix8WWQ&t=309s)
	it can be fine-tuned for specific tasks by adding a task-specific output layer on top of the pre-trained model
**Transformers**:
encoder-decoder model that can take advantage of parrelisation and process a lot of data at the same time
## Bert implementation 
1. text preprocessing:
	give BERT raw text (not sure if all lowercase)
2. use the **transformers** library by Hugging face to load a pre-trained BERT model
	we will convert descriptions resumes into embeddings using the model
3. Calculation/Scoring
	we will use cosine similarity to find matches between the job descriptions and CVs
from the [medium article](https://kartikmadan11.medium.com/building-a-job-description-to-resume-matcher-using-natural-language-processing-5a4f5181cfe4)
	"*To evaluate our matcher, we will use labeled test data consisting of resumes and corresponding job descriptions. We will measure precision, recall, and F1 score for each approach to determine their effectiveness. The **feedback loop** will involve continuously refining the models based on real-world performance and user feedback, allowing for iterative improvements.*"
	
	but how though?

I think that i'm going to use [bert-base-multilingual-cased](https://huggingface.co/google-bert/bert-base-multilingual-cased) because it is small and supports french allegedly - as french is one of the more common languages on wikipidea (large part of BART's training data)

Going to try to use this sentence transformer model: [paraphrase-multilingual-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)
- i know have working code that takes 2 sentences and compares them but the score I get is just bullshit.
Apparently there are some BERT models that are better suited for french: notament [camemBERT](https://camembert-model.fr) - i find this way funnier then i should - there is also an older one called flauBERT
