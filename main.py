# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity

# sentences = ["So the problem is that the model wasn't trained on french", "We can conclude that the AI was french not built for user"]

# model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
# embeddings = model.encode(sentences)

# similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

# threshold = 0.8
# if similarity > threshold:
#     print(f"The sentences are semantically similar (similarity: {similarity:.2f})")
# else:
#     print(f"The sentences are not semantically similar (similarity: {similarity:.2f})")

from transformers import CamembertModel, CamembertTokenizer

# You can replace "camembert-base" with any other model from the table, e.g. "camembert/camembert-large".
# tokenizer = CamembertTokenizer.from_pretrained("camembert/camembert-base")
# camembert = CamembertModel.from_pretrained("camembert/camembert-base")


from transformers import pipeline 

camembert_fill_mask  = pipeline("fill-mask", model="camembert/camembert-base", tokenizer="camembert/camembert-base")
results = camembert_fill_mask("Le camembert est un fromage de <mask>!")
print(results[0]['sequence'], '\n', str(results[0]['score']*100) + ' %')

import torch
# Tokenize in sub-words with SentencePiece
tokenized_sentence = tokenizer.tokenize("J'aime le camembert !")
# ['▁J', "'", 'aime', '▁le', '▁ca', 'member', 't', '▁!'] 

# 1-hot encode and add special starting and end tokens 
encoded_sentence = tokenizer.encode(tokenized_sentence)
# [5, 221, 10, 10600, 14, 8952, 10540, 75, 1114, 6]
# NB: Can be done in one step : tokenize.encode("J'aime le camembert !")

# Feed tokens to Camembert as a torch tensor (batch dim 1)
encoded_sentence = torch.tensor(encoded_sentence).unsqueeze(0)
embeddings, _ = camembert(encoded_sentence)
# embeddings.detach()
# embeddings.size torch.Size([1, 10, 768])
#tensor([[[-0.0928,  0.0506, -0.0094,  ..., -0.2388,  0.1177, -0.1302],
#         [ 0.0662,  0.1030, -0.2355,  ..., -0.4224, -0.0574, -0.2802],
#         [-0.0729,  0.0547,  0.0192,  ..., -0.1743,  0.0998, -0.2677],
#         ...,
