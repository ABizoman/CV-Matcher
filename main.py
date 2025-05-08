from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

DescriptionFilePaths = ["Descriptions/ExperiencedSWE.txt",
                       "Descriptions/ViticultureConsultant.txt", 
                       "Descriptions/SWESystems&performance.txt",
                       "Descriptions/ITAnalyst.txt",
                       "Descriptions/GraduateSWE.txt",
                       "Descriptions/AudioSWE.txt",
                       "Descriptions/AmazonSDEIntern.txt"]

CVFilePath = "CVs/OthmanCV.txt"

def read_text_file(file_path):
    """Read text content from a plain text file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except UnicodeDecodeError:
        # Try with another encoding if UTF-8 fails
        try:
            with open(file_path, 'r', encoding='latin-1') as file:
                return file.read()
        except Exception as e:
            print(f"Error reading text file {file_path}: {e}")
            return ""
    except Exception as e:
        print(f"Error reading text file {file_path}: {e}")
        return ""

CVStr = read_text_file(CVFilePath)
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

for description_path in DescriptionFilePaths:
    DescriptionStr = read_text_file(description_path)
    sentences = [DescriptionStr, CVStr]
    embeddings = model.encode(sentences)
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

    threshold = 0.6
    adjusted_similarity = (similarity / 0.6) * 100
    if adjusted_similarity > threshold * 100:
        print(f"{description_path}: The description is a match (adjusted similarity: {adjusted_similarity:.0f}%)")
    else:
        print(f"{description_path}: The description is not a match (adjusted similarity: {adjusted_similarity:.0f}%)")
