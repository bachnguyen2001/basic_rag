import logging
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import faiss
import google.generativeai as genai
import numpy as np
from collections import defaultdict
from functools import lru_cache
from transformers import AutoTokenizer, AutoModel
import torch
import matplotlib.pyplot as plt
import io
import base64
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from ..core.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Download NLTK data
nltk.download('punkt')

# Configure Gemini
try:
    genai.configure(api_key=settings.GEMINI_API_KEY)
    llm = genai.GenerativeModel(settings.GENERATION_MODEL)
    logger.info("Initialized Gemini API")
except Exception as e:
    logger.error(f"Failed to initialize Gemini API: {e}")
    raise

# Load dataset
try:
    logger.info("Loading dataset...")
    ds = load_dataset("Amod/mental_health_counseling_conversations")
    train = ds["train"]
    contexts = train["Context"]
    responses = train["Response"]
    logger.info(f"Loaded dataset with {len(contexts)} contexts")
except Exception as e:
    logger.error(f"Failed to load dataset: {e}")
    raise

# Map context to responses
context_to_responses = defaultdict(list)
for c, r in zip(contexts, responses):
    context_to_responses[c].append(r)

# Initialize embedding model and FAISS index
try:
    logger.info("Initializing embedding model and FAISS index...")
    embed_model = SentenceTransformer(settings.EMBEDDING_MODEL)
    context_embeddings = embed_model.encode(contexts, convert_to_numpy=True, show_progress_bar=True)
    faiss.normalize_L2(context_embeddings)
    d = context_embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(context_embeddings)
    logger.info(f"Created FAISS index with {index.ntotal} vectors")
except Exception as e:
    logger.error(f"Failed to initialize embedding model or FAISS: {e}")
    raise

# Initialize tokenizer and model for attention analysis
try:
    tokenizer = AutoTokenizer.from_pretrained(settings.EMBEDDING_MODEL)
    transformer_model = AutoModel.from_pretrained(settings.EMBEDDING_MODEL, output_attentions=True)
    logger.info("Initialized transformer model for attention analysis")
except Exception as e:
    logger.error(f"Failed to initialize transformer model: {e}")
    raise

@lru_cache(maxsize=100)
def encode_query(query: str) -> np.ndarray:
    """Cache query embeddings to avoid re-encoding."""
    logger.info(f"Encoding query: {query}")
    q_emb = embed_model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    return q_emb

def pure_gemini(query: str) -> str:
    try:
        prompt = f"Answer empathetically as a counselor: {query}\nResponse:"
        response = llm.generate_content(prompt)
        return response.text
    except Exception as e:
        logger.error(f"Pure Gemini failed: {e}")
        return f"Error: {str(e)}"

def calculate_metrics(reference: str, candidate: str) -> dict:
    try:
        # BLEU with smoothing
        ref_tokens = [nltk.word_tokenize(reference)]
        cand_tokens = nltk.word_tokenize(candidate)
        bleu = sentence_bleu(ref_tokens, cand_tokens, smoothing_function=SmoothingFunction().method1)

        # ROUGE
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        rouge = scorer.score(reference, candidate)

        return {
            "bleu": bleu,
            "rouge": {
                "rouge1": rouge['rouge1'].fmeasure,
                "rouge2": rouge['rouge2'].fmeasure,
                "rougeL": rouge['rougeL'].fmeasure
            }
        }
    except Exception as e:
        logger.error(f"Error calculating metrics: {e}")
        return {"bleu": 0.0, "rouge": {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}}

def analyze_attention(query: str) -> dict:
    try:
        # Tokenize and run model
        inputs = tokenizer(query, return_tensors="pt")
        outputs = transformer_model(**inputs)

        # Extract attention weights (average across layers and heads)
        attentions = outputs.attentions
        avg_attention = torch.mean(torch.stack(attentions), dim=0).mean(dim=1).squeeze(0).detach().numpy()
        tokens = tokenizer.tokenize(query)

        # Generate attention matrix plot
        plt.figure(figsize=(10, 8))
        plt.imshow(avg_attention, cmap='hot', interpolation='nearest')
        plt.xticks(np.arange(len(tokens)), tokens, rotation=90)
        plt.yticks(np.arange(len(tokens)), tokens)
        plt.colorbar()
        plt.title("Average Attention Matrix for Query")
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()

        # Analyze attention for mental health keywords
        keywords = ["depressed", "anxious", "worthless", "stress", "sad"]
        keyword_attention = {}
        for keyword in keywords:
            if keyword in tokens:
                idx = tokens.index(keyword)
                keyword_attention[keyword] = {
                    "attention_weights": avg_attention[idx].tolist(),
                    "top_attended_tokens": [tokens[i] for i in np.argsort(avg_attention[idx])[::-1][:3]]
                }

        return {
            "tokens": tokens,
            "attention_matrix": img_base64,
            "keyword_attention": keyword_attention
        }
    except Exception as e:
        logger.error(f"Error analyzing attention: {e}")
        return {"error": str(e)}

def rag_answer(query: str, top_k: int = settings.TOP_K) -> dict:
    try:
        logger.info(f"Processing query: {query}")
        # Encode query (cached)
        q_emb = encode_query(query)

        # Search top-k contexts
        D, I = index.search(q_emb, top_k)
        retrieved_contexts = [contexts[i] for i in I[0]]
        logger.info(f"Retrieved {len(retrieved_contexts)} contexts")

        # Get corresponding responses
        retrieved_pairs = []
        for ctx in retrieved_contexts:
            for resp in context_to_responses[ctx]:
                retrieved_pairs.append([ctx, resp])

        # Build prompt
        prompt = (
            "You are a helpful and empathetic counselor.\n\n"
            "Here are some similar cases with responses:\n"
            + "\n\n".join(f"Context: {ctx}\nResponse: {resp}" for ctx, resp in retrieved_pairs)
            + f"\n\nNow answer the new query:\n{query}\nResponse:"
        )

        # Call Gemini with retry logic
        rag_response = None
        for attempt in range(3):
            try:
                response = llm.generate_content(prompt)
                rag_response = response.text
                logger.info("Generated response successfully")
                break
            except Exception as e:
                logger.warning(f"Gemini API attempt {attempt + 1} failed: {e}")
                if attempt == 2:
                    logger.error("Failed to generate response after 3 attempts")
                    rag_response = "Error: Could not generate response. Please try again."

        # Get pure Gemini response
        gemini_response = pure_gemini(query)

        # Use first retrieved response as ground truth for evaluation
        ground_truth = retrieved_pairs[0][1] if retrieved_pairs else ""

        # Calculate metrics
        rag_metrics = calculate_metrics(ground_truth, rag_response)
        gemini_metrics = calculate_metrics(ground_truth, gemini_response)

        # Analyze attention
        attention_result = analyze_attention(query)

        return {
            "query": query,
            "retrieved": retrieved_pairs,
            "generated": rag_response,
            "evaluation": {
                "rag_response": rag_response,
                "gemini_response": gemini_response,
                "ground_truth": ground_truth,
                "rag_metrics": rag_metrics,
                "gemini_metrics": gemini_metrics
            },
            "attention_analysis": attention_result
        }
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return {
            "query": query,
            "retrieved": [],
            "generated": f"Error: {str(e)}",
            "disclaimer": "This is not medical advice. Consult a professional for mental health support.",
            "evaluation": {"error": str(e)},
            "attention_analysis": {"error": str(e)}
        }
