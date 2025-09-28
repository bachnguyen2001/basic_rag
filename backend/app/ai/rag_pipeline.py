import logging
import os
import asyncio
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import faiss
import google.generativeai as genai
from collections import defaultdict
from functools import lru_cache
from transformers import AutoTokenizer, AutoModel
import torch
import pandas as pd
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from ..core.config import settings
import time
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Download NLTK data
nltk.download("punkt", quiet=True)

# Configure Gemini
try:
    genai.configure(api_key=settings.GEMINI_API_KEY)
    llm = genai.GenerativeModel(settings.GENERATION_MODEL)
    logger.info("Initialized Gemini API")
except Exception as e:
    logger.error(f"Failed to initialize Gemini API: {e}")
    raise

# Paths for caching
DATASET_CACHE = "mental_health_dataset.parquet"
EMBEDDINGS_CACHE = "context_embeddings.npy"
INDEX_CACHE = "faiss_index.bin"

# Load or preprocess dataset
def load_or_preprocess_dataset():
    if os.path.exists(DATASET_CACHE):
        logger.info("Loading cached dataset...")
        df = pd.read_parquet(DATASET_CACHE)
    else:
        logger.info("Loading and preprocessing dataset...")
        ds = load_dataset("Amod/mental_health_counseling_conversations")
        train = ds["train"]
        df = pd.DataFrame({"Context": train["Context"], "Response": train["Response"]})

        # Clean dataset
        df = df.dropna(subset=['Context', 'Response'])
        df = df[df['Context'].str.strip() != ""]
        df = df[df['Response'].str.strip() != ""]
        df['Context'] = df['Context'].str.strip()
        df['Response'] = df['Response'].str.strip()
        df = df.drop_duplicates(subset=['Context', 'Response']).reset_index(drop=True)
        
        # Save to cache
        df.to_parquet(DATASET_CACHE)
        logger.info(f"Saved cleaned dataset to {DATASET_CACHE}")
    
    return df

# Initialize embedding model and FAISS index
def initialize_embeddings_and_index(df, use_ivf=False):
    embed_model = SentenceTransformer(settings.EMBEDDING_MODEL)
    d = embed_model.get_sentence_embedding_dimension()
    
    if (os.path.exists(EMBEDDINGS_CACHE) and 
        os.path.exists(INDEX_CACHE) and 
        np.load(EMBEDDINGS_CACHE).shape[1] == d):
        logger.info("Loading cached embeddings and FAISS index...")
        context_embeddings = np.load(EMBEDDINGS_CACHE)
        index = faiss.read_index(INDEX_CACHE)
    else:
        logger.info("Generating embeddings and FAISS index...")
        context_embeddings = embed_model.encode(df['Context'].tolist(), convert_to_numpy=True, show_progress_bar=True)
        faiss.normalize_L2(context_embeddings)
        
        d = context_embeddings.shape[1]
        if use_ivf:
            nlist = min(100, len(df))  # Number of clusters for IVF index
            quantizer = faiss.IndexFlatIP(d)
            index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
            index.train(context_embeddings)
        else:
            index = faiss.IndexFlatIP(d)
        
        index.add(context_embeddings)
        
        # Save to cache
        np.save(EMBEDDINGS_CACHE, context_embeddings)
        faiss.write_index(index, INDEX_CACHE)
        logger.info(f"Saved embeddings to {EMBEDDINGS_CACHE} and index to {INDEX_CACHE}")
    
    return embed_model, context_embeddings, index

# Load dataset and initialize
try:
    df = load_or_preprocess_dataset()
    contexts = df['Context'].tolist()
    context_to_responses = defaultdict(list)
    for c, r in zip(df['Context'], df['Response']):
        context_to_responses[c].append(r)
    
    # Use IndexFlatIP for small datasets, switch to IndexIVFFlat for large datasets
    embed_model, context_embeddings, index = initialize_embeddings_and_index(df, use_ivf=len(df) > 10000)
    logger.info(f"Loaded {len(contexts)} unique contexts and initialized FAISS index")
except Exception as e:
    logger.error(f"Failed to initialize: {e}")
    raise

# Initialize tokenizer and model for attention analysis (optional)
try:
    tokenizer = AutoTokenizer.from_pretrained(settings.EMBEDDING_MODEL)
    transformer_model = AutoModel.from_pretrained(settings.EMBEDDING_MODEL, output_attentions=True)
    transformer_model.eval()  # Set to evaluation mode
    logger.info("Initialized transformer model for attention analysis")
except Exception as e:
    logger.error(f"Failed to initialize transformer model: {e}")
    transformer_model = None

async def encode_query(query: str) -> np.ndarray:
    loop = asyncio.get_event_loop()
    logger.info(f"Encoding query: {query}")
    q_emb = await loop.run_in_executor(None, lambda: embed_model.encode([query], convert_to_numpy=True))
    faiss.normalize_L2(q_emb)
    return q_emb

async def pure_gemini(query: str, max_retries=3) -> str:
    for attempt in range(max_retries):
        try:
            prompt = f"Answer empathetically as a counselor: {query}\nResponse:"
            response = await asyncio.wait_for(llm.generate_content_async(prompt), timeout=30.0)
            return response.text
        except BaseException as e:
            logger.warning(f"Gemini API attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
            else:
                logger.error("Failed to generate response after retries")
                return f"Error: {str(e)}"

def calculate_metrics(reference: str, candidate: str, include_rouge2=False) -> dict:
    try:
        ref_tokens = [nltk.word_tokenize(reference)]
        cand_tokens = nltk.word_tokenize(candidate)
        bleu = sentence_bleu(ref_tokens, cand_tokens, smoothing_function=SmoothingFunction().method1)
        rouge_types = ['rouge1', 'rougeL']
        if include_rouge2:
            rouge_types.append('rouge2')
        scorer = rouge_scorer.RougeScorer(rouge_types, use_stemmer=True)
        rouge = scorer.score(reference, candidate)
        return {
            "bleu": bleu,
            "rouge": {rt: rouge[rt].fmeasure for rt in rouge_types}
        }
    except Exception as e:
        logger.error(f"Error calculating metrics: {e}")
        return {"bleu": 0.0, "rouge": {rt: 0.0 for rt in rouge_types}}

async def rag_answer(query: str, top_k: int = settings.TOP_K, include_attention: bool = False, include_rouge2: bool = False) -> dict:
    prompt = "" 
    try:
        logger.info(f"Processing query: {query}")
        q_emb = await encode_query(query)

        # Search top-k contexts
        D, I = index.search(q_emb, top_k)
        logger.info(f"Retrieved {len(I[0])} contexts")

        # Deduplicate context-response pairs with scores
        seen_pairs = set()
        retrieved_pairs = []

        for rank, ctx_idx in enumerate(I[0]):
            ctx = contexts[ctx_idx]
            score = float(D[0][rank])  # inner product score
            for resp in context_to_responses[ctx]:
                pair = (ctx, resp)
                if pair not in seen_pairs:
                    retrieved_pairs.append({
                        "context": ctx,
                        "response": resp,
                        "score": score
                    })
                    seen_pairs.add(pair)

        logger.info(f"Prepared {len(retrieved_pairs)} unique context-response pairs with scores")

        # Build prompt
        prompt = (
            "You are a helpful and empathetic counselor.\n\n"
            "Here are some similar cases with responses:\n"
            + "\n\n".join(f"Context: {item['context']}\nResponse: {item['response']}" for item in retrieved_pairs)
            + f"\n\nNow answer the new query:\n{query}\nResponse:"
        )
        # Call Gemini with exponential backoff
        rag_response = None
        for attempt in range(3):
            try:
                response = await asyncio.wait_for(llm.generate_content_async(prompt), timeout=30.0)
                rag_response = response.text
                logger.info("Generated response successfully")
                break
            except asyncio.TimeoutError:
                logger.warning(f"Gemini API attempt {attempt + 1} timed out")
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                else:
                    rag_response = "Error: Response generation timed out."
            except BaseException as e:
                logger.error(f"Gemini API attempt {attempt + 1} failed: {repr(e)}")
                rag_response = f"Error: {repr(e)}"
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)
                else:
                    rag_response = f"Error: {str(e)}"

        # Get pure Gemini response
        gemini_response = await pure_gemini(query)

        # Use first retrieved response as ground truth
        ground_truth = retrieved_pairs[0]['response'] if retrieved_pairs else ""

        # Calculate metrics
        rag_metrics = calculate_metrics(ground_truth, rag_response, include_rouge2)
        gemini_metrics = calculate_metrics(ground_truth, gemini_response, include_rouge2)

       
        return {
            "query": query,
            "retrieved": retrieved_pairs,
            "prompt": prompt,
            "generated": rag_response,
            "evaluation": {
                "rag_response": rag_response,
                "gemini_response": gemini_response,
                "ground_truth": ground_truth,
                "rag_metrics": rag_metrics,
                "gemini_metrics": gemini_metrics
            },
            "disclaimer": "This is not medical advice. Consult a professional for mental health support."
        }
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return {
            "query": query,
            "retrieved": [],
            "prompt": prompt,
            "generated": f"Error: {str(e)}",
            "disclaimer": "This is not medical advice. Consult a professional for mental health support.",
            "evaluation": {"error": str(e)},
        }
