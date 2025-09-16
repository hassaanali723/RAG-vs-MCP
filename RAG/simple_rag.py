import os
import math
import argparse
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any

from dotenv import load_dotenv
from PyPDF2 import PdfReader
import google.generativeai as genai
from rich import print
import numpy as np


INDEX_DIR = Path("storage")


def extract_text(pdf_path: Path) -> str:
	reader = PdfReader(str(pdf_path))
	texts: List[str] = []
	for page in reader.pages:
		texts.append(page.extract_text() or "")
	return "\n".join(texts)


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 150) -> List[str]:
	chunks: List[str] = []
	start = 0
	while start < len(text):
		end = min(start + chunk_size, len(text))
		chunks.append(text[start:end])
		if end == len(text):
			break
		start = end - overlap
	return chunks


def embed(text: str) -> List[float]:
	# Use Gemini embedding model for text
	resp = genai.embed_content(model="models/text-embedding-004", content=text)
	return resp["embedding"]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
	dot = float(np.dot(a, b))
	norm_a = float(np.linalg.norm(a))
	norm_b = float(np.linalg.norm(b))
	if norm_a == 0.0 or norm_b == 0.0:
		return 0.0
	return dot / (norm_a * norm_b)


class PersistentIndex:
	def __init__(self, name: str):
		self.name = name
		self.dir = INDEX_DIR / name
		self.vectors_path = self.dir / "vectors.npy"
		self.meta_path = self.dir / "meta.json"
		self.dir.mkdir(parents=True, exist_ok=True)
		self.chunks: List[str] = []
		self.embeddings: np.ndarray | None = None

	def exists(self) -> bool:
		return self.vectors_path.exists() and self.meta_path.exists()

	def load(self):
		self.embeddings = np.load(self.vectors_path)
		with open(self.meta_path, "r", encoding="utf-8") as f:
			meta = json.load(f)
		self.chunks = meta["chunks"]

	def build(self, chunks: List[str]):
		self.chunks = chunks
		vecs = [embed(ch) for ch in chunks]
		self.embeddings = np.array(vecs, dtype=np.float32)
		np.save(self.vectors_path, self.embeddings)
		with open(self.meta_path, "w", encoding="utf-8") as f:
			json.dump({"chunks": chunks}, f)

	def search(self, query: str, top_k: int = 5) -> List[Tuple[int, str, float]]:
		if self.embeddings is None:
			raise RuntimeError("Index not loaded")
		q = np.array(embed(query), dtype=np.float32)
		scores = self.embeddings @ q / (
			np.linalg.norm(self.embeddings, axis=1) * (np.linalg.norm(q) + 1e-12)
		)
		idxs = np.argsort(-scores)[:top_k]
		return [(int(i), self.chunks[int(i)], float(scores[int(i)])) for i in idxs]


def answer_with_gemini(question: str, contexts: List[str]) -> str:
	model = genai.GenerativeModel("gemini-1.5-flash")
	context_text = "\n\n".join(contexts)
	prompt = (
		"You are a helpful assistant. Answer using ONLY the provided context."
		" If unsure, say you don't know.\n\n"
		f"Context:\n{context_text}\n\nQuestion: {question}\nAnswer:"
	)
	resp = model.generate_content(prompt)
	return resp.text


def main():
	load_dotenv()
	parser = argparse.ArgumentParser(description="Simple Gemini RAG over a PDF (with persistent index)")
	parser.add_argument("question", type=str, help="Your question")
	parser.add_argument("--pdf", type=str, default="data/10P_Source_Code_Guidelines.pdf")
	parser.add_argument("--k", type=int, default=5)
	parser.add_argument("--chunk_size", type=int, default=1000)
	parser.add_argument("--overlap", type=int, default=150)
	parser.add_argument("--index", type=str, default="guidelines")
	parser.add_argument("--rebuild", action="store_true", help="Rebuild the index from the PDF")
	args = parser.parse_args()

	api_key = os.getenv("GEMINI_API_KEY")
	if not api_key:
		raise RuntimeError("Set GEMINI_API_KEY in .env")
	genai.configure(api_key=api_key)

	pdf_path = Path(args.pdf)
	if not pdf_path.exists():
		raise FileNotFoundError(f"PDF not found: {pdf_path}")

	index = PersistentIndex(args.index)
	if args.rebuild or not index.exists():
		text = extract_text(pdf_path)
		chunks = chunk_text(text, chunk_size=args.chunk_size, overlap=args.overlap)
		index.build(chunks)
	else:
		index.load()

	top = index.search(args.question, top_k=args.k)
	contexts = [c for _, c, _ in top]
	answer = answer_with_gemini(args.question, contexts)
	print("\n[bold]Answer[/bold]:", answer, "\n")
	print("[dim]Top contexts:[/dim]")
	for idx, (i, _, score) in enumerate(top, 1):
		print(f" {idx}. chunk={i}, score={score:.3f}")


if __name__ == "__main__":
	main()
