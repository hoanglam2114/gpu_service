"""
services/clustering.py — Conversation clustering với multilingual-e5-base + DBSCAN + KMeans.
"""
import numpy as np
import torch
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN, KMeans

from app.config import config


class ClusteringService:
    def __init__(self, num_clusters: int = None, eps: float = None, min_samples: int = None):
        self.num_clusters = num_clusters or config.NUM_CLUSTERS
        self.eps = eps or config.DBSCAN_EPS
        self.min_samples = min_samples or config.DBSCAN_MIN_SAMPLES

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[ClusteringService] Loading model on {self.device}...")
        self.model = SentenceTransformer("intfloat/multilingual-e5-base", device=self.device)
        self.model.max_seq_length = 4096
        print("[ClusteringService] Model ready.")

        self._cache: dict | None = None

    # ── Cache ─────────────────────────────────────────────────────────────────

    def _save_cache(self, data, conv_embeddings, assignments, clean_indices, final_labels, centroids):
        self._cache = {
            "data": data,
            "conv_embeddings": conv_embeddings,
            "assignments": assignments,
            "clean_indices": clean_indices,
            "final_labels": final_labels,
            "centroids": centroids,
        }
        print(f"[Cache] Đã lưu {len(data)} embeddings vào RAM.")

    def _load_cache(self) -> dict:
        if self._cache is None:
            raise RuntimeError("Chưa có cache. Hãy gọi POST /api/cluster trước.")
        return self._cache

    def clear_cache(self):
        self._cache = None

    # ── Private helpers ────────────────────────────────────────────────────────

    def _parse_conversations(self, data):
        all_pair_texts = []
        conv_to_pair_indices = []
        count = 0

        for conv in data:
            messages = conv.get("messages", [])
            pairs = []
            i = 0
            while i < len(messages):
                msg = messages[i]
                if msg.get("role") == "user":
                    user_content = msg.get("content", "").strip()
                    if i + 1 < len(messages) and messages[i + 1].get("role") == "assistant":
                        assistant_content = messages[i + 1].get("content", "").strip()
                        if user_content and assistant_content:
                            pairs.append(f"user:{user_content} assistant:{assistant_content}")
                        i += 2
                        continue
                i += 1

            if pairs:
                indices = []
                for pair_text in pairs:
                    all_pair_texts.append(pair_text)
                    indices.append(count)
                    count += 1
                conv_to_pair_indices.append(indices)
            else:
                conv_to_pair_indices.append([])

        return all_pair_texts, conv_to_pair_indices

    def _embed(self, utterances):
        prefixed = ["query: " + u for u in utterances]
        return self.model.encode(
            prefixed, show_progress_bar=True, normalize_embeddings=False, batch_size=64, device=self.device,
        ).astype(np.float32)

    def _mean_pool_and_normalize(self, utt_embeddings, conv_to_utt_indices):
        dim = utt_embeddings.shape[1]
        conv_embeddings = []
        for indices in conv_to_utt_indices:
            if not indices:
                conv_embeddings.append(np.zeros(dim, dtype=np.float32))
            else:
                conv_embeddings.append(np.mean(utt_embeddings[indices], axis=0))
        conv_embeddings = np.array(conv_embeddings, dtype=np.float32)
        norms = np.linalg.norm(conv_embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        return conv_embeddings / norms

    def _normalize_centroids(self, centroids: np.ndarray) -> np.ndarray:
        centroids = centroids.astype(np.float32)
        norms = np.linalg.norm(centroids, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        return centroids / norms

    def _build_output(self, data, assignments) -> dict:
        result_data = [{**item, "cluster": assignments[i]} for i, item in enumerate(data)]
        count_map = defaultdict(int)
        for cid in assignments:
            count_map[cid] += 1
        groups = [
            {"groupId": cid, "count": count_map[cid], "label": "Nhiễu" if cid == -1 else f"Cụm {cid}"}
            for cid in sorted(count_map.keys())
        ]
        return {"data": result_data, "assignments": assignments, "groups": groups}

    # ── Public API ─────────────────────────────────────────────────────────────

    def cluster(self, data: list[dict]) -> dict:
        if not data:
            return {"data": [], "assignments": [], "groups": []}

        print("1. Parsing & embedding utterances...")
        all_utterances, conv_to_utt_indices = self._parse_conversations(data)
        if not all_utterances:
            return {"data": [], "assignments": [], "groups": []}

        utt_embeddings = self._embed(all_utterances)

        print("2. Mean pooling...")
        conv_embeddings = self._mean_pool_and_normalize(utt_embeddings, conv_to_utt_indices)

        print(f"3. DBSCAN (eps={self.eps}, min_samples={self.min_samples})...")
        dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples, metric="cosine")
        dbscan_labels = dbscan.fit_predict(conv_embeddings)

        clean_indices = [i for i, lbl in enumerate(dbscan_labels) if lbl != -1]
        noise_indices = [i for i, lbl in enumerate(dbscan_labels) if lbl == -1]
        print(f"   -> Nhiễu: {len(noise_indices)} | Sạch: {len(clean_indices)}")

        k = max(1, min(self.num_clusters, len(clean_indices)))
        print(f"4. KMeans (K={k})...")
        kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
        final_labels = kmeans.fit_predict(conv_embeddings[clean_indices])
        centroids = self._normalize_centroids(kmeans.cluster_centers_)

        assignments = [-1] * len(data)
        for pos, orig_idx in enumerate(clean_indices):
            assignments[orig_idx] = int(final_labels[pos])

        self._save_cache(
            data=data, conv_embeddings=conv_embeddings, assignments=assignments,
            clean_indices=clean_indices, final_labels=np.array(final_labels, dtype=np.int32), centroids=centroids,
        )
        return self._build_output(data, assignments)

    def filter_by_centroid(self, threshold: float = 0.9) -> dict:
        """Loại bỏ mẫu quá gần tâm cụm (thường là mẫu lặp/template)."""
        cache = self._load_cache()
        data = cache["data"]
        conv_embeddings = cache["conv_embeddings"]
        assignments = cache["assignments"]
        clean_indices = cache["clean_indices"]
        final_labels = cache["final_labels"]
        centroids = cache["centroids"]

        print(f"[Filter] Tính cosine similarity với centroid (threshold={threshold})...")
        clean_embeddings = conv_embeddings[clean_indices]
        assigned_centroids = centroids[final_labels]
        similarities = np.sum(clean_embeddings * assigned_centroids, axis=1)

        too_close_positions = set(np.where(similarities > threshold)[0])
        kept_positions = set(np.where(similarities <= threshold)[0])
        too_close_orig_indices = {clean_indices[pos] for pos in too_close_positions}

        print(f"   -> Tổng clean: {len(clean_indices)}")
        print(f"   -> Giữ lại (similarity <= {threshold}): {len(kept_positions)}")
        print(f"   -> Loại bỏ (similarity >  {threshold}): {len(too_close_orig_indices)}")

        kept_data, kept_assignments = [], []
        for i, item in enumerate(data):
            if assignments[i] == -1 or (i not in too_close_orig_indices and assignments[i] != -1):
                kept_data.append(item)
                kept_assignments.append(assignments[i])

        return self._build_output(kept_data, kept_assignments)
