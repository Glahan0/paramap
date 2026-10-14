"""
Dynamic Domain Discovery and Activation Signature Validation.
Discovers functional domains via LLM-guided exploration and validates them
against real model activation signatures.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Callable, Any
import json
import numpy as np

@dataclass
class DomainDefinition:
    domain_id: str
    name_en: str
    name_tr: str
    description: str
    seed_prompts_en: List[str] = field(default_factory=list)
    seed_prompts_tr: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    activation_signature: Optional[List[float]] = None  # Block-importance signature

    def all_prompts(self) -> List[str]:
        return self.seed_prompts_en + self.seed_prompts_tr

    def to_dict(self) -> Dict:
        return {
            "domain_id": self.domain_id,
            "name_en": self.name_en,
            "name_tr": self.name_tr,
            "description": self.description,
            "seed_prompts_en": self.seed_prompts_en,
            "seed_prompts_tr": self.seed_prompts_tr,
            "keywords": self.keywords,
            "has_signature": self.activation_signature is not None,
        }

class DomainDiscovery:
    """
    Manages LLM-guided dynamic domain synthesis, merging, and activation signature validation.
    """

    def __init__(self):
        self.domains: Dict[str, DomainDefinition] = {}

    def add_domain(self, domain: DomainDefinition) -> None:
        self.domains[domain.domain_id] = domain

    def get_domain(self, domain_id: str) -> Optional[DomainDefinition]:
        return self.domains.get(domain_id)

    def list_domains(self) -> List[DomainDefinition]:
        return list(self.domains.values())

    def create_bootstrap_domains(self) -> List[DomainDefinition]:
        """
        Creates foundational seed domains in both English and Turkish.
        Used as dynamic bootstrapping seeds when running offline or initializing.
        """
        bootstrap = [
            DomainDefinition(
                domain_id="math_formal",
                name_en="Formal Mathematics & Calculations",
                name_tr="Biçimsel Matematik ve Hesaplamalar",
                description="Algebraic equations, calculus, proofs, numerical logic, and arithmetic.",
                seed_prompts_en=[
                    "Calculate the integral of x^3 * e^(2x) dx with respect to x.",
                    "Prove that the square root of 2 is irrational using contradiction.",
                    "Solve the system of differential equations dy/dx = 2y - 4x, y(0) = 1.",
                    "What is the eigenvalue decomposition of a 3x3 symmetric matrix?",
                ],
                seed_prompts_tr=[
                    "x^3 * e^(2x) ifadesinin x'e göre integralini adım adım çöz.",
                    "Karekök 2 sayısının irrasyonel olduğunu çelişki yöntemiyle ispatla.",
                    "dy/dx = 2y - 4x diferansiyel denklemini y(0)=1 başlangıç koşulu ile çöz.",
                    "3x3 simetrik bir matrisin özdeğer ve özvektörlerini nasıl hesaplarız?",
                ],
                keywords=["integral", "derivative", "matrix", "eigenvalue", "proof", "ispat", "denklem", "fonksiyon", "türev", "integral", "kalkülüs", "calculus", "limit"],
            ),
            DomainDefinition(
                domain_id="code_python_system",
                name_en="Systems & Algorithmic Programming",
                name_tr="Sistem ve Algoritmik Programlama",
                description="Software engineering, algorithms, concurrency, low-level data structures.",
                seed_prompts_en=[
                    "Write an async task scheduler in Python using asyncio.PriorityQueue.",
                    "Implement a memory-mapped lock-free ring buffer in C++.",
                    "How to debug memory leaks in PyTorch CUDA pinned memory buffers?",
                    "Implement an LRU cache with O(1) get and put time complexity.",
                ],
                seed_prompts_tr=[
                    "Python'da asyncio.PriorityQueue kullanarak asenkron bir iş zamanlayıcı yaz.",
                    "C++ ile paylaşımlı bellek (shared memory) üzerinde kilitiz (lock-free) dairesel arabellek tasarla.",
                    "PyTorch CUDA pinned memory sızıntılarını tespit etmek için hangi adımlar izlenir?",
                    "O(1) sürede get ve put yapan bir LRU Cache sınıfı yaz.",
                ],
                keywords=["python", "c++", "code", "algorithm", "kod", "fonksiyon", "yazılım", "queue", "cache", "thread", "async", "def ", "class ", "pointer", "bellek", "arabellek"],
            ),
            DomainDefinition(
                domain_id="general_conversation",
                name_en="General Conversational Dialogue & Commonsense",
                name_tr="Genel Sohbet ve Günlük Yaşam",
                description="Social interaction, opinions, casual chit-chat, and common sense advice.",
                seed_prompts_en=[
                    "How was your day? Could you recommend a relaxing evening routine?",
                    "What are some fun things to do on a rainy weekend with friends?",
                    "Can you share tips for maintaining focus while working from home?",
                ],
                seed_prompts_tr=[
                    "Günün nasıl geçti? Akşam dinlenmek için güzel bir rutin önerir misin?",
                    "Yağmurlu bir hafta sonunda arkadaşlarla evde yapılabilecek keyifli aktiviteler neler?",
                    "Evden çalışırken odaklanmayı yüksek tutmak için pratik ipuçları paylaşır mısın?",
                ],
                keywords=["nasılsın", "günaydın", "teşekkürler", "routine", "weekend", "arkadaş", "tavsiye"],
            ),
            DomainDefinition(
                domain_id="creative_writing",
                name_en="Creative & Literary Writing",
                name_tr="Yaratıcı ve Edebi Yazım",
                description="Poetry, storytelling, metaphorical prose, world-building, and artistic expression.",
                seed_prompts_en=[
                    "Write a melancholic poem about an ancient lighthouse forgotten on a misty shoreline.",
                    "Describe a futuristic city built inside an extinct volcano using vivid sensory imagery.",
                ],
                seed_prompts_tr=[
                    "Sisli bir kıyıda unutulmuş asırlık bir deniz feneri hakkında hüzünlü bir şiir yaz.",
                    "Sönmüş bir volkanın kraterinde kurulmuş fütüristik bir şehri duyusal imgelerle betimle.",
                ],
                keywords=["şiir", "öykü", "metafor", "betimleme", "poem", "story", "rüzgar", "hüzün"],
            ),
        ]

        for d in bootstrap:
            self.add_domain(d)
        return bootstrap

    def validate_and_prune(
        self,
        similarity_threshold: float = 0.88,
        min_variance: float = 1e-4
    ) -> Dict[str, Any]:
        """
        Validates domain signatures.
        1. Checks variance of activation signature (drops domains indistinguishable from noise).
        2. Computes pairwise cosine similarity between signatures.
        3. Merges redundant domains whose signatures are too similar.
        """
        domain_ids = [d_id for d_id, d in self.domains.items() if d.activation_signature is not None]
        if len(domain_ids) < 2:
            return {"status": "insufficient_signatures", "domain_count": len(domain_ids)}

        matrix = np.array([self.domains[d_id].activation_signature for d_id in domain_ids], dtype=np.float32)

        # Variance check
        variances = np.var(matrix, axis=1)
        pruned_low_var = []
        for i, d_id in enumerate(domain_ids):
            if variances[i] < min_variance:
                pruned_low_var.append(d_id)

        for d_id in pruned_low_var:
            del self.domains[d_id]

        # Re-evaluate remaining
        active_ids = [d_id for d_id in domain_ids if d_id not in pruned_low_var]
        if len(active_ids) < 2:
            return {"status": "done", "pruned_low_variance": pruned_low_var, "merged_pairs": []}

        active_matrix = np.array([self.domains[d_id].activation_signature for d_id in active_ids], dtype=np.float32)
        # Normalize
        norms = np.linalg.norm(active_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normed = active_matrix / norms
        sim_matrix = np.dot(normed, normed.T)

        merged_pairs = []
        merged_away = set()

        for i in range(len(active_ids)):
            if active_ids[i] in merged_away:
                continue
            for j in range(i + 1, len(active_ids)):
                if active_ids[j] in merged_away:
                    continue
                if sim_matrix[i, j] > similarity_threshold:
                    keep_id = active_ids[i]
                    drop_id = active_ids[j]
                    # Merge prompts from drop into keep
                    self.domains[keep_id].seed_prompts_en.extend(self.domains[drop_id].seed_prompts_en)
                    self.domains[keep_id].seed_prompts_tr.extend(self.domains[drop_id].seed_prompts_tr)
                    self.domains[keep_id].keywords.extend(self.domains[drop_id].keywords)
                    merged_away.add(drop_id)
                    merged_pairs.append((drop_id, keep_id, float(sim_matrix[i, j])))

        for drop_id in merged_away:
            del self.domains[drop_id]

        return {
            "status": "success",
            "pruned_low_variance": pruned_low_var,
            "merged_pairs": merged_pairs,
            "remaining_domains": list(self.domains.keys()),
        }

    def save_to_json(self, file_path: str) -> None:
        data = {d_id: d.to_dict() for d_id, d in self.domains.items()}
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load_from_json(self, file_path: str) -> None:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.domains.clear()
        for d_id, item in data.items():
            self.domains[d_id] = DomainDefinition(
                domain_id=item["domain_id"],
                name_en=item["name_en"],
                name_tr=item["name_tr"],
                description=item["description"],
                seed_prompts_en=item.get("seed_prompts_en", []),
                seed_prompts_tr=item.get("seed_prompts_tr", []),
                keywords=item.get("keywords", []),
            )
