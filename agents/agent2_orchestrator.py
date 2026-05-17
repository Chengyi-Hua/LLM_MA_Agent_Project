"""
agents/agent2_orchestrator.py

Agent 2: Orchestration & Planning Agent (Steps A + B).
Inherits from BaseRAG.

Responsibilities:
  - Rerank chunks per section and generate a quick NLI summary for each
  - Compute asymmetric entailment probabilities across section summaries (Step A)
  - Build a DAG, resolve cycles, perform topological sort (Step B)
  - Return execution order + dependency map + summaries for Agent 3

Output format:
{
    "status"    : "success",
    "order"     : list[str],             # DAG topological execution order
    "dependency": dict[str, list[str]],  # section → list of dependency sections
    "summaries" : dict[str, str]         # section → pre-built NLI summary
}
"""
import torch  
import os
import json
import yaml
import networkx as nx
from pathlib import Path
from sentence_transformers import CrossEncoder, SentenceTransformer, util
from scipy.special import softmax
import textwrap
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time
import numpy as np
from dotenv import load_dotenv

from methods.base_rag import BaseRAG, load_config

load_dotenv()


class GraphAwareRAG(BaseRAG):
    """
    Agent 2: NLI-based dependency graph builder.
    Uses BaseRAG for LLM calls (_call_llm) and chunk reranking (_rerank_chunks).
    Adds NLI cross-encoder and DAG construction on top.
    """

    agent_key = "agent2"   # maps to config["llm"]["agent2"] in settings.yaml

    def __init__(self, config=None):
        super().__init__(config)

        # Load Agent 2 specific graph config from settings.yaml
        graph_config = self.config["llm"]["agent2"]["graph_logic"]
        model_name = graph_config["nli_model"]["model_name"]
        self.nli_threshold = graph_config["algorithm"]["threshold"]

        print(f"⏳ Loading NLI model ({model_name}) ...")
        print(f"⚙️  NLI threshold set to: {self.nli_threshold}")
        self.nli_model = CrossEncoder(model_name)
        print("✅ Agent 2 initialised.")

    '''

    # ── Override _init_llm to use agent2's own llm config block ───────────────


    # 这个函数是因为我在google colab上跑 token不够了 然后用来限制token的
    # 如果够用不用就删了
    def _call_llm(self, prompt: str, system_prompt: str = "You are a helpful assistant.") -> str:
        # 这里需要根据你使用的具体包库（比如 groq 官方库）调整
        # 假设使用的是 groq 包
        from groq import Groq
        client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            max_tokens=400, 
            temperature=0.1
        )
        return response.choices[0].message.content

'''
    # ==========================================
    # 这里我改变了rerank的方法 我选择了一种会尽量避免挑选有重复内容的chunks的方法
    def _rerank_chunks_with_mmr(self, raw_chunks: list, query: str, top_k: int = 5, lambda_mult: float = 0.5) -> list:
        if not raw_chunks:
            return []

        # ==========================================
        # 阶段 1：扩大海选池 (召回前 40 名候选者)
        # ==========================================
        # 注意：需要根据现有的 base_rag 评分接口调整这里
        pairs = [[query, chunk["text"]] for chunk in raw_chunks]
        scores = self.nli_model.predict(pairs)

        for i, chunk in enumerate(raw_chunks):
            # 提取数组中的第 2 个值（索引为 1 的 Entailment 分数）
            chunk["relevance_score"] = float(scores[i][1])

        # 按相关性从高到低排序，截取前 40 名作为我们的“待挑池”
        candidates = sorted(raw_chunks, key=lambda x: x["relevance_score"], reverse=True)[:40]

        if len(candidates) <= top_k:
            return candidates[:top_k]

        # ==========================================
        # 阶段 2：数据归一化
        # ==========================================
        # 将相关性得分映射到 0~1 之间，以便后续与相似度完美对抗
        relevance_scores = [c["relevance_score"] for c in candidates]
        min_score = min(relevance_scores)
        max_score = max(relevance_scores)

        # 避免分母为0
        if max_score == min_score:
            normalized_relevance = [1.0] * len(candidates)
        else:
            normalized_relevance = [(s - min_score) / (max_score - min_score) for s in relevance_scores]

        # ==========================================
        # 阶段 3：计算冗余度矩阵 (计算所有候选者两两之间的相似度)
        # ==========================================
        texts = [c["text"] for c in candidates]
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(texts)
        # 生成一个 20x20 的矩阵，代表谁和谁说了同样的话
        similarity_matrix = cosine_similarity(tfidf_matrix)

        # ==========================================
        # 阶段 4：MMR 核心挑选逻辑
        # ==========================================
        selected_indices = []
        unselected_indices = list(range(len(candidates)))

        
        selected_indices.append(unselected_indices.pop(0))

        # 挑选剩下的 top_k - 1 个名额
        while len(selected_indices) < top_k and unselected_indices:
            best_mmr_score = -np.inf
            best_index_to_pick = -1

            for unselected_idx in unselected_indices:
                # 1. 拿它的【相关性】
                rel_score = normalized_relevance[unselected_idx]

                # 2. 算它的【冗余度】(它跟已经挑进篮子里的 Chunk，最像的那个有多像？)
                max_sim_with_selected = max(
                    [similarity_matrix[unselected_idx][sel_idx] for sel_idx in selected_indices]
                )

                # 3. MMR 公式
                # 分数 = (权重 * 相关性) - (剩余权重 * 冗余度)
                mmr_score = (lambda_mult * rel_score) - ((1 - lambda_mult) * max_sim_with_selected)

                if mmr_score > best_mmr_score:
                    best_mmr_score = mmr_score
                    best_index_to_pick = unselected_idx

        
            selected_indices.append(best_index_to_pick)
            unselected_indices.remove(best_index_to_pick)

        # 根据选出的索引，打包最终的 chunks 返还给 Agent 3 和 Summary 函数
        return [candidates[i] for i in selected_indices]

    def _init_llm(self):
        """
        Agent 2 has its own LLM config under config["llm"]["agent2"]["llm"],
        separate from the method-level configs.
        """
        agent2_llm = self.config["llm"]["agent2"]["llm"]
        provider   = agent2_llm["provider"]
        model      = agent2_llm["model"]

        if provider == "openai":
            from openai import OpenAI
            api_key = os.getenv(self.config["api_keys"]["openai_env"])
            return OpenAI(api_key=api_key), model

        elif provider == "groq":
            from groq import Groq
            api_key = os.getenv(self.config["api_keys"]["groq_env"])
            return Groq(api_key=api_key), model

        elif provider == "openrouter":
            from openai import OpenAI
            api_key = os.getenv(self.config["api_keys"]["openrouter_env"])
            return OpenAI(
                api_key=api_key,
                base_url="https://openrouter.ai/api/v1"
            ), model

        else:
            raise ValueError(f"Unknown provider: {provider}")

    # ── Core methods ──────────────────────────────────────────────────────────

    def _generate_quick_summary(self, section_name: str, reranked_chunks: list) -> str:
        if not reranked_chunks: return ""

        # 打印调试信息保持不变
        print(f"\n🔍 [调试信息] 为章节 '{section_name}' 选中的 Chunks:")
        for i, c in enumerate(reranked_chunks):
            clean_text = re.sub(r'\s+', ' ', c['text']).strip()
            short_text = clean_text[:200] + ("..." if len(clean_text) > 200 else "")
            formatted_text = textwrap.fill(short_text, width=80, initial_indent="      ", subsequent_indent="      ")
            print(f"   📄 Chunk {i+1}:\n{formatted_text}\n" + "-" * 40)

        # 组合并截断文本 (防限流)
        combined_text = "\n\n".join([c["text"] for c in reranked_chunks])
        combined_text = combined_text[:4000]

        system_prompt = f"""You are a sterile, automated data-extraction API. Your output feeds directly into a programmatic pipeline. 
You possess ZERO conversational abilities. You do not greet, you do not explain, you do not apologize.

CRITICAL RULES:
1. OUTPUT ONLY A SINGLE CONTINUOUS PARAGRAPH. 
2. NO bullet points (1., 2., -, *). NO line breaks. 
3. ZERO CONVERSATIONAL FILLER. (BANNED PHRASES: "Based on the provided documents", "Here are the extracted facts", "I found", "However").
4. If there is no exact matching info for the Topic, output EXACTLY THIS STRING: "No relevant information regarding {section_name} is available." """

        # 2：User Prompt + 强制示例 (Few-Shot)
        prompt = f"""Topic: [{section_name}]

Source Documents:
{combined_text}

=== EXPECTED OUTPUT EXAMPLES ===
Example for a successful match (Starts immediately with facts, dense text):
Nishinoshima is a stratovolcano rising 3,000 meters from the ocean floor. It consists of basalt and andesite magmas, with the caldera almost filled in by 1973. The volcano provides a unique opportunity to study island-forming processes.

Example for NO match (Output this exact string):
No relevant information regarding {section_name} is available.
================================

Output your final paragraph below:"""

        # 调用 LLM，注入我们刚刚写的严格的 System Prompt
        return self._call_llm(prompt, system_prompt)
    
    # 修改点 3：新增核心判别方法
    # 修改点 3：新增核心判别方法
    # 修改点 3：新增核心判别方法
    # 修改点 3：新增核心判别方法
    # 修改点 3：新增核心判别方法
    def _check_dependency(self, sec_a_name: str, sec_a_text: str, sec_b_name: str, sec_b_text: str) -> bool:
        print(f"\n🔍 正在分析: [{sec_a_name}] -> [{sec_b_name}]")

        # --- 关卡 1：NLI 快速确诊 (CrossEncoder) ---
        logits = self.nli_model.predict([(sec_a_text, sec_b_text)])[0]
        score_contradiction = logits[0]
        score_entailment = logits[1]
        score_neutral = logits[2]

        print(f"   📊 [关卡1 NLI 得分] 矛盾:{score_contradiction:.2f} | 顺延:{score_entailment:.2f} | 中立:{score_neutral:.2f}")

        if score_entailment > score_contradiction and score_entailment > 1.5:
            print("   ✅ [关卡1 确诊] 逻辑顺延分数极高，无需打扰大模型，直接建立依赖！")
            return True

        # --- 关卡 2：LLM 终极裁判 (大语言模型) ---
        print(f"   🤖 [关卡2 介入] NLI 拿不准，呼叫大模型裁判综合评估 [{sec_a_name}] 与 [{sec_b_name}] 的关系...")

        # 升级版 Prompt：引入 RELATED 选项
        prompt = f"""Topic A: 【{sec_a_name}】
        Summary A: {sec_a_text}

        Topic B: 【{sec_b_name}】
        Summary B: {sec_b_text}

        Question: Analyze the relationship between Topic A and Topic B. Choose EXACTLY ONE of the following three categories:
        1. "PREREQUISITE": Understanding Topic A is strictly required before learning Topic B (A determines B).
        2. "RELATED": They are interconnected and mutually reference each other, but there is no strict chronological or prerequisite order. They just shouldn't contradict each other.
        3. "NONE": They are practically independent.

        Output exactly and only one word: "PREREQUISITE", "RELATED", or "NONE"."""

        llm_response = self._call_llm(prompt, "You are a logical relationship classifier. Output only one word.")
        clean_response = llm_response.strip().upper()

        if "PREREQUISITE" in clean_response:
            print(f"   ⚖️ [LLM 判决] PREREQUISITE (严格先后依赖)")
            return "PREREQUISITE"
        elif "RELATED" in clean_response:
            print(f"   ⚖️ [LLM 判决] RELATED (互为参考，防冲突)")
            return "RELATED"
        else:
            print(f"   ⚖️ [LLM 判决] NONE (无关联)")
            return "NONE"

    def _build_nli_graph(self, summaries: dict) -> dict:
        sections = list(summaries.keys())
        summary_texts = list(summaries.values())

        dag = nx.DiGraph()
        dag.add_nodes_from(sections)

        # 临时记录互相参考的兄弟对
        related_pairs = []

        for i, sec_a in enumerate(sections):
            for j, sec_b in enumerate(sections):
                if i == j: continue

                # 调用裁判进行分类
                relation = self._check_dependency(sec_a, summary_texts[i], sec_b, summary_texts[j])

                if relation == "PREREQUISITE":
                    dag.add_edge(sec_a, sec_b, weight=1.0)
                elif relation == "RELATED":
                    # 记录这对兄弟 (避免重复记录，比如记录了 A,B 就不再记录 B,A)
                    if (sec_b, sec_a) not in related_pairs:
                        related_pairs.append((sec_a, sec_b))

        # 破环逻辑
        while not nx.is_directed_acyclic_graph(dag):
            cycle = nx.find_cycle(dag)
            min_edge = min(cycle, key=lambda e: dag.edges[e[0], e[1]]['weight'])
            dag.remove_edge(*min_edge)

        # 1. 获取基础的排版顺序和依赖表
        final_order = list(nx.topological_sort(dag))
        dependencies = {n: list(dag.predecessors(n)) for n in dag.nodes()}

        # 2. 动态分配 RELATED 关系
        for a, b in related_pairs:
            # 查一下在最终排版里，谁在前面？
            idx_a = final_order.index(a)
            idx_b = final_order.index(b)

            if idx_a < idx_b:
                # A 既然排在 B 前面，就把 A 塞进 B 的依赖表里（让 B 写的时候参考 A）
                if a not in dependencies[b]:
                    dependencies[b].append(a)
            else:
                # 反之亦然
                if b not in dependencies[a]:
                    dependencies[a].append(b)

        
        return {
            "order": final_order,
            "map": dependencies,  
        }
    
    def generate(self, input_data: dict) -> dict:
        island_name, sections_data = self._parse_input(input_data)

        print("\n🚀 阶段 1：正在进行 rerank 并提取摘要...")
        summaries = {}
        reranked_data_store = {}
        for section, data in sections_data.items():
            # 1. 抓取最相关的 chunks
            best_chunks = self._rerank_chunks_with_mmr(
            data.get("chunks", []),
            query=section,
            top_k=5,
            lambda_mult=0.5  # 👈 你可以在这里自由调参！
        )
            # 2. 把选中的 chunks 存进字典备查
            reranked_data_store[section] = best_chunks
            # 3. 让 LLM 生成摘要
            summaries[section] = self._generate_quick_summary(section, best_chunks)

            print(f"   ⏳ [{section}] 摘要生成完毕。")



        print("\n🚀 阶段 2：正在分析 NLI 逻辑依赖...")
        plan = self._build_nli_graph(summaries)


        output_payload = {
            "status"    : "success",
            "island"    : island_name,
            "order"     : plan["order"],
            "dependency": plan["map"],
            "summaries" : summaries
        }
        save_dir = "logs/agent2_plans"
        os.makedirs(save_dir, exist_ok=True)
        
        save_path = os.path.join(save_dir, f"{island_name}_plan.json")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(output_payload, f, ensure_ascii=False, indent=4)
        
        print(f"💾 [Agent 2] Plan saved to: {save_path}")

        return output_payload

# 修改点 增加evaluation的文件
def evaluate_graph_metrics(blueprint_path):
    # 1. 读取 Agent 2 生成的原始图纸
    with open(blueprint_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    dependency = data.get("dependency", {})
    nodes = list(dependency.keys())
    total_nodes = len(nodes)

    if total_nodes == 0:
        return "没有节点数据"

    # 指标 1：孤立节点率 (Orphan Node Ratio)
    # 没有任何前置依赖的节点就是孤立节点（或者根节点）
    orphan_nodes = [node for node, deps in dependency.items() if len(deps) == 0]
    orphan_ratio = len(orphan_nodes) / total_nodes

    # 指标 2：图的深度 (Graph Depth)
    # 用递归计算最长逻辑链
    memo = {}
    def get_depth(node):
        if node in memo: return memo[node]
        deps = dependency.get(node, [])
        if not deps:
            memo[node] = 1
            return 1
        # 当前节点的深度 = 它所有依赖节点的最大深度 + 1
        max_dep_depth = max([get_depth(d) for d in deps])
        memo[node] = max_dep_depth + 1
        return memo[node]

    max_depth = max([get_depth(n) for n in nodes]) if nodes else 0

    # 3. 组装测评报告
    evaluation_report = {
        "graph_analysis": {
            "total_sections": total_nodes,
            "orphan_nodes_count": len(orphan_nodes),
            "orphan_nodes_list": orphan_nodes,
            "orphan_node_ratio": f"{orphan_ratio:.2%}", # 转成百分比
            "max_graph_depth": max_depth
        }
    }

    # 4. 导出为单独的测评文件
    report_filename = "evaluation_graph_metrics.json"
    with open(report_filename, "w", encoding="utf-8") as f:
        json.dump(evaluation_report, f, indent=4, ensure_ascii=False)

    print(f"✅ 图谱测评报告已生成: {report_filename}")
    
    print(json.dumps(evaluation_report, indent=4, ensure_ascii=False))
    return evaluation_report


# ==========================================
# 启动实验
# ==========================================
if __name__ == "__main__":
    DATA_PATH = "data/Nishinoshima_(Ogasawara)_rag_context.json"

    if os.path.exists(DATA_PATH):
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            full_data = json.load(f)

        # 明确传入 NLI 阈值
        rag = GraphAwareRAG(nli_threshold=0.1)
        final_result = rag.generate(full_data)

        import json
        print("\n=== 真实 generate 函数的完整返回结果 ===")
        print(json.dumps(final_result, indent=4, ensure_ascii=False))
        # ========================================
        print("\n" + "="*50)
        print("🏆 最终拓扑排序结果")
        print("="*50)
        for i, s in enumerate(final_result["order"]):
            print(f"Step {i+1}: {s}")
    else:
        print(f"❌ 找不到数据文件，请确保它在 {DATA_PATH}")

