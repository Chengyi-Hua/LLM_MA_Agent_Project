import torch  # 强行先加载，防止循环引入报错
import os
import json
import yaml
import networkx as nx
from pathlib import Path
from google.colab import userdata
from sentence_transformers import CrossEncoder
from scipy.special import softmax

# 1. 注入 API Key 这里只是为了实验 所以用了不要钱的模型 后面一定要改回来
os.environ["GROQ_API_KEY"] = userdata.get('key')

# 2. 引入基类 就是把计算逻辑关系的模型 以及它的参数载入
from base_rag import BaseRAG, load_config

# Inheritance
class GraphAwareRAG(BaseRAG):
    agent_key = "method2"
    
    def __init__(self, config=None):
        # 1. 执行父类（BaseRAG）的 __init__ 方法 所以不用再重复导入setting.yaml
        super().__init__(config)
        
        # 2. 从 YAML 中读取 Agent 2 的专属配置
        agent2_config = self.config["llm"]["agent2"]
        graph_config = agent2_config["graph_logic"]
        
        # 动态获取模型名和阈值 
        model_name = graph_config["nli_model"]["model_name"]
        self.nli_threshold = graph_config["algorithm"]["threshold"]  
        
        print(f"⏳ 正在加载 NLI 模型 ({model_name})...")
        print(f"⚙️ 当前逻辑阈值已根据 YAML 设为: {self.nli_threshold}")
        self.nli_model = CrossEncoder(model_name)
        print("✅ 系统初始化完成！")

  
        print(f"⏳ 正在加载 NLI 模型 ({model_name})...")
        print(f"⚙️ 当前逻辑阈值已根据 YAML 设为: {self.nli_threshold}")
        self.nli_model = CrossEncoder(model_name)
        print("✅ 系统初始化完成！")

  
    def _generate_quick_summary(self, section_name: str, reranked_chunks: list) -> str:
        # 如果没有搜到任何相关的参考资料，直接返回空字符串
        if not reranked_chunks: return ""
        
        # 查看rerank选出来 chunks 是什么样的
        print(f"\n🔍 [调试信息] 为章节 '{section_name}' 选中的 Chunks:")
        for i, c in enumerate(reranked_chunks):
            print(f"   Chunk {i+1}: {c['text'][:100]}...")
            
        combined_text = "\n\n".join([c["text"] for c in reranked_chunks])
        prompt = f"Topic: {section_name}\nDocs:\n{combined_text}\nSummarize into one dense factual paragraph."
        return self._call_llm(prompt, "You are a technical summarizer.")
        # 第二个参数是 system prompt

    def _build_nli_graph(self, summaries: dict) -> dict:
        # 把所有的章节名称化作一个点
        sections = list(summaries.keys())
        summary_texts = list(summaries.values())
        dag = nx.DiGraph()
        dag.add_nodes_from(sections)


        for i, sec_a in enumerate(sections):
            for j, sec_b in enumerate(sections):
                if i == j: continue
                # 这里的 logits 是一组还未处理的数字，比如 [-1.2, 4.5, -0.5]
                # 每一个里面的数字都代表一个section的 summary_texts 和另一个 summary_texts 的包含程度
                logits = self.nli_model.predict([(summary_texts[i], summary_texts[j])])[0]
                # softmax 是一个数学函数，它把那些乱七八糟的原始分转化成总和为 1 的概率分布。
                # 原始分：[-1.2, 4.5, -0.5]
                # 经过 Softmax 后：[0.01, 0.92, 0.07]
                score = softmax(logits)[1]
                if score > self.nli_threshold:
                    dag.add_edge(sec_a, sec_b, weight=score)
                    #Weight 代表这段逻辑依赖关系的强度。

        # 这一段是为了处理逻辑死循坏
        while not nx.is_directed_acyclic_graph(dag):
            # 1. 找到一个圈
            cycle = nx.find_cycle(dag)
            # 2. 找证据最弱的一环
            min_edge = min(cycle, key=lambda e: dag.edges[e[0], e[1]]['weight'])
            # 3. 把这环拆掉
            dag.remove_edge(*min_edge)

        return {
            "order": list(nx.topological_sort(dag)),
            "map": {n: list(dag.predecessors(n)) for n in dag.nodes()}
        }

    def generate(self, input_data: dict) -> dict:
        island_name, sections_data = self._parse_input(input_data)
        
        # 阶段 1：Rerank + Summary
        print("\n🚀 阶段 1：正在进行 rerank 并提取摘要...")
        summaries = {}
        reranked_data_store = {}

        for section, data in sections_data.items():
            # 🌟 调用父类的 _rerank_chunks
            best_chunks = self._rerank_chunks(data.get("chunks", []), query=section)
            summaries[section] = self._generate_quick_summary(section, best_chunks)
            
        # 阶段 2：Graph
        print("\n🚀 阶段 2：正在分析 NLI 逻辑依赖...")
        plan = self._build_nli_graph(summaries)
        
        return {"status": "success", 
                "order": plan["order"], 
                "dependency": plan["map"], 
                "summaries": summaries}

# ==========================================
# 🚀 启动实验
# ==========================================
if __name__ == "__main__":
    DATA_PATH = "data/mock_data.json"
    
    if os.path.exists(DATA_PATH):
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            full_data = json.load(f)
        
        rag = GraphAwareRAG()
        final_result = rag.generate(full_data)
        
        print("\n" + "="*50)
        print("🏆 最终拓扑排序结果")
        print("="*50)
        for i, s in enumerate(final_result["order"]):
            print(f"Step {i+1}: {s}")
    else:
        print(f"❌ 找不到数据文件，请确保它在 {DATA_PATH}")
