"""
RAG 问答系统效果评估
对比「有 RAG」vs「无 RAG」的问答准确率
"""

import json
import re
import time
from rag_engine import KnowledgeIndex, set_api_key, ask

# ── 配置 ──────────────────────────────────────────────
API_KEY = "sk-e0030bf1c8734118b7e4127dac4a1ce2"
set_api_key(API_KEY)

# ── 测试数据集 ─────────────────────────────────────────
# question: 测试问题
# expected_keywords: 回答中应包含的关键词（用于客观评估）
TEST_QUESTIONS = [
    {
        "id": 1,
        "question": "请简要说明本产品的技术参数。",
        "expected_keywords": ["电压", "电流", "功率", "参数"],
        "note": "检验系统能否正确召回产品参数类信息"
    },
    {
        "id": 2,
        "question": "产品的维护周期是多久？",
        "expected_keywords": ["维护", "周期", "检修", "保养", "年", "月"],
        "note": "检验系统能否正确召回维护类信息"
    },
    {
        "id": 3,
        "question": "请说明操作步骤的注意事项。",
        "expected_keywords": ["注意", "步骤", "操作", "规程", "规范"],
        "note": "检验系统能否正确召回注意事项类信息"
    },
    {
        "id": 4,
        "question": "技术要求的验收标准是什么？",
        "expected_keywords": ["验收", "标准", "合格", "要求", "指标"],
        "note": "检验系统能否正确召回验收标准类信息"
    },
    {
        "id": 5,
        "question": "请描述故障处理的基本流程。",
        "expected_keywords": ["故障", "处理", "排查", "检修", "步骤"],
        "note": "检验系统能否正确召回故障处理类信息"
    },
    {
        "id": 6,
        "question": "产品的生产成本控制措施有哪些？",
        "expected_keywords": ["成本", "控制", "节约", "效率", "优化"],
        "note": "检验系统能否正确召回成本控制类信息"
    },
    {
        "id": 7,
        "question": "操作过程中的安全规范有哪些？",
        "expected_keywords": ["安全", "防护", "警示", "规范", "禁止"],
        "note": "检验系统能否正确召回安全规范类信息"
    },
    {
        "id": 8,
        "question": "原材料的技术要求是什么？",
        "expected_keywords": ["材料", "标准", "规格", "性能", "合格"],
        "note": "检验系统能否正确召回材料要求类信息"
    },
    {
        "id": 9,
        "question": "请说明设备的安装与调试步骤。",
        "expected_keywords": ["安装", "调试", "步骤", "检查", "测试"],
        "note": "检验系统能否正确召回安装调试类信息"
    },
    {
        "id": 10,
        "question": "产品运行的环境条件有哪些要求？",
        "expected_keywords": ["温度", "湿度", "环境", "条件", "范围"],
        "note": "检验系统能否正确召回环境要求类信息"
    },
     {
        "id": 11,
        "question": "产品的主要性能指标有哪些？",
        "expected_keywords": ["性能", "指标", "精度", "效率", "可靠性"],
        "note": "检验系统能否正确召回性能指标类信息"
    },
    {
        "id": 12,
        "question": "产品的储存条件有什么要求？",
        "expected_keywords": ["储存", "存放", "温度", "防潮", "环境"],
        "note": "检验系统能否正确召回储存要求类信息"
    },
    {
        "id": 13,
        "question": "产品的设计使用寿命是多久？",
        "expected_keywords": ["寿命", "使用", "年限", "设计", "周期"],
        "note": "检验系统能否正确召回使用寿命类信息"
    },
    {
        "id": 14,
        "question": "请说明出厂前的检验项目。",
        "expected_keywords": ["检验", "检测", "出厂", "项目", "测试"],
        "note": "检验系统能否正确召回检验项目类信息"
    },
    {
        "id": 15,
        "question": "产品的包装和运输有什么要求？",
        "expected_keywords": ["包装", "运输", "标识", "防护", "要求"],
        "note": "检验系统能否正确召回包装运输类信息"
    },
    {
        "id": 16,
        "question": "请说明关键零部件的设计要求。",
        "expected_keywords": ["设计", "零部件", "参数", "配合", "公差"],
        "note": "检验系统能否正确召回设计要求类信息"
    },
    {
        "id": 17,
        "question": "操作人员需要具备哪些资质？",
        "expected_keywords": ["资质", "培训", "上岗", "操作证", "资格"],
        "note": "检验系统能否正确召回人员资质类信息"
    },
    {
        "id": 18,
        "question": "请描述备品备件的管理规范。",
        "expected_keywords": ["备件", "库存", "管理", "采购", "记录"],
        "note": "检验系统能否正确召回备件管理类信息"
    },
    {
        "id": 19,
        "question": "产品运行中的异常情况如何处理？",
        "expected_keywords": ["异常", "报警", "处理", "排查", "停机"],
        "note": "检验系统能否正确召回异常处理类信息"
    },
    {
        "id": 20,
        "question": "请说明质量不合格品的处理流程。",
        "expected_keywords": ["不合格", "质量", "返工", "报废", "处理"],
        "note": "检验系统能否正确召回质量处理类信息"
    },
]

# ── 评估指标 ──────────────────────────────────────────

def keyword_hit_rate(answer: str, keywords: list[str]) -> float:
    """关键词命中率：回答中命中了多少关键词"""
    if not keywords or not answer:
        return 0.0
    answer_lower = answer.lower()
    hits = sum(1 for kw in keywords if kw.lower() in answer_lower)
    return hits / len(keywords)


def answer_length_score(answer: str) -> float:
    """答案长度得分：过短说明没召回内容，过长可能是幻觉"""
    length = len(answer)
    if length < 20:
        return 0.0
    elif length < 100:
        return 0.5
    elif length < 500:
        return 1.0
    else:
        return 0.8  # 过长适当扣分


def hallucination_score(answer: str) -> float:
    """幻觉检测：检查回答中是否有不确定表述"""
    uncertain_phrases = [
        "不确定", "不知道", "没有找到", "未找到",
        "无法确定", "无法回答", "我无法", "我不清楚",
        "知识库中未找到", "未在文档中", "没有相关"
    ]
    for phrase in uncertain_phrases:
        if phrase in answer:
            return 1.0  # 明确说了找不到，完全正常
    # 没找到不确定表述，给一个基础分
    return 0.9


def rag_has_context_score(chunks: list) -> float:
    """RAG 模式下，检查是否成功召回上下文"""
    if not chunks:
        return 0.0
    return 1.0


def composite_score(answer: str, keywords: list[str], chunks: list = None) -> dict:
    """综合得分"""
    hit = keyword_hit_rate(answer, keywords)
    length = answer_length_score(answer)
    hallucination = hallucination_score(answer)

    scores = {
        "keyword_hit": hit,
        "length_score": length,
        "hallucination_score": hallucination,
    }

    if chunks is not None:
        scores["has_context"] = rag_has_context_score(chunks)

    # 综合得分 = 关键词命中 * 0.5 + 长度得分 * 0.2 + 幻觉得分 * 0.3
    composite = hit * 0.5 + length * 0.2 + hallucination * 0.3
    scores["composite"] = round(composite, 4)

    return scores


# ── 主评估流程 ─────────────────────────────────────────

def run_evaluation():
    print("=" * 60)
    print("  RAG 问答系统效果评估")
    print("  有 RAG vs 无 RAG 准确率对比")
    print("=" * 60)

    # 检查索引是否就绪
    ki = KnowledgeIndex()
    loaded = ki.load()
    if not loaded:
        print("\n[错误] 索引未构建！请先在网页上传 PDF 并构建索引。")
        print("（当前使用随机初始化索引，仅供演示）")
        # 创建模拟索引用于演示
        print("[提示] 以下为演示模式，结果仅供参考\n")
        use_demo = True
    else:
        print(f"\n[OK] 索引加载成功，共 {ki.stats()['chunk_count']} 个文本块")
        use_demo = False

    results = {
        "with_rag": [],
        "without_rag": [],
    }

    print("\n" + "-" * 60)
    print("开始评估（共 {} 道题）...".format(len(TEST_QUESTIONS)))
    print("-" * 60)

    for item in TEST_QUESTIONS:
        qid = item["id"]
        question = item["question"]
        keywords = item["expected_keywords"]

        print(f"\n【题目 {qid}】{question}")
        print(f"  期望关键词: {', '.join(keywords)}")

        # ── 有 RAG ──
        print("  [有 RAG] 检索中...", end=" ")
        try:
            if not use_demo:
                chunks = ki.search(question, top_k=5)
                answer_rag = ask(question, chunks)
            else:
                chunks = []
                answer_rag = "（演示模式：索引未构建，无法生成有 RAG 的回答）"
            scores_rag = composite_score(answer_rag, keywords, chunks)
            print("OK")
        except Exception as e:
            answer_rag = f"[错误] {e}"
            scores_rag = composite_score(answer_rag, keywords, [])
            print("FAIL")

        # ── 无 RAG ──
        print("  [无 RAG] 直接提问 LLM...", end=" ")
        try:
            chunks_empty = []
            answer_no_rag = ask(question, chunks_empty)
            scores_no_rag = composite_score(answer_no_rag, keywords)
            print("OK")
        except Exception as e:
            answer_no_rag = f"[错误] {e}"
            scores_no_rag = composite_score(answer_no_rag, keywords)
            print("FAIL")

        # ── 打印结果 ──
        print(f"  有 RAG 得分: {scores_rag['composite']} | 无 RAG 得分: {scores_no_rag['composite']}")

        results["with_rag"].append({**item, "answer": answer_rag, "scores": scores_rag})
        results["without_rag"].append({**item, "answer": answer_no_rag, "scores": scores_no_rag})

        # 控制请求频率
        time.sleep(1)

    # ── 统计汇总 ──
    print("\n" + "=" * 60)
    print("  评估结果汇总")
    print("=" * 60)

    def avg_score(items):
        if not items:
            return 0
        return sum(i["scores"]["composite"] for i in items) / len(items)

    avg_rag = avg_score(results["with_rag"])
    avg_no_rag = avg_score(results["without_rag"])
    improvement = avg_rag - avg_no_rag
    improvement_pct = (improvement / avg_no_rag * 100) if avg_no_rag > 0 else 0

    print(f"\n  {'模式':<10} {'平均得分':<12} {'关键词命中率':<15}")
    print(f"  {'-' * 40}")
    print(f"  {'有 RAG':<10} {avg_rag:<12.4f} {sum(i['scores']['keyword_hit'] for i in results['with_rag'])/len(results['with_rag']):.2%}")
    print(f"  {'无 RAG':<10} {avg_no_rag:<12.4f} {sum(i['scores']['keyword_hit'] for i in results['without_rag'])/len(results['without_rag']):.2%}")
    print(f"  {'-' * 40}")
    print(f"\n  [Chart] 准确率提升: {improvement:+.4f}")
    print(f"  [Chart] 提升幅度:   {improvement_pct:+.2f}%")

    if improvement > 0:
        print(f"\n  [RAG] 机制有效提升了问答质量！")
    elif improvement < 0:
        print(f"\n  [!] 当前测试集下无 RAG 表现更好，建议扩充测试题库")
    else:
        print(f"\n  [-] 两种模式表现相当")

    # ── 保存详细报告 ──
    report_path = "evaluation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "summary": {
                "avg_score_rag": round(avg_rag, 4),
                "avg_score_no_rag": round(avg_no_rag, 4),
                "improvement": round(improvement, 4),
                "improvement_pct": f"{improvement_pct:.2f}%",
                "total_questions": len(TEST_QUESTIONS),
            },
            "results": results,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n  [Report] 详细报告已保存至: {report_path}")

    return results


if __name__ == "__main__":
    run_evaluation()
