"""
Production-shaped chat prompts for benchmarks.

The production chatflow sent: the system prompt below, with ~4 retrieved
knowledge-base chunks inserted at {context}, plus the user's question.
The real chunks came from copyrighted course notes, so the context here is
original filler text about the same topics; only its *length* matters for
serving performance (prefill cost and KV-cache size).

Length is controlled in characters and calibrated to tokens of the
ssc-FinLLM / InternLM2 tokenizer measured *after* the broker's 繁→簡 conversion
(see CHARS_PER_TOKEN). Production median prompt: ~727 tokens.
"""
import random

# The system prompt used in production (Dify chatflow 「金融知識小幫手」).
SYSTEM_PROMPT = """你是「金融知識小幫手」，請根據以下文件內容回答：
{context}。

請用繁體中文回答，語氣專業又親切，步驟化、條列化為主，回答要簡單易懂，並提供實例。優先根據「檢索到的文件節選」回答，若節選不足以支持答案，請直說「資料不足，無法確定」，並提出可以再提供的資訊。

原則：
1) 先用1~2句話回顧使用者需求，再給出清楚的解題步驟或建議。
2) 能算則算：涉及金額/比例/期數時，先寫出計算假設與步驟，再給數值。
3) 不進行投資勸誘，不保證報酬；不提供個股買賣指令。
4) 如果資訊不足或不確定，請明確說明限制，提出可查證或蒐集的資料清單。
5) 優先提供ETF、指數化、風險分散等保守中性方案；必要時給2~3種策略並列比較。
6) 最後用「快速檢核清單」收斂要點（3~5條），協助使用者下一步。
7) 僅使用繁體中文與台灣常用用語。

輸出格式：
- 先「重述需求」
- 接著「解題步驟 / 分析」
- 如有數字，列出「計算假設」與「計算過程」
- 最後給「快速檢核清單」與「可能風險 / 注意事項」"""

# Original filler text on the knowledge base's topics (not taken from the course notes).
CONTEXT_SENTENCES = [
    "貨幣具有時間價值，今天的一元在未來會因為利息而增值，因此不同時間點的現金流不能直接相加比較。",
    "終值是現在的一筆金額在固定利率下複利成長到未來某一時點的金額，計算方式為現值乘以一加利率的期數次方。",
    "現值是將未來的一筆金額依照折現率換算回今天的價值，折現率越高或期間越長，現值就越低。",
    "年金是指在固定期間內每期支付或收取相同金額的一系列現金流，例如每月定期定額儲蓄或貸款本息攤還。",
    "普通年金在每期期末發生現金流，期初年金則在每期期初發生，因此期初年金的終值與現值都比普通年金多累積一期利息。",
    "年金現值可以用來評估分期付款的總成本，也可以反推在既定預算下每期可以負擔的還款金額。",
    "有效年利率會考慮一年內的複利次數，名目年利率相同時，複利次數越多，有效年利率越高。",
    "分散投資是把資金配置在相關性較低的資產上，以降低單一資產價格波動對整體投資組合的影響。",
    "指數型基金追蹤特定市場指數的表現，管理費用通常較低，適合希望取得市場平均報酬的長期投資人。",
    "通貨膨脹會降低貨幣的購買力，因此在規劃長期儲蓄目標時，應以扣除通膨後的實質報酬率估算。",
    "緊急預備金通常建議準備三到六個月的生活支出，並存放在流動性高、風險低的帳戶中。",
    "信用卡循環利息屬於高成本負債，若同時有投資與高利負債，通常應優先償還高利負債。",
]

QUESTIONS = [
    "如果我每個月存五千元，年利率百分之二，十年後大約會有多少錢？",
    "期初年金和普通年金差在哪裡？可以舉例說明嗎？",
    "房貸三百萬、利率百分之二點一、二十年期，每月要繳多少？",
    "為什麼說今天的一百元比明年的一百元更有價值？",
    "新手想開始投資，應該先存緊急預備金還是先買ETF？",
    "名目利率和有效年利率有什麼不同？",
]

# Characters per model token for this text after 繁→簡 conversion, measured with the
# model's tokenizer.model (sentencepiece); see calibrate() below.
CHARS_PER_TOKEN = 1.52


def build_messages(prompt_tokens: int, rng: random.Random | None = None) -> list[dict]:
    """System prompt + filler context + one question, totalling ~prompt_tokens tokens."""
    rng = rng or random.Random()
    question = rng.choice(QUESTIONS)
    base = SYSTEM_PROMPT.replace("{context}", "") + question
    target_chars = int(prompt_tokens * CHARS_PER_TOKEN) - len(base)
    parts, n = [], 0
    start = rng.randrange(len(CONTEXT_SENTENCES))
    i = start
    while n < target_chars:
        s = CONTEXT_SENTENCES[i % len(CONTEXT_SENTENCES)]
        parts.append(s)
        n += len(s)
        i += 1
    context = "".join(parts)[:max(target_chars, 0)]
    return [
        {"role": "system", "content": SYSTEM_PROMPT.replace("{context}", context)},
        {"role": "user", "content": question},
    ]


def calibrate(tokenizer_model_path: str, prompt_tokens: int = 727) -> None:
    """Print chars/token and the token count build_messages() actually produces."""
    import sentencepiece as spm
    from opencc import OpenCC
    sp = spm.SentencePieceProcessor(model_file=tokenizer_model_path)
    t2s = OpenCC("t2s").convert
    text = t2s("".join(CONTEXT_SENTENCES) + SYSTEM_PROMPT + "".join(QUESTIONS))
    print(f"chars/token on filler+system text: {len(text) / len(sp.encode(text)):.3f}")
    rng = random.Random(0)
    counts = []
    for _ in range(20):
        msgs = build_messages(prompt_tokens, rng)
        counts.append(sum(len(sp.encode(t2s(m["content"]))) for m in msgs))
    print(f"build_messages({prompt_tokens}) -> {min(counts)}..{max(counts)} tokens "
          f"(content only; the chat template adds a few more)")


if __name__ == "__main__":
    import sys
    calibrate(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 727)
