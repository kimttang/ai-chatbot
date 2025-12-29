import csv
import os
import requests
import json
import numpy as np
import unicodedata
from flask import Flask, request, jsonify, render_template_string
from collections import defaultdict
from konlpy.tag import Okt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =========================================
# 1. 설정 및 초기화
# =========================================
app = Flask(__name__)

TRANSLATION_CSV = "sentences.csv"
KNOWLEDGE_CSV = "company_docs.csv"
MODEL_NAME = "gemma3:4b"
API_URL = "http://localhost:11434/api/chat"

okt = Okt()

# [뇌 1] 번역 통계 (Dice Score용)
co_occurrence = defaultdict(lambda: defaultdict(int))
k_total_count = defaultdict(int)
e_total_count = defaultdict(int)

# 🚫 학습용 노이즈 (통계 왜곡 방지)
LEARNING_NOISE = {
    # 관사, 전치사, be동사
    'a', 'an', 'the', 'this', 'that', 'it', 'there', 'here',
    'is', 'are', 'am', 'was', 'were', 'be', 'been',
    'to', 'in', 'on', 'at', 'of', 'for', 'with', 'by', 'from', 'up', 'out',

    # 조동사 (학습 제외)
    'will', 'can', 'must', 'should', 'have', 'has', 'had',
    'do', 'does', 'did', 'done',

    # 대명사 (점수 독식 방지)
    'i', 'you', 'he', 'she', 'we', 'they', 'my', 'your', 'his', 'her', 'our', 'their',

    # 부사 및 의미 없는 수식어
    'so', 'its', 'very', 'really', 'just', 'currently',

    # [중요] 통계를 망치는 회사 용어들 (노이즈 등록)
    'please', 'check', 'business', 'trip', 'scheduled', 'planning', 'share',
    'ran', 'running', 'due', 'mr', 'ms', 'homepage', 'enter', 'secret',
    'emails', 'company_staff_5g', 'let\'s', 'lets'
}

# 👑 VIP 고정석 (통계 무시하고 최우선 적용)
FIXED_MAPPING = {
    # 지시사 & 대명사
    '이것': 'this', '그것': 'that', '저것': 'that',
    '이': 'this', '그': 'the', '저': 'that', '이번': 'this',
    '여기': 'here', '저기': 'there',
    '나': 'i', '너': 'you', '우리': 'we', '제': 'my', '내': 'my',
    '무엇': 'what', '누구': 'who', '언제': 'when', '어디': 'where', '왜': 'why', '어떻게': 'how',

    # 동사 & 서술어
    '있다': 'have', '없다': 'not have',
    '이다': 'is', '다': 'is', '입니다': 'is', '이에요': 'is',
    '하다': 'do', '합니다': 'do', '해요': 'do',
    '않다': 'not',

    # 자주 틀리는 동사 고정
    '좋아하다': 'like', '먹다': 'eat', '가다': 'go', '보다': 'see',

    # 시간 관련
    '지금': 'now', '오늘': 'today', '내일': 'tomorrow', '어제': 'yesterday',
    '매일': 'every day', '매주': 'every week'
}

# [뇌 2] RAG 저장소
rag_documents = []
vectorizer = None
doc_vectors = None
chat_history = []

# =========================================
# 2. 한글 정규화 (자소 분리 방지)
# =========================================
def normalize_text(text):
    if not text: return ""
    # NFC: 자음+모음을 하나로 합침
    text = unicodedata.normalize('NFC', text)
    return " ".join(text.strip().split())

# =========================================
# 3. 데이터 로드 (모든 데이터 학습)
# =========================================
def load_all_data():
    global vectorizer, doc_vectors, rag_documents

    # --- 1. 번역 데이터 학습 ---
    print("⚙️ [1/2] 번역 데이터 로드 (승자 독식 모드)...")
    if os.path.exists(TRANSLATION_CSV):
        co_occurrence.clear()
        k_total_count.clear()
        e_total_count.clear()

        lines = []
        try:
            with open(TRANSLATION_CSV, 'r', encoding='utf-8-sig') as f: lines = list(csv.DictReader(f))
        except:
            with open(TRANSLATION_CSV, 'r', encoding='cp949') as f: lines = list(csv.DictReader(f))

        for row in lines:
            kr = normalize_text(row.get("korean") or row.get("text") or "")
            en = row.get("english") or row.get("intent") or ""

            kr_tokens = okt.morphs(kr, stem=True)
            en_tokens = en.lower().replace(".", "").replace("?", "").replace(",", "").split()

            # 중복 카운트 방지
            unique_k = set(k for k in kr_tokens if len(k) >= 1)
            unique_e = set(e for e in en_tokens if e not in LEARNING_NOISE)

            for k in unique_k:
                # 1글자는 제외하되, '팀', '일' 등은 허용
                if len(k) < 2 and k not in ['일', '집', '방', '문', '팀', '3', '2']: continue
                k_total_count[k] += 1
            for e in unique_e:
                e_total_count[e] += 1
            for k in unique_k:
                if len(k) < 2 and k not in ['일', '집', '방', '문', '팀', '3', '2']: continue
                for e in unique_e:
                    co_occurrence[k][e] += 1

    # --- 2. 회사 규정 로드 ---
    print("⚙️ [2/2] 회사 규정 로드...")
    rag_documents = []
    corpus = []

    if os.path.exists(KNOWLEDGE_CSV):
        lines = []
        try:
            with open(KNOWLEDGE_CSV, 'r', encoding='utf-8-sig') as f: lines = list(csv.DictReader(f))
        except:
            print("⚠️ CP949 모드로 전환하여 읽습니다.")
            with open(KNOWLEDGE_CSV, 'r', encoding='cp949') as f: lines = list(csv.DictReader(f))

        for row in lines:
            q = normalize_text(row.get("text") or row.get("korean") or "")
            a = normalize_text(row.get("intent") or row.get("english") or "")
            if q and a:
                rag_documents.append({"text": q, "intent": a})
                corpus.append(q + " " + a)

        if corpus:
            vectorizer = TfidfVectorizer(preprocessor=normalize_text, analyzer='char_wb', ngram_range=(2, 4))
            doc_vectors = vectorizer.fit_transform(corpus)

    print("✅ 모든 준비 완료!")

# =========================================
# 4. [핵심] 번역 로직
# =========================================
def perform_strict_translation(text):
    target_text = normalize_text(text.replace("번역", ""))

    # [추가] 뭉쳐서 쪼개지는 고유명사 강제 분리 (경영지원팀 -> 경영 지원 팀)
    target_text = target_text.replace("경영지원팀", "경영 지원 팀")

    morphs = okt.morphs(target_text, stem=True)
    print(f"\n🔍 [번역 분석] {morphs}")

    # [1단계] 철통 방어 (VIP는 면제)
    missing_words = []
    for k_word in morphs:
        # VIP는 통과
        if k_word in FIXED_MAPPING: continue

        # 조사/어미/불용어 필터링
        if k_word in ['은', '는', '이', '가', '을', '를', '에', '에서', '하다', '이다', '의', '으로', '와', '과', '고', '다', '요', '것', '수',
                      '하고', '하는', '된', '될', '할', '인', '져']:
            continue
        # 1글자 필터링
        if len(k_word) < 2 and k_word not in ['일', '집', '방', '문', '팀', '3', '2']: continue

        # 통계에 없으면 모르는 단어
        if k_word not in co_occurrence:
            missing_words.append(k_word)

    if missing_words:
        return f"🚫 다음 단어를 배운 적이 없습니다: {', '.join(missing_words)}"

    # [2단계] 승자 독식 매칭
    candidates = []
    for k_word in morphs:
        # VIP 1순위
        if k_word in FIXED_MAPPING:
            candidates.append((2.0, k_word, FIXED_MAPPING[k_word]))
            continue

        # 불용어 필터링
        if k_word in ['은', '는', '이', '가', '을', '를', '에', '에서', '하다', '이다', '의', '으로', '와', '과', '고', '다', '요', '것', '수',
                      '하고', '하는', '된', '될', '할', '인', '져']: continue
        if len(k_word) < 2 and k_word not in ['일', '집', '방', '문', '팀', '3', '2']: continue

        # 통계 매칭
        mappings = co_occurrence.get(k_word)
        if mappings:
            for e_word, joint_count in mappings.items():
                # Dice Score
                score = (2 * joint_count) / (k_total_count[k_word] + e_total_count[e_word] + 0.1)
                candidates.append((score, k_word, e_word))

    # 점수순 정렬
    candidates.sort(key=lambda x: x[0], reverse=True)

    final_keywords = []
    used_korean = set()
    used_english = set()

    for score, k_word, e_word in candidates:
        if k_word in used_korean: continue
        if e_word in used_english and not e_word.startswith("["): continue

        final_keywords.append(e_word)
        used_korean.add(k_word)
        used_english.add(e_word)
        print(f"   MATCH: {k_word} <-> {e_word} (점수: {score:.2f})")

    keyword_str = ", ".join(final_keywords)
    print(f"🤖 [최종 재료] {keyword_str}")

    # [3단계] Gemma 조립 (할루시네이션 방지)
    prompt = (
        f"Task: Construct a simple English sentence using keywords: [{keyword_str}]\n"
        f"Rules:\n"
        f"1. Use ALL keywords provided.\n"
        f"2. IMPORTANT: Do NOT add new ideas or follow-up sentences. Keep it short.\n"
        f"3. If keywords are only nouns (e.g., 'this', 'computer'), use a simple structure like 'This is a computer'.\n"
        f"4. If a keyword like 'have' doesn't fit naturally, omit it.\n"
        f"5. Output ONLY the English sentence."
    )

    # 번역은 기억력 끄고(use_history=False) 실행
    return call_ollama(prompt, system_role="grammar_corrector", use_history=False)

# =========================================
# 5. RAG 검색 로직
# =========================================
def perform_rag_chat(text):
    text = normalize_text(text)
    print(f"\n🔍 [RAG 검색] 사용자 질문: {text}")

    found = []

    # 1. 벡터 검색
    if vectorizer and doc_vectors is not None:
        try:
            query_vec = vectorizer.transform([text])
            sims = cosine_similarity(query_vec, doc_vectors)[0]
            ranked = np.argsort(-sims)[::-1]
            for i in ranked[:3]:
                if sims[i] > 0.15:
                    found.append(rag_documents[i])
        except: pass

    # 2. 키워드 검색 (점수제)
    if len(found) < 3:
        keywords = text.split()
        keyword_candidates = []
        for doc in rag_documents:
            match_count = 0
            for k in keywords:
                if len(k) > 1 and (k in doc['text'] or k in doc['intent']):
                    match_count += 1
            if match_count > 0:
                keyword_candidates.append((match_count, doc))

        keyword_candidates.sort(key=lambda x: x[0], reverse=True)
        for count, doc in keyword_candidates[:3]:
            if doc not in found:
                found.append(doc)

    if found:
        context = "\n".join([f"Q: {d['text']}\nA: {d['intent']}" for d in found[:3]])
    else:
        context = "관련된 내부 규정을 찾지 못했습니다."

    prompt = f"질문: {text}"
    system_msg = f"참고 문서:\n{context}\n\n문서 내용을 바탕으로 답변하세요. 내용이 없으면 '관련 규정을 찾을 수 없습니다'라고 답하세요."

    # RAG는 기억력 켜고(use_history=True) 실행
    return call_ollama(prompt, system_role="assistant", context_msg=system_msg, use_history=True)

# =========================================
# 6. Ollama 호출
# =========================================
def call_ollama(user_msg, system_role="assistant", context_msg="", use_history=False):
    messages = [{"role": "system", "content": context_msg if context_msg else "You are a helpful assistant."}]

    if use_history and chat_history:
        messages.extend(chat_history[-4:])

    messages.append({"role": "user", "content": user_msg})

    try:
        response = requests.post(API_URL, json={"model": MODEL_NAME, "messages": messages, "stream": False}, timeout=60)
        response.raise_for_status()
        answer = response.json()['message']['content'].strip().replace('"', '')

        if use_history:
            chat_history.append({"role": "user", "content": user_msg})
            chat_history.append({"role": "assistant", "content": answer})

        return answer
    except Exception as e:
        return f"오류: {str(e)}"

# =========================================
# 7. 웹 서버
# =========================================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <title>성장형 AI 통합 봇 (최종)</title>
    <style>
        body { font-family: 'Malgun Gothic', sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; background: #f4f4f9; }
        .container { background: white; padding: 20px; border-radius: 15px; box-shadow: 0 4px 10px rgba(0,0,0,0.1); }
        h2 { text-align: center; color: #333; }
        #chat-box { height: 500px; overflow-y: auto; border: 1px solid #ddd; padding: 20px; margin-bottom: 20px; background: #fff; border-radius: 8px; }
        .message { margin-bottom: 10px; padding: 10px 15px; border-radius: 20px; max-width: 80%; word-wrap: break-word; }
        .user { background: #007bff; color: white; margin-left: auto; text-align: right; border-bottom-right-radius: 2px; }
        .bot { background: #e9ecef; color: #333; margin-right: auto; text-align: left; border-bottom-left-radius: 2px; }
        .input-area { display: flex; gap: 10px; }
        input { flex: 1; padding: 15px; border: 1px solid #ddd; border-radius: 30px; outline: none; }
        button { padding: 15px 25px; background: #28a745; color: white; border: none; border-radius: 30px; cursor: pointer; font-weight: bold; }
        button:hover { background: #218838; }
        .tip { font-size: 12px; color: #666; text-align: center; margin-top: 10px; }
    </style>
</head>
<body>
    <div class="container">
        <h2>🤖 성장형 AI 비서 (Final)</h2>
        <div id="chat-box">
            <div class="message bot">안녕하세요! 규정 질문이나 번역 요청을 입력해주세요.</div>
        </div>
        <div class="input-area">
            <input type="text" id="user-input" placeholder="질문 입력 (끝에 '번역' 붙이면 번역 모드)" onkeypress="if(event.keyCode==13) sendMessage()">
            <button onclick="sendMessage()">전송</button>
        </div>
        <div class="tip">※ 예: "연차 규정 알려줘", "보고서 제출했어 번역"</div>
    </div>
    <script>
        async function sendMessage() {
            const input = document.getElementById('user-input');
            const chatBox = document.getElementById('chat-box');
            const text = input.value.trim();
            if (!text) return;
            
            chatBox.innerHTML += `<div class="message user">${text}</div>`;
            input.value = '';
            chatBox.scrollTop = chatBox.scrollHeight;
            
            try {
                const res = await fetch('/chat', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ message: text })
                });
                const data = await res.json();
                chatBox.innerHTML += `<div class="message bot">${data.answer}</div>`;
            } catch (e) {
                chatBox.innerHTML += `<div class="message bot" style="color:red">오류 발생</div>`;
            }
            chatBox.scrollTop = chatBox.scrollHeight;
        }
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_input = data.get('message', '').strip()
    if user_input.endswith("번역"):
        answer = perform_strict_translation(user_input)
    else:
        answer = perform_rag_chat(user_input)
    return jsonify({"answer": answer})

if __name__ == '__main__':
    load_all_data()
    print("🚀 서버 실행: http://localhost:5000")
    app.run(host='0.0.0.0', port=5000)