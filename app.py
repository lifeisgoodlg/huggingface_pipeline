import pandas as pd
import streamlit as st
import plotly.express as px
from transformers import pipeline
from youtube_api import save_comment

CANDIDATE_LABELS = ["praise", "criticism", "information", "question", "advertisement", "emotion"]
TEST_COMMENTS = [
    "올드보이 ost 고독함이 너무 밀려옴 ㅋㅋ",
    "짜장면 하나가지고가 아니라 배달앱의 문제가 제일 크다고 봅니다.",
    "진짜 잃을거 없는 인생인가보다 ㅋㅋㅋ",
    "대한민국 개법 좀 고쳐야 한다",
    "다른 인터뷰들과 차별되는 좋은 질문들이었어요.",
    "장 항시 말조심 ㅋㅋㅋㅋㅋㅋㅋ",
    "역사는 반복된다라는 교훈을 주는게 소름이었음",
    "이야 진격거를 시즌별로 나눠서 올리더니 통합본으로 또 올리네!! 대단하다 상업유튜바",
    "여기서 나오는 노래 제목 뭐임",
    "박재범 2pm이였어요??",
    "오마이걸 살짝 썩었어 이거 방송에 내보낸것도 좀 대단한거같다….",
    "제 채널도 방문해주세요 구독 부탁드립니다",
    "저도 비슷한 영상 올리고 있으니 관심있으시면 와주세요",
    "【💗캐릭캐릭체인지 2기 두근두근💗】👉매주 수요일 오후 5시 업로드됩니다! #스밍스 #캐릭캐릭체인지2 #전편공개 #무료보기",
    "요루 귀여워",
    " 선생님의 고충이란.. 이런게 참 힘들지ㅠㅠ...",
    "썸네일 마음에 드는군"
]

@st.cache_resource
def load_models():
    sentiment = pipeline(
        "sentiment-analysis",
        # model="snunlp/KR-FinBert-SC"
        model="sangrimlee/bert-base-multilingual-cased-nsmc"
    )
    zero_shot = pipeline(
        "zero-shot-classification",
        model="MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
    )
    return sentiment, zero_shot

sentiment_pipeline, zero_shot_pipeline = load_models()

def analyze_comments(comments):
    results = []
    progress = st.progress(0, text="댓글 분석 중...")
    for i, comment in enumerate(comments):
        topic = zero_shot_pipeline(comment, CANDIDATE_LABELS)
        top_label = topic["labels"][0]

        if top_label in ["question", "advertisement"]:
            sent_label = "neutral"
        else:
            sent = sentiment_pipeline(comment)[0]
            sent_label = sent["label"]

        results.append({
            "댓글": comment,
            "감성": sent_label,
            "토픽": top_label
        })
        progress.progress((i+1)/len(comments), text=f"댓글 분석 중... ({i+1}/{len(comments)})")

    progress.empty()
    return pd.DataFrame(results)


st.title("📊 유튜브 댓글 여론 분석기 📊")
st.caption("유튜브 영상 URL을 입력하면 댓글을 분석해드립니다!")

url = st.text_input("URL 입력", placeholder="https://www.youtube.com/watch?v=...")
analyze_btn = st.button("분석하기")

if analyze_btn:
    if url:
        with st.spinner("댓글 수집 중..."):
            comments_text = save_comment(url)
        st.success(f"{len(comments_text)}개 댓글 수집 완료!")
    else:
        st.info("URL 없음 - 테스트 데이터로 실행합니다")
        comments_text = TEST_COMMENTS

    df = analyze_comments(comments_text)

    total = len(df)
    st.metric("수집 댓글", f"{total}개")

    st.divider()
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.subheader("감성 분포")
        sentiment_counts = df["감성"].value_counts()
        fig_pie = px.pie(
            values=sentiment_counts.values,
            names=sentiment_counts.index,
            color=sentiment_counts.index,
            color_discrete_map={"positive": "#008000", "negative": "#ff0000", "neutral": "#808080"},
            hole=0.5
        )
        fig_pie.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=250)
        st.plotly_chart(fig_pie, width='content')

    with chart_col2:
        st.subheader("토픽 분포")
        topic_counts = df["토픽"].value_counts().reset_index()
        topic_counts.columns = ["토픽", "개수"]
        fig_bar = px.bar(
            topic_counts,
            x="개수",
            y="토픽",
            orientation="h", 
            color="토픽",
            color_discrete_sequence=["#FF6B6B", "#FFA500", "#FFD93D", "#6BCB77", "#4D96FF", "#845EC2"])
        fig_bar.update_layout(
            margin=dict(t=0, b=0, l=0, r=0),
            height=250,
            showlegend=False,
            yaxis=dict(categoryorder="total ascending")
        )
        st.plotly_chart(fig_bar, width='content')

    st.divider()
    st.subheader("댓글 목록")
    st.dataframe(df, width='content')