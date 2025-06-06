import streamlit as st
import io
from PIL import Image
import base64
from openai import OpenAI
import time
from pydantic import BaseModel, Field
from typing import List, Optional
import json

# 페이지 설정
st.set_page_config(
    page_title="산업현장 안전진단 AI 시스템",
    page_icon="🔍",
    layout="centered"  # 중앙 정렬 레이아웃
)

# OpenAI API 키 설정 (Streamlit secrets에서 가져오기)
try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
except KeyError:
    st.error("OpenAI API 키가 설정되지 않았습니다. Streamlit secrets에 OPENAI_API_KEY를 추가해주세요.")
    st.stop()

# OpenAI 클라이언트 초기화
client = OpenAI(api_key=OPENAI_API_KEY)


# Pydantic 모델 정의 - 출력 형식 강제
class RiskItem(BaseModel):
    description: str = Field(..., description="위험요소 설명")


class SafetyMeasure(BaseModel):
    description: str = Field(..., description="안전조치 설명")


class SafetyAnalysis(BaseModel):
    workplace_type: str = Field(..., description="산업현장 유형")
    workplace_details: List[str] = Field(..., description="산업현장 유형 식별 근거")
    workplace_subtype: str = Field(..., description="세부 분류")
    situation: str = Field(..., description="현장 상황 묘사")
    risks: List[RiskItem] = Field(..., description="안전 위험요소 목록")
    safety_measures: List[SafetyMeasure] = Field(..., description="안전조치 가이드라인")


# 세션 상태 초기화
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None

# 제목 및 설명
st.title("산업현장 안전진단 AI 시스템")
st.markdown("""
이 시스템은 업로드된 이미지를 분석하여 산업현장의 종류를 파악하고, 
해당 현장에 맞는 안전 매뉴얼을 기반으로 위험요소를 진단합니다.
""")


# 프롬프트 로드 함수
def load_prompt():
    try:
        with open("safety_prompt.txt", "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        st.error("safety_prompt.txt 파일을 찾을 수 없습니다.")
        return None


# 이미지 전처리 함수
def preprocess_image(img):
    """이미지 전처리 함수: RGBA->RGB 변환 및 크기 조정"""
    try:
        # RGBA 이미지를 RGB로 변환
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        # 이미지 크기가 너무 크면 리사이징
        max_size = (1200, 1200)
        if img.width > max_size[0] or img.height > max_size[1]:
            img.thumbnail(max_size, Image.LANCZOS)
        return img
    except Exception as e:
        st.error(f"이미지 처리 중 오류가 발생했습니다: {str(e)}")
        return None


# 이미지를 base64로 인코딩
def encode_image_to_base64(image):
    buffered = io.BytesIO()
    # RGBA 이미지를 RGB로 변환 (투명도 채널 제거)
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


# 통합된 안전 분석 함수
def analyze_safety(image, prompt):
    base64_image = encode_image_to_base64(image)

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "system",
                    "content": prompt + "\n\n중요: 모든 응답은 반드시 한국어로 작성하고, JSON 형식으로 반환해주세요."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "이 이미지의 산업현장을 분석하고 안전 위험요소를 진단해주세요. 결과는 JSON 형식으로 작성해주세요."},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            response_format={"type": "json_object"},
            max_tokens=2000
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"API 호출 중 오류가 발생했습니다: {str(e)}")
        return None


# 사이드바: 정보 표시
with st.sidebar:
    st.header("시스템 정보")
    st.markdown("---")
    st.markdown("## 안전진단 과정")
    st.markdown("1. 이미지 업로드")
    st.markdown("2. 산업현장 유형 식별 및 안전 위험요소 진단")

    st.markdown("---")
    st.markdown("### 지원되는 산업현장 유형")
    st.markdown("- 건설현장")
    st.markdown("- 제조공장")
    st.markdown("- 의료현장")
    st.markdown("- 물류센터")
    st.markdown("- 에너지 시설")
    st.markdown("- 화학 공장")
    st.markdown("- 광업 현장")
    st.markdown("- 농업 현장")

# 이미지 업로드 섹션
st.header("이미지 업로드")
uploaded_file = st.file_uploader("산업현장 이미지를 업로드하세요", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # 이미지 로드
        image = Image.open(uploaded_file)

        # 이미지 전처리
        image = preprocess_image(image)

        if image is not None:
            # 전처리된 이미지 표시
            st.image(image, caption="업로드된 이미지", use_container_width=True)
            st.session_state.uploaded_image = image

            # 분석 버튼 (중앙 배치)
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                analyze_button = st.button("분석 시작", key="analyze_button", use_container_width=True)

            if analyze_button:
                with st.spinner("이미지 분석 중..."):
                    # 프롬프트 로드
                    prompt = load_prompt()
                    if prompt is None:
                        st.stop()

                    # 통합 분석 실행
                    analysis_json = analyze_safety(image, prompt)
                    if analysis_json:
                        try:
                            # JSON 파싱
                            analysis_data = json.loads(analysis_json)
                            # Pydantic 모델을 사용하여 검증 (옵션)
                            analysis_result = SafetyAnalysis(**analysis_data)
                            st.session_state.analysis_result = analysis_result
                        except Exception as e:
                            st.error(f"결과 파싱 중 오류가 발생했습니다: {str(e)}")
                            st.error("원본 결과:")
                            st.json(analysis_json)
    except Exception as e:
        st.error(f"이미지 로드 중 오류가 발생했습니다: {str(e)}")
        st.error("유효한 이미지 파일을 업로드해 주세요. (JPG, PNG 형식)")

# 분석 결과 표시 섹션
if st.session_state.analysis_result:
    analysis = st.session_state.analysis_result

    st.markdown("---")
    st.header("분석 결과")

    # 1단계 결과 표시
    st.subheader("1단계: 산업현장 유형")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**산업현장 유형**: {analysis.workplace_type}")
        st.markdown(f"**세부 분류**: {analysis.workplace_subtype}")

    with col2:
        st.markdown("**식별 근거**:")
        for item in analysis.workplace_details:
            st.markdown(f"- {item}")

    # 구분선
    st.markdown("---")

    # 2단계 결과 표시
    st.subheader("2단계: 안전 위험요소 진단")

    st.markdown("### 현장 상황")
    st.markdown(analysis.situation)

    st.markdown("### 안전 위험요소 진단")
    for i, risk in enumerate(analysis.risks, 1):
        st.markdown(f"- {risk.description}")

    st.markdown("### 안전조치 가이드라인")
    for i, measure in enumerate(analysis.safety_measures, 1):
        st.markdown(f"- {measure.description}")

    # 결과 다운로드 버튼 (중앙 배치)
    try:
        # 마크다운 형식으로 변환
        combined_result = f"""# 산업현장 안전진단 결과 보고서

## 1단계: 산업현장 유형
**산업현장 유형**: {analysis.workplace_type}
**세부 분류**: {analysis.workplace_subtype}

**식별 근거**:
{chr(10).join([f"- {item}" for item in analysis.workplace_details])}

## 2단계: 안전 위험요소 진단

### 현장 상황
{analysis.situation}

### 안전 위험요소 진단
{chr(10).join([f"- {risk.description}" for risk in analysis.risks])}

### 안전조치 가이드라인
{chr(10).join([f"- {measure.description}" for measure in analysis.safety_measures])}

분석 일시: {time.strftime("%Y-%m-%d %H:%M:%S")}
"""

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.download_button(
                label="분석 결과 다운로드",
                data=combined_result,
                file_name="안전진단_결과.md",
                mime="text/markdown",
                use_container_width=True
            )
    except Exception as e:
        st.error(f"결과 다운로드 준비 중 오류가 발생했습니다: {str(e)}")

# 푸터
st.markdown("---")
st.markdown("© 2025 산업현장 안전진단 AI 시스템")
