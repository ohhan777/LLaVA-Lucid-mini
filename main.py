import streamlit as st
from PIL import Image
import base64
import io

import torch
from llava.model.builder import load_pretrained_model
from llava.conversation import conv_templates, SeparatorStyle
from llava.utils import disable_torch_init
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token, KeywordsStoppingCriteria
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from transformers import TextStreamer
import re


# 페이지 설정
st.set_page_config(
    page_title="KOMPSAT Image Chatbot",
    page_icon="🤖",
    layout="wide"
)

# 모델 설정
@st.cache_resource
def load_model():
    """모델을 한 번만 로딩하는 함수"""
    args = {
        "model_path": "./checkpoints/llava-Qwen2.5-7B-Instruct-s2-finetune-kompsat",
        "conv_mode": "qwen_2_5",
        "temperature": 0.6,
        "max_new_tokens": 1024,
    }
    
    model_name = get_model_name_from_path(args["model_path"])
    
    disable_torch_init()
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args["model_path"],
        model_base=None,
        model_name=model_name
    )
    
    return tokenizer, model, image_processor, args

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None

if "conv" not in st.session_state:
    st.session_state.conv = None

def encode_image_to_base64(image):
    """이미지를 base64로 인코딩"""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

def display_chat_message(role, content):
    """채팅 메시지 표시"""
    with st.chat_message(role):
        st.write(content)

def clean_output(text):
    """출력에서 특수 토큰 제거"""
    # <|im_end|>, <|im_start|> 등의 특수 토큰 제거
    text = re.sub(r'<\|im_end\|>', '', text)
    text = re.sub(r'<\|im_start\|>', '', text)
    text = re.sub(r'<\|.*?\|>', '', text)  # 기타 특수 토큰들
    return text.strip()

class StreamlitStreamer:
    """Streamlit용 커스텀 스트리머"""
    def __init__(self, tokenizer, placeholder):
        self.tokenizer = tokenizer
        self.placeholder = placeholder
        self.text = ""
        self.buffer = ""
        
    def put(self, value):
        if len(value.shape) > 1:
            value = value[0]
        
        # 토큰을 텍스트로 디코드
        token_text = self.tokenizer.decode(value, skip_special_tokens=True)
        
        # 특수 토큰이 포함된 경우 스트리밍 중단
        if '<|im_end|>' in token_text:
            return
            
        self.text += token_text
        
        # 버퍼에 추가
        self.buffer += token_text
        
        # 완전한 문자 단위로 분리
        try:
            # 현재까지의 텍스트를 디코딩
            decoded_text = self.text.encode('utf-8').decode('utf-8')
            # 특수 토큰 제거
            clean_text = clean_output(decoded_text)
            # 프롬프트 부분 제거
            if "<|im_start|>assistant\n" in clean_text:
                clean_text = clean_text.split("<|im_start|>assistant\n")[-1]
            # 화면에 표시
            self.placeholder.write(clean_text)
        except UnicodeDecodeError:
            # 디코딩이 실패하면 버퍼링만 하고 표시하지 않음
            pass
        
    def end(self):
        # 마지막에 버퍼의 내용을 한 번에 표시
        try:
            decoded_text = self.text.encode('utf-8').decode('utf-8')
            clean_text = clean_output(decoded_text)
            if "<|im_start|>assistant\n" in clean_text:
                clean_text = clean_text.split("<|im_start|>assistant\n")[-1]
            self.placeholder.write(clean_text)
        except UnicodeDecodeError:
            # 디코딩 실패 시 원본 텍스트 표시
            self.placeholder.write(self.text)

def main():
    # 모델 로딩 (캐시됨)
    tokenizer, model, image_processor, args = load_model()
    
    # conversation 초기화
    if st.session_state.conv is None:
        st.session_state.conv = conv_templates[args["conv_mode"]].copy()

    st.title("🤖 LLaVA-Lucid-mini Chatbot")
    st.markdown("이미지를 업로드하고 질문해보세요!")
    
    # 사이드바 - 이미지 업로드
    with st.sidebar:
        # 대화 초기화 버튼을 상단에 추가
        if st.button("🔄 대화 초기화", key="reset_chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.uploaded_image = None
            if "image_first_use" in st.session_state:
                del st.session_state.image_first_use
            st.session_state.conv = conv_templates[args["conv_mode"]].copy()
            st.rerun()
        
        st.divider()  # 구분선 추가
        
        st.header("📷 이미지 업로드")
        
        uploaded_file = st.file_uploader(
            "이미지를 선택하세요",
            type=['png', 'jpg', 'jpeg', 'gif', 'bmp'],
            help="PNG, JPG, JPEG, GIF, BMP 형식을 지원합니다"
        )
        
        if uploaded_file is not None:
            # 이미지 로드 및 표시
            image = Image.open(uploaded_file)
            st.image(image, caption="업로드된 이미지", width=250, use_container_width=True)
            st.session_state.uploaded_image = image
            
            if "image_first_use" not in st.session_state:
                st.session_state.image_first_use = True
            
                
                
            # 이미지 정보 표시
            st.write(f"**파일명:** {uploaded_file.name}")
            st.write(f"**크기:** {image.size}")
            st.write(f"**포맷:** {image.format}")
        
        # 이미지가 제거되었을 때 처리
        if uploaded_file is None and st.session_state.uploaded_image is not None:
            # 이미지가 제거되면 관련 상태 초기화
            st.session_state.uploaded_image = None
            if "image_first_use" in st.session_state:
                del st.session_state.image_first_use
            st.rerun()
    
    # 메인 채팅 영역
    chat_container = st.container()
    
    with chat_container:
        # 이전 메시지들 표시
        for message in st.session_state.messages:
            display_chat_message(
                message["role"], 
                message["content"], 
            )
    
    # 채팅 입력
    if input_text := st.chat_input("메시지를 입력하세요..."):
        # 사용자 메시지 추가
        user_message = {
            "role": "user",
            "content": input_text,
            "image": st.session_state.uploaded_image
        }
        st.session_state.messages.append(user_message)
        
        # 사용자 메시지 표시
        display_chat_message("user", input_text)
        
        # AI 응답 생성 
        conv = st.session_state.conv
        
        if st.session_state.image_first_use:
            # first message
            if model.config.mm_use_im_start_end:
                input_text = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + input_text
            else:
                input_text = DEFAULT_IMAGE_TOKEN + "\n" + input_text
            conv.append_message(conv.roles[0], input_text)
            
            # 이미지 텐서 처리 - 안전하게 처리
            try:
                if st.session_state.uploaded_image is not None:
                    # 이미지 전처리
                    image_tensor = image_processor.preprocess(
                        st.session_state.uploaded_image, 
                        return_tensors="pt"
                    )["pixel_values"]
                    
                    # half precision으로 변환
                    image_tensor = image_tensor.half()
                    
                    # CUDA로 이동
                    image_tensor = image_tensor.cuda()
                else:
                    image_tensor = None
            except Exception as e:
                st.error(f"이미지 처리 중 오류가 발생했습니다: {str(e)}")
                image_tensor = None
                st.session_state.image_first_use = False
                return
            
            st.session_state.image_first_use = False
        else:
            # later messages
            conv.append_message(conv.roles[0], input_text)
            image_tensor = None
        
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt() + "<|im_start|>assistant\n"
        stop_str = "<|im_end|>"
        tokenizer.pad_token_id = 151662 
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        with st.chat_message("assistant"):
            # 스트리밍 응답을 위한 플레이스홀더
            response_placeholder = st.empty()
            
            # 커스텀 스트리머 생성
            streamer = StreamlitStreamer(tokenizer, response_placeholder)
            
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids, 
                    images=image_tensor, 
                    do_sample=True, 
                    temperature=args["temperature"], 
                    max_new_tokens=args["max_new_tokens"],
                    streamer=streamer,
                    use_cache=True, 
                    pad_token_id=tokenizer.pad_token_id, 
                    stopping_criteria=[stopping_criteria]
                )

            # 최종 출력 생성 및 정리
            outputs = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            
            # 프롬프트 부분 제거 (assistant 이후 부분만 추출)
            if "<|im_start|>assistant\n" in outputs:
                outputs = outputs.split("<|im_start|>assistant\n")[-1]
            
            # 특수 토큰 제거
            outputs = clean_output(outputs)
            
            # 최종 결과 표시
            response_placeholder.write(outputs)
        
        # AI 응답 저장
        st.session_state.messages.append({
            "role": "assistant",
            "content": outputs
        })

        # conversation에도 응답 저장
        conv.messages[-1][-1] = outputs
        


if __name__ == "__main__":
    main()
