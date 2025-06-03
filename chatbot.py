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

# 페이지 설정
st.set_page_config(
    page_title="LLaVA-Lucid-mini Chatbot",
    page_icon="🛰️",
    layout="wide"
)

# 모델 설정
@st.cache_resource
def load_model():
    """모델을 한 번만 로딩하는 함수"""
    args = {
        "model_path": "ohhan777/llava-Qwen2.5-7B-Instruct-s2-finetune-kompsat",
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

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

if "conv" not in st.session_state:
    st.session_state.conv = None

if "image_tensor" not in st.session_state:
    st.session_state.image_tensor = None


def display_chat_message(role, content):
    """채팅 메시지 표시"""
    with st.chat_message(role):
        st.write(content)

class StreamlitStreamer:
    def __init__(self, tokenizer, placeholder):
        self.tokenizer = tokenizer
        self.placeholder = placeholder
        self.buffer = b""
        self.text = ""

    def put(self, value):
        if len(value.shape) > 1:
            value = value[0]
        token_bytes = self.tokenizer.decode(value, skip_special_tokens=True).encode("utf-8")
        self.buffer += token_bytes
        try:
            decoded = self.buffer.decode("utf-8")
            self.text += decoded
            self.placeholder.write(self.text + "▌")
            self.buffer = b""
        except UnicodeDecodeError:
            pass

    def end(self):
        try:
            decoded = self.buffer.decode("utf-8")
            self.text += decoded
        except UnicodeDecodeError:
            pass
        self.placeholder.write(self.text)
        self.buffer = b""

def main():

    # 모델 로딩 (캐시됨)
    tokenizer, model, image_processor, args = load_model()

    if st.session_state.conv is None:
        st.session_state.conv = conv_templates[args["conv_mode"]].copy()

    st.title("🛰️ LLaVA-Lucid-mini Chatbot")
    st.markdown("이미지를 업로드하고 질문해보세요!")
    
    # 사이드바 - 이미지 업로드
    with st.sidebar:
        # 대화 초기화 버튼을 상단에 추가
        if st.button("🔄 채팅 새로 시작하기", key="reset_chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.uploaded_image = None
            st.session_state.uploader_key += 1  # key 변경
            if "image_first_use" in st.session_state:
                del st.session_state.image_first_use
            st.session_state.image_tensor = None
            st.session_state.conv = None
            st.rerun()
        
        st.divider()  # 구분선 추가
        
        st.header("🛰️ 이미지 업로드")
        
        uploaded_file = st.file_uploader(
            "이미지를 선택하세요",
            type=['png', 'jpg', 'jpeg', 'gif', 'bmp'],
            key=st.session_state.uploader_key,  # key를 동적으로 할당
            help="PNG, JPG, JPEG, GIF, BMP 형식을 지원합니다"
        )
        
        if uploaded_file is not None:
            # 이미지 로드 및 표시
            image = Image.open(uploaded_file)
            st.image(image, caption="업로드된 이미지", width=250, use_container_width=True)
            st.session_state.uploaded_image = image
            print("image uploaded")
            if "image_first_use" not in st.session_state:
                st.session_state.image_first_use = True
                # print("image_first_use is True")
            # 이미지 정보 표시
            st.write(f"**파일명:** {uploaded_file.name}")
            st.write(f"**크기:** {image.size}")
            st.write(f"**포맷:** {image.format}")
        
        # 이미지가 제거되었을 때 처리
        if uploaded_file is None and st.session_state.uploaded_image is not None:
            # 이미지가 제거되면 관련 상태 초기화
            st.session_state.uploaded_image = None
            st.rerun()
    
    # 메인 채팅 영역
    chat_container = st.container()
    
    with chat_container:
        # 이전 메시지들 표시
        for message in st.session_state.messages:
            display_chat_message(
                message["role"], 
                message["content"]
            )
    
    # 채팅 입력
    if input_text := st.chat_input("메시지를 입력하세요..."):
        # 사용자 메시지 추가
        user_message = {
            "role": "user",
            "content": input_text
        }
        st.session_state.messages.append(user_message)

         # 사용자 메시지 표시
        display_chat_message("user", input_text)

        conv = st.session_state.conv

        # if image is uploaded and first message, then add image to conv
        if st.session_state.uploaded_image is not None and st.session_state.image_first_use:
            # print("first image message")
            if model.config.mm_use_im_start_end:
                input_text = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + input_text
            else:
                input_text = DEFAULT_IMAGE_TOKEN + "\n" + input_text
            st.session_state.image_tensor = image_processor.preprocess(st.session_state.uploaded_image, return_tensors="pt")["pixel_values"].half().cuda()
            conv.append_message(conv.roles[0], input_text)
            st.session_state.image_first_use = False
        else:
            #print("later message")
            conv.append_message(conv.roles[0], input_text)

        conv.append_message(conv.roles[1], None)
            
            
        prompt = conv.get_prompt() + "<|im_start|>assistant\n"
        stop_str = "<|im_end|>"
        tokenizer.pad_token_id = 151662 
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        
       
        # AI 응답 생성 (여기에 실제 AI 모델 연동 코드 추가)
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            streamer = StreamlitStreamer(tokenizer, response_placeholder)
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=st.session_state.image_tensor,
                    do_sample=True,
                    temperature=args["temperature"],
                    max_new_tokens=args["max_new_tokens"],
                    streamer=streamer,
                    use_cache=True,
                    pad_token_id=tokenizer.pad_token_id,
                    stopping_criteria=[stopping_criteria]
                )
            streamer.end()
            response = streamer.text.strip()
            conv.messages[-1][-1] = response
            response_placeholder.write(response)
            # print(conv)
        
        # AI 응답 저장
        st.session_state.messages.append({
            "role": "assistant",
            "content": response
        })

if __name__ == "__main__":
    main()
