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
    page_title="Image Chatbot",
    page_icon="🤖",
    layout="wide"
)

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None

def encode_image_to_base64(image):
    """이미지를 base64로 인코딩"""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

def display_chat_message(role, content, image=None):
    """채팅 메시지 표시"""
    with st.chat_message(role):
        if image:
            st.image(image, width=300)
        st.write(content)

def main():

    args = {
        "model_path": "./checkpoints/llava-Qwen2.5-7B-Instruct-s2-finetune-kompsat",
        "conv_mode": "qwen_2_5",
        "temperature": 0.6,
        "max_new_tokens": 1024,
    }

    model_name = get_model_name_from_path(args["model_path"])
    conv_mode = args["conv_mode"]
    temperature = args["temperature"]
    max_new_tokens = args["max_new_tokens"]

    disable_torch_init()
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args["model_path"],
        model_base=None,
        model_name=model_name)

    conv = conv_templates[conv_mode].copy()
    roles = conv.roles

    st.title("🤖 LLaVA-Lucid-mini Chatbot")
    st.markdown("이미지를 업로드하고 질문해보세요!")
    
    # 사이드바 - 이미지 업로드
    with st.sidebar:
        # 대화 초기화 버튼을 상단에 추가
        if st.button("🔄 대화 초기화", key="reset_chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.uploaded_image = None
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
            st.image(image, caption="업로드된 이미지", width=250)
            st.session_state.uploaded_image = image
            
            # 이미지 정보 표시
            st.write(f"**파일명:** {uploaded_file.name}")
            st.write(f"**크기:** {image.size}")
            st.write(f"**포맷:** {image.format}")
        
        # 이미지 제거 버튼
        if st.session_state.uploaded_image is not None:
            if st.button("🗑️ 이미지 제거"):
                st.session_state.uploaded_image = None
                st.rerun()
    
    # 메인 채팅 영역
    chat_container = st.container()
    
    with chat_container:
        # 이전 메시지들 표시
        for message in st.session_state.messages:
            display_chat_message(
                message["role"], 
                message["content"], 
                message.get("image")
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
        display_chat_message("user", input_text, st.session_state.uploaded_image)
        
        # AI 응답 생성 
        if st.session_state.uploaded_image is not None:
            # first message
            if model.config.mm_use_im_start_end:
                input_text = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + input_text
            else:
                input_text = DEFAULT_IMAGE_TOKEN + "\n" + input_text
            conv.append_message(conv.roles[0], input_text)
            # 이미지 텐서 처리
            image_tensor = image_processor.preprocess(st.session_state.uploaded_image, return_tensors="pt")["pixel_values"].half().cuda()
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
            with st.spinner("응답 생성 중..."):
                # 실제 AI 모델 호출 부분
                with torch.inference_mode():
                    output_ids = model.generate(
                        input_ids, 
                        images=image_tensor, 
                        do_sample=True, 
                        temperature=args["temperature"], 
                        max_new_tokens=args["max_new_tokens"], 
                        use_cache=True, 
                        pad_token_id=tokenizer.pad_token_id, 
                        stopping_criteria=[stopping_criteria]
                    )

                outputs = tokenizer.decode(output_ids[0]).strip()            
                st.write(outputs)
        
        # AI 응답 저장
        st.session_state.messages.append({
            "role": "assistant",
            "content": outputs
        })

        conv.messages[-1][-1] = outputs
        
        # 업로드된 이미지 초기화 (한 번 사용 후)
        if st.session_state.uploaded_image:
            st.session_state.uploaded_image = None

if __name__ == "__main__":
    main()
