import streamlit as st
from openai import OpenAI
import json
import os
from dotenv import load_dotenv
import logging
from datetime import datetime
import pandas as pd
import traceback
import sys
from pydantic import BaseModel
from typing import List, Optional, Dict

# 🔧 Models Definition
class ProductRecommendation(BaseModel):
    category: str
    reason: str
    confidence: float

class RecommendationResponse(BaseModel):
    recommendations: List[ProductRecommendation]

class ProductDisplay(BaseModel):
    category: str
    product: str
    price: float
    reason: str
    confidence: float
    image: str

# 📦 Product Catalog
PRODUCT_CATALOG = {
    "Sports & Fitness": [
        {"id": "S001", "name": "Nike Air Zoom Pegasus", "price": 4500, "description": "รองเท้าวิ่งระดับพรีเมียม", "image": "https://picsum.photos/300/200?random=1"},
        {"id": "S002", "name": "Fitbit Charge 5", "price": 6900, "description": "สมาร์ทวอทช์สำหรับออกกำลังกาย", "image": "https://picsum.photos/300/200?random=2"},
        {"id": "S003", "name": "Yoga Mat Premium", "price": 1200, "description": "เสื่อโยคะคุณภาพสูง", "image": "https://picsum.photos/300/200?random=3"}
    ],
    "Photography": [
        {"id": "P001", "name": "Sony A7 III", "price": 59900, "description": "กล้อง Mirrorless ระดับมืออาชีพ", "image": "https://picsum.photos/300/200?random=4"},
        {"id": "P002", "name": "DJI Mini 3 Pro", "price": 29900, "description": "โดรนถ่ายภาพขนาดพกพา", "image": "https://picsum.photos/300/200?random=5"},
        {"id": "P003", "name": "Peak Design Backpack", "price": 8900, "description": "กระเป๋ากล้องระดับพรีเมียม", "image": "https://picsum.photos/300/200?random=6"}
    ],
    "Cooking": [
        {"id": "C001", "name": "Instant Pot Duo", "price": 3900, "description": "หม้อทำอาหารอเนกประสงค์", "image": "https://picsum.photos/300/200?random=7"},
        {"id": "C002", "name": "Vitamix Blender", "price": 15900, "description": "เครื่องปั่นระดับมืออาชีพ", "image": "https://picsum.photos/300/200?random=8"},
        {"id": "C003", "name": "Kitchen Scale Digital", "price": 890, "description": "เครื่องชั่งดิจิตอลแม่นยำสูง", "image": "https://picsum.photos/300/200?random=9"}
    ]
}

# 🔧 Setup Logging with Detail
def setup_logging():
    """ตั้งค่าระบบ logging แบบละเอียด"""
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    log_filename = f"logs/recommender_{datetime.now().strftime('%Y%m%d')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("="*50)
    logger.info("🚀 Starting AI Product Recommender")
    logger.info(f"📝 Log file: {log_filename}")
    logger.info("="*50)
    
    return logger

# 🔐 API Key Validation
def validate_api_key(api_key: str) -> bool:
    """ตรวจสอบรูปแบบของ API key"""
    logger.info("🔍 Validating API key format")
    if not api_key.startswith('sk-'):
        logger.warning("❌ API key validation failed: Doesn't start with 'sk-'")
        return False
    if len(api_key) < 20:
        logger.warning("❌ API key validation failed: Length too short")
        return False
    logger.info("✅ API key format validation passed")
    return True

# 🔐 OpenAI Client
def init_openai_client():
    """สร้างและทดสอบการเชื่อมต่อกับ OpenAI API"""
    try:
        api_key = st.session_state.get('openai_api_key', '')
        if not api_key:
            logger.warning("❌ No API key found in session state")
            return None
            
        logger.info("🔄 Initializing OpenAI client")
        client = OpenAI(api_key=api_key)
        
        # ทดสอบการเชื่อมต่อ
        logger.info("🔄 Testing API connection")
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Test connection"}],
            max_tokens=5
        )
        
        if response:
            logger.info("✅ API connection test successful")
            return client
            
    except Exception as e:
        error_details = traceback.format_exc()
        logger.error(f"❌ Error initializing OpenAI client:\n{error_details}")
        st.sidebar.error(f"เกิดข้อผิดพลาด: {str(e)}")
        return None

def get_recommendations_with_products(interest: str, client) -> List[ProductDisplay]:
    """ขอคำแนะนำสินค้าโดยใช้ Structured Output"""
    try:
        logger.info(f"🔄 Getting recommendations for interest: {interest}")
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": """You are an AI assistant functioning as a recommendation system for an ecommerce website. Be specific and limit your answers to the requested format. Keep your answers short and concise.
                    Your Task: วิเคราะห์ความสนใจของลูกค้าและแนะนำสินค้าที่เหมาะสม 
                    หมวดหมู่ที่มี: Sports & Fitness, Photography, Cooking
                    
                    ตอบในรูปแบบ JSON ดังนี้:
                    {
                        "recommendations": [
                            {
                                "category": "ชื่อหมวดหมู่ตามที่กำหนด",
                                "reason": "เหตุผลที่แนะนำ",
                                "confidence": 0.8
                            }
                        ]
                    }"""
                },
                {
                    "role": "user",
                    "content": f"ลูกค้าสนใจ: {interest}"
                }
            ],
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        logger.info(f"✅ Received recommendations: {result}")
        
        # แปลงคำแนะนำเป็นสินค้าจริง
        all_product_recommendations = []
        for rec in result["recommendations"]:
            if rec["category"] in PRODUCT_CATALOG:
                # Get all products from the category
                products = PRODUCT_CATALOG[rec["category"]]
                for p in products:
                    product_display = ProductDisplay(
                        category=rec["category"],
                        product=p["name"],
                        price=p["price"],
                        reason=f"{rec['reason']} - {p['description']}",
                        confidence=rec["confidence"],
                        image=p["image"]
                    )
                    all_product_recommendations.append(product_display)
        
        # Sort all products by confidence (high to low)
        all_product_recommendations.sort(key=lambda x: x.confidence, reverse=True)
        
        logger.info(f"✅ Generated {len(all_product_recommendations)} product recommendations")
        return all_product_recommendations
        
    except Exception as e:
        error_details = traceback.format_exc()
        logger.error(f"❌ Error in recommendation generation:\n{error_details}")
        return None

def render_product_cards(products: List[ProductDisplay]):
    """แสดงสินค้าในรูปแบบ card"""
    st.markdown("""
    <style>
    .product-card {
        background-color: var(--st-card-background);
        border-radius: 15px;
        padding: 20px;
        margin: 15px 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
        border: 1px solid rgba(128, 128, 128, 0.1);
    }
    
    .product-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.15);
    }
    
    .product-title {
        color: var(--st-text);
        font-size: 1.2rem;
        font-weight: 600;
        margin: 10px 0;
    }
    
    .price-tag {
        background-color: #ff4b4b;
        color: white;
        padding: 8px 15px;
        border-radius: 25px;
        font-weight: 600;
        display: inline-block;
    }
    
    .confidence-badge {
        background-color: #4CAF50;
        color: white;
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # แสดงผลแบบเรียงลำดับ
    cols = st.columns(3)
    for idx, item in enumerate(products):
        with cols[idx % 3]:
            confidence_percent = int(item.confidence * 100)
            st.markdown(f"""
            <div class="product-card">
                <div style="width:100%; height:200px; background-color:rgba(128,128,128,0.1); border-radius:10px; margin-bottom:15px;">
                    <img src="{item.image}" style="width:100%; height:100%; object-fit:cover; border-radius:10px;" alt="{item.product}">
                </div>
                <div class="product-title">{item.product}</div>
                <div style="color: #666; font-size: 0.9rem; margin: 5px 0;">📂 {item.category}</div>
                <div style="display:flex; justify-content:space-between; align-items:center; margin:15px 0;">
                    <span class="price-tag">฿{item.price:,}</span>
                    <span class="confidence-badge">ความเหมาะสม {confidence_percent}%</span>
                </div>
                <p style="margin:10px 0;">{item.reason}</p>
                <div style="display:flex; gap:10px; margin-top:20px;">
                    <button style="background-color:#4CAF50; color:white; border:none; padding:8px 15px; border-radius:5px; cursor:pointer; width:100%;">
                        🛒 ใส่ตะกร้า
                    </button>
                    <button style="background-color:#2196F3; color:white; border:none; padding:8px 15px; border-radius:5px; cursor:pointer;">
                        ❤️
                    </button>
                </div>
            </div>
            """, unsafe_allow_html=True)

def main():
    # เริ่มต้น logging
    global logger
    logger = setup_logging()
    
    st.title("🛍️ AI Product Recommender")
    
    # API Key Management
    st.sidebar.title("⚙️ ตั้งค่าระบบ")
    
    with st.sidebar.expander("🔐 API Settings", expanded=True):
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            value=st.session_state.get('openai_api_key', ''),
            help="รูปแบบ API key จะขึ้นต้นด้วย 'sk-'"
        )
        
        col1, col2 = st.columns([1,1])
        with col1:
            if st.button("💾 บันทึก API Key", use_container_width=True):
                if not api_key:
                    logger.warning("⚠️ Empty API key submitted")
                    st.sidebar.warning("⚠️ กรุณาใส่ API key")
                elif not validate_api_key(api_key):
                    logger.warning("⚠️ Invalid API key format")
                    st.sidebar.error("⚠️ รูปแบบ API key ไม่ถูกต้อง")
                else:
                    st.session_state['openai_api_key'] = api_key
                    client = init_openai_client()
                    if client:
                        logger.info("✅ API key saved and connection tested successfully")
                        st.session_state['client'] = client
                        st.sidebar.success("✅ เชื่อมต่อสำเร็จ!")
                        st.rerun()
        
        with col2:
            if st.button("🗑️ ลบ API Key", use_container_width=True):
                logger.info("🗑️ Clearing API key from session state")
                st.session_state.pop('openai_api_key', None)
                st.session_state.pop('client', None)
                st.rerun()

    # แสดงสถานะการเชื่อมต่อ
    if 'client' in st.session_state:
        st.sidebar.success("🟢 สถานะ: เชื่อมต่อแล้ว")
    else:
        st.sidebar.error("🔴 สถานะ: ยังไม่ได้เชื่อมต่อ")

    # Categories
    st.sidebar.title("🏷️ หมวดหมู่สินค้า")
    for category in PRODUCT_CATALOG.keys():
        st.sidebar.write(f"• {category}")

# Debug Mode
    if st.sidebar.checkbox("🔍 Debug Mode"):
        st.sidebar.write("📋 Latest Logs:")
        try:
            with open(f"logs/recommender_{datetime.now().strftime('%Y%m%d')}.log", 'r', encoding='utf-8') as f:
                logs = f.readlines()[-10:]  # แสดง 10 บรรทัดล่าสุด
                st.sidebar.code(''.join(logs))
        except FileNotFoundError:
            st.sidebar.info("ยังไม่มี logs สำหรับวันนี้")

    # เช็คการเชื่อมต่อก่อนแสดงส่วนอื่นๆ
    if 'client' not in st.session_state:
        st.warning("🔑 กรุณาใส่ OpenAI API Key และทดสอบการเชื่อมต่อก่อนใช้งาน")
        st.info("""
        💡 **วิธีการรับ API Key:**
        1. ไปที่ https://platform.openai.com/account/api-keys
        2. Log in หรือสร้าง account
        3. คลิก "Create new secret key"
        4. คัดลอก API key มาใส่ในช่องด้านบน
        
        ⚠️ **หมายเหตุ:**
        - API key จะขึ้นต้นด้วย 'sk-'
        - ต้องมีการชำระเงินที่ OpenAI ก่อนใช้งาน
        - สามารถตรวจสอบยอดการใช้งานได้ที่ https://platform.openai.com/usage
        """)
        st.stop()

    # Main Interface
    interest = st.text_area(
        "🤔 บอกเราเกี่ยวกับความสนใจของคุณ",
        placeholder="เช่น: ฉันชอบถ่ายรูปวิวทิวทัศน์ตอนท่องเที่ยว",
        key="interest_input"
    )

    # Tips and Examples
    with st.expander("💡 ตัวอย่างการบอกความสนใจ"):
        st.markdown("""
        ยิ่งคุณให้รายละเอียดมาก ระบบยิ่งแนะนำได้ตรงใจ! ลองบอกเราเกี่ยวกับ:
        - กิจกรรมที่คุณชอบทำ
        - สไตล์ที่ชื่นชอบ
        - เป้าหมายที่ต้องการ
        - ประสบการณ์ที่ผ่านมา
        
        **ตัวอย่าง:**
        ```
        ฉันชอบถ่ายภาพวิวทิวทัศน์ โดยเฉพาะตอนพระอาทิตย์ขึ้นและตก 
        กำลังมองหาอุปกรณ์ที่จะช่วยให้ภาพสวยขึ้น
        ```
        """)

    if st.button("🎯 ขอคำแนะนำ", use_container_width=True):
        if interest:
            with st.spinner("🤖 กำลังวิเคราะห์ความสนใจของคุณ..."):
                products = get_recommendations_with_products(
                    interest, 
                    st.session_state['client']
                )
                
                if products:
                    render_product_cards(products)
                else:
                    st.error("❌ ไม่สามารถสร้างคำแนะนำได้ กรุณาลองใหม่อีกครั้ง")
                    if st.checkbox("🔍 แสดงรายละเอียด error"):
                        try:
                            with open(f"logs/recommender_{datetime.now().strftime('%Y%m%d')}.log", 'r', encoding='utf-8') as f:
                                logs = f.readlines()[-10:]
                                st.code(''.join(logs))
                        except FileNotFoundError:
                            st.info("ไม่พบไฟล์ log")
        else:
            st.warning("⚠️ กรุณาบอกความสนใจของคุณก่อน")

    # Footer
    st.markdown("---")
    st.markdown("""
    ### 💝 เพิ่มประสิทธิภาพการแนะนำ
    
    การแนะนำสินค้าจะแม่นยำขึ้นเมื่อคุณ:
    1. ให้รายละเอียดเกี่ยวกับความสนใจมากขึ้น
    2. ระบุวัตถุประสงค์การใช้งานที่ชัดเจน
    3. บอกระดับประสบการณ์ของคุณ
    
    💡 **Tip:** ลองบอกเราว่าคุณจะนำสินค้าไปใช้ทำอะไร และมีประสบการณ์มากน้อยแค่ไหน
    """)

if __name__ == "__main__":
    main()
