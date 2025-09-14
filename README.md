# StressFree-Chatbot
เพื่อนใจวัยเรียน AI (Student's Mind Mate AI)
📝 ภาพรวมโปรเจกต์ (Project Overview)
เพื่อนใจวัยเรียน AI คือโปรเจกต์ AI Chatbot ที่ถูกพัฒนาขึ้นเพื่อเป็นเพื่อนคุยและให้คำแนะนำเบื้องต้นแก่นักเรียนและนักศึกษาที่กำลังเผชิญกับความเครียดและความวิตกกังวลด้านการเรียน Chatbot นี้ทำหน้าที่เป็นผู้รับฟังที่เข้าอกเข้าใจ, ให้กำลังใจ, และแนะนำแนวทางการจัดการความเครียดเบื้องต้น เพื่อช่วยลดความเสี่ยงด้านสุขภาพจิตในกลุ่มเป้าหมาย

กลุ่มเป้าหมาย: นักเรียนระดับมัธยมปลายและนักศึกษามหาวิทยาลัย

ข้อจำกัดที่สำคัญ: Chatbot นี้เป็นเพียงเครื่องมือให้คำแนะนำเบื้องต้น ไม่ใช่การทดแทนการวินิจฉัยหรือการรักษาจากแพทย์หรือผู้เชี่ยวชาญด้านสุขภาพจิต หากผู้ใช้มีอาการรุนแรงหรือต้องการความช่วยเหลือเร่งด่วน จะมีการแนะนำให้ติดต่อผู้เชี่ยวชาญทันที

✨ คุณสมบัติหลัก (Core Features)
บุคลิกเป็นมิตรและเข้าอกเข้าใจ (Friendly & Empathetic Persona): ใช้ภาษาไทยที่อบอุ่น เป็นกันเอง และพร้อมรับฟังทุกปัญหา

การรับฟังเชิงรุก (Active Listening): สามารถจับใจความสำคัญของปัญหาและตอบสนองได้อย่างตรงจุด

การให้คำแนะนำเบื้องต้น (Initial Guidance): แนะนำเทคนิคการจัดการความเครียดที่นำไปใช้ได้จริง เช่น

เทคนิคการหายใจ (Breathing Exercises)

การจัดการเวลา (Time Management)

การทำสมาธิเบื้องต้น (Mindfulness)

พื้นที่ปลอดภัย ไม่ตัดสิน (Non-judgmental Space): ผู้ใช้สามารถระบายความรู้สึกได้อย่างอิสระโดยไม่ต้องกังวล

การแนะนำผู้เชี่ยวชาญ (Professional Referral): ในกรณีที่การสนทนามีความเสี่ยงสูง Chatbot จะแนะนำให้ผู้ใช้ปรึกษาผู้เชี่ยวชาญทันที

🛠️ ข้อกำหนดทางเทคนิค (Technical Specifications)
Backend: Python (FastAPI)

Frontend: HTML, CSS, Vanilla JavaScript

LLM Model: ออกแบบมาเพื่อทำงานร่วมกับ OpenThaiGPT-1.5-7b (หรือโมเดลอื่นที่ใกล้เคียง)

Inference Engine: VLLM เพื่อการประมวลผลโมเดลที่มีประสิทธิภาพ

Deployment: Docker

🚀 ขั้นตอนการติดตั้ง (Installation)
คุณสามารถติดตั้งและรันโปรเจกต์นี้บนเซิร์ฟเวอร์ของคุณได้ตามขั้นตอนต่อไปนี้

1. ข้อกำหนดเบื้องต้น (Prerequisites)
Hardware: Server พร้อม GPU ที่มี VRAM อย่างน้อย 6GB (แนะนำ NVIDIA RTX 3060 / 2060 Super หรือสูงกว่า)

Software:

Git

Docker และ NVIDIA Container Toolkit สำหรับการใช้งาน GPU ภายใน Container

Python 3.9+ (สำหรับ Local Development)

2. Clone Repository
git clone [https://github.com/your-username/students-mind-mate-ai.git](https://github.com/your-username/students-mind-mate-ai.git)
cd students-mind-mate-ai

3. ตั้งค่าโมเดล (Model Setup)
ดาวน์โหลดโมเดล OpenThaiGPT-1.5-7b ที่ผ่านการ Fine-tune และ Quantize (4-bit) ของคุณ

นำไฟล์โมเดลทั้งหมดไปวางไว้ในไดเรกทอรี /model

โปรดอ่านคำแนะนำเพิ่มเติมใน model/README.md

4. แก้ไข Dockerfile (ถ้าจำเป็น)
หากโมเดลของคุณต้องการ Dependencies เพิ่มเติม หรือมีการตั้งค่าที่แตกต่างไปจากปกติ ให้แก้ไข Dockerfile ตามความเหมาะสม

5. Build และ Run Docker Container
รันคำสั่งต่อไปนี้จากไดเรกทอรีรากของโปรเจกต์:

docker build -t mind-mate-ai .

docker run --gpus all -d -p 8000:8000 \
  -v $(pwd)/model:/app/model \
  --name mind-mate-container mind-mate-ai

--gpus all: อนุญาตให้ Container ใช้งาน GPU ได้

-d: รัน Container ในโหมด Detached (Background)

-p 8000:8000: Map Port 8000 ของเครื่อง Host เข้ากับ Port 8000 ของ Container

-v $(pwd)/model:/app/model: Mount ไดเรกทอรี model จากเครื่อง Host ไปยัง /app/model ใน Container เพื่อให้ Backend โหลดโมเดลได้

6. ตรวจสอบการทำงาน
เปิด Web Browser แล้วเข้าไปที่ http://<your-server-ip>:8000

คุณควรจะเห็นหน้าแชทของ "เพื่อนใจวัยเรียน AI" และสามารถเริ่มต้นการสนทนาได้

kullanım (Usage)
Chat Interface: พิมพ์ข้อความของคุณในช่องแชทและกด "ส่ง"

API Endpoint: หากต้องการเชื่อมต่อกับระบบอื่น สามารถส่ง POST Request มายัง http://<your-server-ip>:8000/api/chat ด้วย Payload รูปแบบ JSON:

{
  "message": "สวัสดีครับ"
}

📄 License
โปรเจกต์นี้อยู่ภายใต้ใบอนุญาตของ MIT
