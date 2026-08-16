import base64
import numpy as np
from PIL import Image

# 生成一个简单的测试图片
img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
img.save("test.jpg")

# 转 base64
with open("test.jpg", "rb") as f:
    b64 = base64.b64encode(f.read()).decode("utf-8")
    print(b64)