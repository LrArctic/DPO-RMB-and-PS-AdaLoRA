from PIL import Image

# 替换成你自己的三张图片路径
img_path1 = "0000.png"   # 第一张图（最上面）
img_path2 = "0001.png"   # 第二张图（中间）
img_path3 = "0002.png"   # 第三张图（最下面）

# 打开三张图片
img1 = Image.open(img_path1)
img2 = Image.open(img_path2)
img3 = Image.open(img_path3)

# -------------------------------
# 选项1：强制统一宽度为最窄的那张（推荐，保持比例）
# -------------------------------
min_width = min(img1.width, img2.width, img3.width)

# 按比例缩放到统一宽度
img1 = img1.resize((min_width, int(img1.height * min_width / img1.width)))
img2 = img2.resize((min_width, int(img2.height * min_width / img2.width)))
img3 = img3.resize((min_width, int(img3.height * min_width / img3.width)))

# 计算总高度
total_height = img1.height + img2.height + img3.height

# 创建新的空白图片（白色背景，你也可以改成 (255,255,255,0) 透明）
combined = Image.new('RGB', (min_width, total_height), (255, 255, 255))

# 依次粘贴
combined.paste(img1, (0, 0))
combined.paste(img2, (0, img1.height))
combined.paste(img3, (0, img1.height + img2.height))

# # 保存为 PNG（无损）
# combined.save("combined_vertical.png", "PNG", quality=95)

# print("已保存为 combined_vertical.png")
combined.save("combined_result.pdf", "PDF")

print("已保存为单页 PDF: combined_result.pdf")