import fitz  # PyMuPDF

def combine_pdfs_2x2(input_files, output_file):
    if len(input_files) != 4:
        raise ValueError("需要提供恰好4个PDF文件")

    # 打开第一个文件获取尺寸
    first_doc = fitz.open(input_files[0])
    rect = first_doc[0].rect
    w, h = rect.width, rect.height
    
    # 创建目标 PDF：宽度为 2w，高度为 2h
    out_doc = fitz.open()
    out_page = out_doc.new_page(width=w * 2, height=h * 2)
    
    # 坐标定义：(x, y) 对应 2x2 的位置
    # (0,0) (1,0)
    # (0,1) (1,1)
    positions = [(0, 0), (1, 0), (0, 1), (1, 1)]
    
    for i, file_path in enumerate(input_files):
        src_doc = fitz.open(file_path)
        col, row = positions[i]
        
        # 定义放置的目标矩形区域
        # fitz.Rect(x0, y0, x1, y1)
        target_rect = fitz.Rect(
            col * w, row * h, 
            (col + 1) * w, (row + 1) * h
        )
        
        # 将源 PDF 页面作为矢量图层置入
        out_page.show_pdf_page(target_rect, src_doc, 0)
        src_doc.close()

    out_doc.save(output_file)
    out_doc.close()
    print(f"合成成功！保存为: {output_file}")

# 使用示例
files = ["./loss_comparison.pdf","./grad_norm_comparison.pdf", "./accuracy_comparison_false.pdf",  "./margins_comparison_false.pdf"]
combine_pdfs_2x2(files, "Training curves_2x2.pdf")