"""
日语PDF API响应转Titles JSON格式工具
将api_response_debug_1.json的API响应格式转换为安徒生童话_titles.json的目标格式
"""

import json
import hashlib
import os
from datetime import datetime
from typing import Dict, List, Any, Optional


def load_api_response(file_path: str) -> Dict[str, Any]:
    """加载API响应JSON文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def transform_position_to_coordinates(position: List[float]) -> Dict[str, float]:
    """
    将position数组转换为坐标格式
    position: [x0, y0, x1, y0, x1, y1, x0, y1] - 8个点的多边形坐标
    转换为: {x0, y0, x1, y1} - 边界框坐标
    """
    if not position or len(position) < 8:
        return {
            "x0": 0.0,
            "y0": 0.0,
            "x1": 0.0,
            "y1": 0.0
        }

    x_coords = [position[0], position[2], position[4], position[6]]
    y_coords = [position[1], position[3], position[5], position[7]]

    return {
        "x0": float(min(x_coords)),
        "y0": float(min(y_coords)),
        "x1": float(max(x_coords)),
        "y1": float(max(y_coords))
    }


def extract_bounding_regions(raw_data: Dict[str, Any]) -> Dict[int, List[Dict[str, Any]]]:
    """
    从raw_json中提取所有边界区域信息
    返回: {page_id: [{text, coordinates, ...}, ...], ...}
    """
    coordinates_by_page = {}

    for key, value in raw_data.items():
        if not isinstance(value, dict):
            continue

        bounding_regions = value.get("bounding_regions", [])
        if not bounding_regions:
            continue

        for region in bounding_regions:
            if not isinstance(region, dict):
                continue

            page_id = region.get("page_id")
            if page_id is None:
                continue

            text = region.get("value", "")
            position = region.get("position", [])

            if page_id not in coordinates_by_page:
                coordinates_by_page[page_id] = []

            coordinates_by_page[page_id].append({
                "text": text,
                "coordinates": transform_position_to_coordinates(position)
            })

    return coordinates_by_page


def generate_text_hash(text: str) -> str:
    """生成文本的MD5哈希值（16位十六进制）"""
    if not text:
        return ""
    cleaned_text = clean_text(text)
    return hashlib.md5(cleaned_text.encode('utf-8')).hexdigest()[:16]


def clean_text(text: str) -> str:
    """清理文本中的特殊字符"""
    if not isinstance(text, str):
        return text
    text = text.replace("\\)", " ")
    text = text.replace("\\\\)", " ")
    text = text.replace("\\u3000", " ")
    text = text.replace("\u3000", " ")
    text = text.replace("\\", " ")
    text = text.replace("\n", " ")
    text = text.replace("\t", " ")
    text = text.replace("\r", " ")
    text = ' '.join(text.split())
    return text


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
    """将长文本分割成重叠的块"""
    if not text or len(text) <= chunk_size:
        return [text] if text else []

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
        if start >= text_len:
            break

    return chunks


def convert_api_response_to_titles_format(
    api_response: Dict[str, Any],
    pdf_filename: str,
    pdf_path: str,
    language: str = "Japanese"
) -> Dict[str, Any]:
    """
    将API响应转换为titles JSON格式

    Args:
        api_response: API响应数据
        pdf_filename: PDF文件名
        pdf_path: PDF文件路径
        language: 语言

    Returns:
        符合titles格式的JSON数据
    """
    result = api_response.get("result", {})

    llm_json = result.get("llm_json", {})
    raw_json = result.get("raw_json", {})

    coordinates_by_page = extract_bounding_regions(raw_json)

    document_chunks = []
    chunk_id_counter = {}
    section_titles = {}

    chapter_num = 1
    while True:
        title_key = f"chapter_{chapter_num}_title"
        content_key = f"chapter_{chapter_num}_content"

        if title_key not in llm_json:
            break

        title = llm_json.get(title_key)
        content = llm_json.get(content_key, "")

        if not content:
            chapter_num += 1
            continue

        section_titles[chapter_num] = title
        cleaned_content = clean_text(content)
        text_chunks = chunk_text(cleaned_content, chunk_size=512, overlap=50)

        page_range = {}
        for page_id, coords_list in coordinates_by_page.items():
            for coords_item in coords_list:
                if coords_item["text"] and coords_item["text"] in content:
                    if chapter_num not in page_range:
                        page_range[chapter_num] = {"start": page_id, "end": page_id}
                    else:
                        page_range[chapter_num]["start"] = min(page_range[chapter_num]["start"], page_id)
                        page_range[chapter_num]["end"] = max(page_range[chapter_num]["end"], page_id)

        for chunk_idx, chunk in enumerate(text_chunks):
            chunk_id_prefix = f"{pdf_filename.replace('.pdf', '')}_c{chapter_num:03d}"

            if chunk_id_prefix not in chunk_id_counter:
                chunk_id_counter[chunk_id_prefix] = 0
            chunk_num = chunk_id_counter[chunk_id_prefix]
            chunk_id_counter[chunk_id_prefix] += 1

            page_nums = []
            for page_id, coords_list in coordinates_by_page.items():
                for coords_item in coords_list:
                    if coords_item["text"] and coords_item["text"] in chunk:
                        if page_id not in page_nums:
                            page_nums.append(page_id)

            page_number = page_nums[0] if page_nums else chapter_num

            chunk_coordinates = None
            if page_number in coordinates_by_page and coordinates_by_page[page_number]:
                for coords_item in coordinates_by_page[page_number]:
                    if coords_item["text"] and coords_item["text"] in chunk:
                        chunk_coordinates = coords_item["coordinates"]
                        break

            if not chunk_coordinates and page_number in coordinates_by_page and coordinates_by_page[page_number]:
                chunk_coordinates = coordinates_by_page[page_number][0]["coordinates"]

            document_chunks.append({
                "id": f"{chunk_id_prefix}_p{page_number:03d}_c{chunk_num:03d}",
                "text": chunk,
                "page_number": page_number,
                "section_title": title,
                "chunk_index": chunk_idx,
                "total_chunks_in_page": len(text_chunks),
                "coordinates": chunk_coordinates if chunk_coordinates else {
                    "x0": 0.0,
                    "y0": 0.0,
                    "x1": 0.0,
                    "y1": 0.0
                },
                "text_hash": generate_text_hash(chunk)
            })

        chapter_num += 1

    parent_document = {
        "filename": pdf_filename,
        "file_path": pdf_path.replace("/", "\\"),
        "total_pages": len(coordinates_by_page),
        "language": language,
        "total_chunks": len(document_chunks),
        "processing_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    return {
        "parent_document": parent_document,
        "document_chunks": document_chunks
    }


def main():
    """主函数"""
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    api_response_path = os.path.join(base_dir, "src", "data", "api_response_debug_1.json")
    output_path = os.path.join(base_dir, "src", "data", "pages_title", "日语化物语_titles.json")
    pdf_filename = "日语化物语.pdf"
    pdf_path = "src\\data\\source\\Japanese\\日语化物语.pdf"

    print("加载API响应数据...")
    api_response = load_api_response(api_response_path)
    print(f"API响应加载成功，包含 {len(api_response.get('result', {}).get('llm_json', {}).get('chapter_1_content', ''))} 个章节")

    print("转换数据格式...")
    titles_data = convert_api_response_to_titles_format(
        api_response=api_response,
        pdf_filename=pdf_filename,
        pdf_path=pdf_path,
        language="Japanese"
    )

    print(f"转换完成，共生成 {len(titles_data['document_chunks'])} 个文本块")

    print(f"保存结果到 {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(titles_data, f, ensure_ascii=False, indent=2)

    print("转换完成！")

    return titles_data


if __name__ == "__main__":
    main()
