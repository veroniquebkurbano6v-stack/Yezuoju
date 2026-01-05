import os
import sys
import base64
import json
import requests
from typing import List, Dict, Any, Optional
from pathlib import Path
import fitz  # PyMuPDF
from pdf2image import convert_from_path
import io
from PIL import Image
import logging
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class UmiOCRClient:
    """Umi-OCR客户端类，用于调用Umi-OCR API"""
    
    def __init__(self, host: str = "127.0.0.1", port: int = 1224):
        """
        初始化Umi-OCR客户端
        
        Args:
            host: Umi-OCR服务器地址
            port: Umi-OCR服务器端口
        """
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"
        self.api_url = f"{self.base_url}/api/ocr"
    
    def is_available(self) -> bool:
        """
        检查Umi-OCR服务是否可用
        
        Returns:
            bool: 服务是否可用
        """
        try:
            response = requests.get(f"{self.base_url}/api/status", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"无法连接到Umi-OCR服务: {e}")
            return False
    
    def ocr_image(self, image_data: bytes, language: str = "jpn") -> Optional[str]:
        """
        对图片进行OCR识别
        
        Args:
            image_data: 图片的二进制数据
            language: OCR语言，默认为日语"jpn"
            
        Returns:
            str: 识别出的文本，失败返回None
        """
        try:
            # 将图片转换为base64编码
            image_base64 = base64.b64encode(image_data).decode('utf-8')
            
            # 准备请求数据
            payload = {
                "image": image_base64,
                "language": language,
                "is_table": False,
                "is_pdf": False
            }
            
            # 发送请求
            response = requests.post(
                self.api_url,
                json=payload,
                timeout=60  # 增加超时时间，因为OCR可能需要较长时间
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get("code") == 0:
                    return result.get("data", {}).get("text", "")
                else:
                    logger.error(f"OCR识别失败: {result.get('msg', '未知错误')}")
                    return None
            else:
                logger.error(f"API请求失败: HTTP {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"OCR识别过程出错: {e}")
            return None
    
    def ocr_image_file(self, image_path: str, language: str = "jpn") -> Optional[str]:
        """
        对图片文件进行OCR识别
        
        Args:
            image_path: 图片文件路径
            language: OCR语言
            
        Returns:
            str: 识别出的文本
        """
        try:
            with open(image_path, "rb") as f:
                image_data = f.read()
            return self.ocr_image(image_data, language)
        except Exception as e:
            logger.error(f"读取图片文件失败: {e}")
            return None

class JapanesePDFExtractor:
    """日语PDF提取器，使用Umi-OCR技术"""
    
    def __init__(self, umi_ocr_host: str = "127.0.0.1", umi_ocr_port: int = 1224):
        """
        初始化日语PDF提取器
        
        Args:
            umi_ocr_host: Umi-OCR服务器地址
            umi_ocr_port: Umi-OCR服务器端口
        """
        self.ocr_client = UmiOCRClient(umi_ocr_host, umi_ocr_port)
        self.temp_dir = Path("temp_images")
        self.temp_dir.mkdir(exist_ok=True)
    
    def pdf_to_images(self, pdf_path: str, max_pages: Optional[int] = None) -> List[str]:
        """
        将PDF转换为图片
        
        Args:
            pdf_path: PDF文件路径
            max_pages: 最大转换页数
            
        Returns:
            List[str]: 转换后的图片文件路径列表
        """
        try:
            logger.info(f"开始转换PDF: {pdf_path}")
            
            # 使用pdf2image转换PDF为图片
            images = convert_from_path(
                pdf_path,
                dpi=300,  # 高分辨率以提高OCR准确度
                first_page=1,
                last_page=max_pages
            )
            
            image_paths = []
            pdf_name = Path(pdf_path).stem
            
            for i, image in enumerate(images):
                # 保存图片
                image_path = self.temp_dir / f"{pdf_name}_page_{i+1}.png"
                image.save(image_path, "PNG")
                image_paths.append(str(image_path))
                logger.info(f"已转换第 {i+1} 页: {image_path}")
            
            return image_paths
            
        except Exception as e:
            logger.error(f"PDF转换失败: {e}")
            return []
    
    def extract_text_from_pdf(self, pdf_path: str, max_pages: Optional[int] = None) -> Dict[str, Any]:
        """
        从PDF中提取文本
        
        Args:
            pdf_path: PDF文件路径
            max_pages: 最大处理页数
            
        Returns:
            Dict[str, Any]: 包含提取结果的字典
        """
        # 检查Umi-OCR服务是否可用
        if not self.ocr_client.is_available():
            return {
                "success": False,
                "error": "Umi-OCR服务不可用，请确保Umi-OCR正在运行",
                "pages": []
            }
        
        try:
            # 转换PDF为图片
            image_paths = self.pdf_to_images(pdf_path, max_pages)
            
            if not image_paths:
                return {
                    "success": False,
                    "error": "PDF转换图片失败",
                    "pages": []
                }
            
            # 对每张图片进行OCR识别
            pages_data = []
            total_pages = len(image_paths)
            
            logger.info(f"开始OCR识别，共 {total_pages} 页")
            
            for i, image_path in enumerate(image_paths):
                logger.info(f"正在处理第 {i+1}/{total_pages} 页")
                
                # OCR识别
                ocr_text = self.ocr_client.ocr_image_file(image_path, language="jpn")
                
                if ocr_text:
                    page_data = {
                        "page_number": i + 1,
                        "text": ocr_text.strip(),
                        "image_path": image_path,
                        "extraction_method": "umi_ocr"
                    }
                    pages_data.append(page_data)
                    logger.info(f"第 {i+1} 页识别成功，文本长度: {len(ocr_text)}")
                else:
                    page_data = {
                        "page_number": i + 1,
                        "text": "",
                        "image_path": image_path,
                        "extraction_method": "umi_ocr_failed"
                    }
                    pages_data.append(page_data)
                    logger.warning(f"第 {i+1} 页识别失败")
            
            # 清理临时文件
            self._cleanup_temp_files(image_paths)
            
            # 统计信息
            total_text_length = sum(len(page["text"]) for page in pages_data)
            successful_pages = sum(1 for page in pages_data if page["text"])
            
            return {
                "success": True,
                "pdf_path": pdf_path,
                "total_pages": total_pages,
                "successful_pages": successful_pages,
                "total_text_length": total_text_length,
                "pages": pages_data,
                "extraction_summary": {
                    "method": "Umi-OCR",
                    "language": "Japanese",
                    "success_rate": successful_pages / total_pages if total_pages > 0 else 0
                }
            }
            
        except Exception as e:
            logger.error(f"PDF文本提取失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "pages": []
            }
    
    def _cleanup_temp_files(self, image_paths: List[str]):
        """
        清理临时图片文件
        
        Args:
            image_paths: 图片文件路径列表
        """
        try:
            for image_path in image_paths:
                if os.path.exists(image_path):
                    os.remove(image_path)
            logger.info("临时文件清理完成")
        except Exception as e:
            logger.warning(f"清理临时文件时出错: {e}")

def main():
    """主函数"""
    # PDF文件路径
    pdf_path = "src\data\source\Japanese\日语化物语.pdf"
    
    # 创建提取器实例
    extractor = JapanesePDFExtractor()
    
    # 提取文本
    logger.info("开始提取日语PDF文本...")
    result = extractor.extract_text_from_pdf(pdf_path, max_pages=5)  # 先处理前5页作为测试
    
    # 输出结果
    if result["success"]:
        logger.info("提取成功！")
        logger.info(f"总共 {result['total_pages']} 页")
        logger.info(f"成功识别 {result['successful_pages']} 页")
        logger.info(f"提取文本总长度: {result['total_text_length']} 字符")
        
        # 保存结果到JSON文件
        output_path = "src\data\japanese_extraction_result.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        logger.info(f"结果已保存到: {output_path}")
        
        # 显示每页的前100个字符作为预览
        for page in result["pages"]:
            page_text = page["text"]
            preview = page_text[:100] + "..." if len(page_text) > 100 else page_text
            logger.info(f"第 {page['page_number']} 页预览: {preview}")
    else:
        logger.error(f"提取失败: {result.get('error', '未知错误')}")

if __name__ == "__main__":
    main()