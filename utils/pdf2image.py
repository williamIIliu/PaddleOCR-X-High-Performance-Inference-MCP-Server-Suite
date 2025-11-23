#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDF转图片工具 - 使用PyMuPDF实现
支持将PDF文件转换为图片格式，适用于OCR处理
不依赖系统第三方库，仅使用PyMuPDF和Pillow
"""

import os
import argparse
from pathlib import Path
from typing import Optional, Union, List
import logging

try:
    import fitz  # PyMuPDF
    from PIL import Image
    import io
except ImportError as e:
    print(f"缺少必要的依赖包: {e}")
    print("请安装依赖: pip install PyMuPDF pillow")
    print("注意: PyMuPDF不需要额外的系统依赖")
    exit(1)


class PDF2ImageConverter:
    """PDF转图片转换器 - 基于PyMuPDF实现"""
    
    def __init__(self, dpi: int = 300):
        """
        初始化转换器
        
        Args:
            dpi: 图片分辨率，默认300 DPI，适合OCR处理
        """
        self.dpi = dpi
        # PyMuPDF使用缩放因子而不是DPI
        # 缩放因子 = DPI / 72 (72是PDF的默认DPI)
        self.zoom = dpi / 72.0
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志"""
        logger = logging.getLogger('PDF2Image')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def convert_pdf_to_images(
        self,
        input_path: Union[str, Path],
        output_dir: Union[str, Path],
        page_numbers: Optional[Union[int, List[int]]] = None,
        output_format: str = 'PNG',
        quality: int = 95
    ) -> List[str]:
        """
        将PDF转换为图片
        
        Args:
            input_path: PDF文件路径
            output_dir: 输出目录
            page_numbers: 要转换的页码，None表示全部页面，int表示单页，list表示多页
            output_format: 输出格式 ('PNG', 'JPEG', 'TIFF')
            quality: 图片质量 (1-100，仅对JPEG有效)
        
        Returns:
            生成的图片文件路径列表
        """
        input_path = Path(input_path)
        output_dir = Path(output_dir)
        
        # 检查输入文件
        if not input_path.exists():
            raise FileNotFoundError(f"PDF文件不存在: {input_path}")
        
        if not input_path.suffix.lower() == '.pdf':
            raise ValueError(f"输入文件不是PDF格式: {input_path}")
        
        # 创建输出目录
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"开始转换PDF: {input_path}")
        self.logger.info(f"输出目录: {output_dir}")
        self.logger.info(f"DPI设置: {self.dpi} (缩放因子: {self.zoom:.2f})")
        
        try:
            # 打开PDF文档
            pdf_document = fitz.open(str(input_path))
            total_pages = len(pdf_document)
            self.logger.info(f"PDF总页数: {total_pages}")
            
            # 确定要转换的页面
            if page_numbers is None:
                # 转换全部页面
                pages_to_convert = list(range(total_pages))
                self.logger.info(f"转换全部页面，共 {total_pages} 页")
            elif isinstance(page_numbers, int):
                # 转换单页 (转换为0-based索引)
                if page_numbers < 1 or page_numbers > total_pages:
                    raise ValueError(f"页码超出范围: {page_numbers} (总页数: {total_pages})")
                pages_to_convert = [page_numbers - 1]
                self.logger.info(f"转换第 {page_numbers} 页")
            elif isinstance(page_numbers, list):
                # 转换指定页面 (转换为0-based索引)
                pages_to_convert = []
                for page_num in page_numbers:
                    if page_num < 1 or page_num > total_pages:
                        raise ValueError(f"页码超出范围: {page_num} (总页数: {total_pages})")
                    pages_to_convert.append(page_num - 1)
                self.logger.info(f"转换指定页面: {page_numbers}")
            else:
                raise ValueError("page_numbers参数格式错误")
            
            # 创建变换矩阵
            mat = fitz.Matrix(self.zoom, self.zoom)
            
            # 转换页面为图片
            output_files = []
            base_name = input_path.stem
            
            for i, page_index in enumerate(pages_to_convert):
                # 获取页面
                page = pdf_document[page_index]
                
                # 渲染页面为像素图
                pix = page.get_pixmap(matrix=mat)
                
                # 转换为PIL Image
                img_data = pix.tobytes("png")
                image = Image.open(io.BytesIO(img_data))
                
                # 确定输出文件名
                if page_numbers is None:
                    # 全部页面
                    page_num = page_index + 1
                elif isinstance(page_numbers, int):
                    # 单页
                    page_num = page_numbers
                elif isinstance(page_numbers, list):
                    # 指定页面
                    page_num = page_numbers[i]
                
                output_filename = f"{base_name}_page_{page_num:03d}.{output_format.lower()}"
                output_path = output_dir / output_filename
                
                # 保存图片
                if output_format.upper() == 'JPEG':
                    # JPEG格式需要转换为RGB
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                    image.save(output_path, output_format, quality=quality, optimize=True)
                else:
                    image.save(output_path, output_format)
                
                output_files.append(str(output_path))
                self.logger.info(f"保存图片: {output_path} (尺寸: {image.size})")
                
                # 清理内存
                pix = None
                image.close()
            
            # 关闭PDF文档
            pdf_document.close()
            
            self.logger.info(f"转换完成，共生成 {len(output_files)} 个图片文件")
            return output_files
            
        except Exception as e:
            self.logger.error(f"转换失败: {str(e)}")
            raise
    
    def get_pdf_info(self, input_path: Union[str, Path]) -> dict:
        """
        获取PDF文档信息
        
        Args:
            input_path: PDF文件路径
            
        Returns:
            包含PDF信息的字典
        """
        input_path = Path(input_path)
        
        if not input_path.exists():
            raise FileNotFoundError(f"PDF文件不存在: {input_path}")
        
        try:
            pdf_document = fitz.open(str(input_path))
            
            info = {
                'filename': input_path.name,
                'page_count': len(pdf_document),
                'metadata': pdf_document.metadata,
                'is_encrypted': pdf_document.needs_pass,
                'is_pdf': pdf_document.is_pdf,
                'pages_info': []
            }
            
            # 获取每页信息
            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]
                page_info = {
                    'page_number': page_num + 1,
                    'width': page.rect.width,
                    'height': page.rect.height,
                    'rotation': page.rotation
                }
                info['pages_info'].append(page_info)
            
            pdf_document.close()
            return info
            
        except Exception as e:
            raise RuntimeError(f"获取PDF信息失败: {str(e)}")


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(
        description='PDF转图片工具 - 基于PyMuPDF实现，无需系统依赖',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 转换全部页面
  python pdf2image.py input.pdf output_dir/
  
  # 转换第5页
  python pdf2image.py input.pdf output_dir/ --page 5
  
  # 转换第1,3,5页
  python pdf2image.py input.pdf output_dir/ --pages 1 3 5
  
  # 设置DPI为150
  python pdf2image.py input.pdf output_dir/ --dpi 150
  
  # 输出JPEG格式
  python pdf2image.py input.pdf output_dir/ --format JPEG --quality 90
  
  # 查看PDF信息
  python pdf2image.py input.pdf --info
        """
    )
    
    parser.add_argument('input_path', help='输入PDF文件路径')
    parser.add_argument('output_dir', nargs='?', help='输出目录路径（查看信息时可选）')
    
    # 页面选择参数（互斥）
    page_group = parser.add_mutually_exclusive_group()
    page_group.add_argument('--page', type=int, help='转换指定单页（页码从1开始）')
    page_group.add_argument('--pages', type=int, nargs='+', help='转换指定多页（页码从1开始）')
    
    # 图片参数
    parser.add_argument('--dpi', type=int, default=300, 
                       help='图片分辨率DPI (默认: 300，适合OCR)')
    parser.add_argument('--format', choices=['PNG', 'JPEG', 'TIFF'], 
                       default='PNG', help='输出图片格式 (默认: PNG)')
    parser.add_argument('--quality', type=int, default=95, 
                       help='JPEG图片质量 1-100 (默认: 95)')
    
    # 功能参数
    parser.add_argument('--info', action='store_true', 
                       help='显示PDF文档信息')
    
    # 日志参数
    parser.add_argument('--verbose', '-v', action='store_true', 
                       help='显示详细日志')
    
    args = parser.parse_args()
    
    # 设置日志级别
    if args.verbose:
        logging.getLogger('PDF2Image').setLevel(logging.DEBUG)
    
    try:
        # 创建转换器
        converter = PDF2ImageConverter(dpi=args.dpi)
        
        # 如果只是查看信息
        if args.info:
            info = converter.get_pdf_info(args.input_path)
            print(f"\nPDF文档信息:")
            print(f"文件名: {info['filename']}")
            print(f"页数: {info['page_count']}")
            print(f"是否加密: {'是' if info['is_encrypted'] else '否'}")
            
            if info['metadata']:
                print(f"\n元数据:")
                for key, value in info['metadata'].items():
                    if value:
                        print(f"  {key}: {value}")
            
            print(f"\n页面信息:")
            for page_info in info['pages_info']:
                print(f"  第{page_info['page_number']}页: "
                      f"{page_info['width']:.1f}x{page_info['height']:.1f} "
                      f"(旋转: {page_info['rotation']}°)")
            
            return 0
        
        # 检查输出目录参数
        if not args.output_dir:
            print("错误: 转换图片时必须指定输出目录")
            return 1
        
        # 确定页面参数
        page_numbers = None
        if args.page:
            page_numbers = args.page
        elif args.pages:
            page_numbers = args.pages
        
        # 执行转换
        output_files = converter.convert_pdf_to_images(
            input_path=args.input_path,
            output_dir=args.output_dir,
            page_numbers=page_numbers,
            output_format=args.format,
            quality=args.quality
        )
        
        print(f"\n转换成功！生成了 {len(output_files)} 个图片文件:")
        for file_path in output_files:
            file_size = Path(file_path).stat().st_size
            print(f"  - {file_path} ({file_size:,} bytes)")
            
    except Exception as e:
        print(f"错误: {str(e)}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
