import os
import sys
import tarfile
import collections
import torch.utils.data as data
import shutil
import numpy as np
from torchvision.transforms.functional import to_pil_image
from PIL import Image
from torchvision.datasets.utils import download_url, check_integrity


def voc_cmap(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap

import os
from xml.etree import ElementTree as ET
from PIL import Image
import torch
import numpy as np
from torchvision.transforms import functional as F

import os
import torch
import numpy as np
from PIL import Image
from xml.etree import ElementTree as ET
from torchvision import transforms
import torch.utils.data as data
from PIL import ImageDraw
from collections import namedtuple

class EggDataset(data.Dataset):
    from collections import namedtuple

# Định nghĩa cấu trúc cho lớp EggClass
    EggClass = namedtuple('EggClass', ['name', 'id', 'ignore_in_eval', 'color'])

# Tạo danh sách các lớp EggClass
    egg_classes = [
    EggClass('unlabeled', 0, 'True', (0, 0, 0)),
    EggClass('broken yolk', 1, 'False', (111, 74, 0)),
    EggClass('chamber', 2, 'False', (150, 100, 100)),
    EggClass('crack', 3, 'False', (81, 0, 81)),
    EggClass('embryo', 4, 'False', (244, 35, 232)),
    EggClass('spoiled area', 5, 'False', (102, 102, 156))
    ] 
    class_names = [c.name for c in egg_classes]

    cmap = voc_cmap() 

    def __init__(self, root, split, transforms=None):
        """
        Args:
            root (str): Đường dẫn tới thư mục dataset chứa cả file ảnh và file XML.
            transforms (callable, optional): Các phép biến đổi ảnh (transform).
        """
        self.root = root
        self.split = split
        self.transforms = transforms

        # Lấy danh sách file ảnh (.jpg, .png, ... )
        self.files = [f for f in os.listdir(root + split) if f.endswith(('.jpg', '.png'))]
        assert len(self.files) > 0, "Không tìm thấy file ảnh nào trong thư mục!"

    def __len__(self):
        return len(self.files)

    def parse_xml(self, xml_path):
        """
        Phân tích cú pháp file XML để lấy nhãn và polygon.
        Args:
            xml_path (str): Đường dẫn tới file .xml.
        Returns:
            dict: {'mask': Tensor, 'labels': Tensor}
        """
        tree = ET.parse(xml_path)
        root = tree.getroot()

        labels = []
        pixel_ids = []
        polygons = []

        # Duyệt qua từng đối tượng (object) trong file XML
        for obj in root.findall("object"):
            name = obj.find("name").text
            labels.append(name)

            # Lấy polygon
            polygon_points = []

            # Lấy tất cả các phần tử <x> và <y> trong thẻ <polygon>
            x_elements = obj.findall("polygon/x")
            y_elements = obj.findall("polygon/y")

            # Đảm bảo số lượng <x> và <y> là đồng nhất
            if len(x_elements) == len(y_elements):
                for x_elem, y_elem in zip(x_elements, y_elements):
                    # Kiểm tra xem phần tử có tồn tại không trước khi lấy giá trị
                    if x_elem is not None and y_elem is not None:
                        x = float(x_elem.text)
                        y = float(y_elem.text)
                        polygon_points.append((x, y))

            pixel_id = self.class_names.index(name)
            pixel_ids.append(pixel_id)
            
        # Lấy kích thước ảnh từ phần tử <size>
        width = int(root.find("size").find("width").text)  # Chuyển sang int
        height = int(root.find("size").find("height").text)  # Chuyển sang int

        # Tạo mask cho từng đối tượng (polygon)
        mask = Image.new("L", (width, height), 0)  # Khởi tạo ảnh đen (0)
        draw = ImageDraw.Draw(mask)

        for idx, polygon in enumerate(polygons):
            pixel_id = pixel_ids[idx]
            draw.polygon(polygon, outline=pixel_id, fill=pixel_id)  # Vẽ polygon lên mask (giá trị 1 cho đối tượng)

        mask = np.array(mask)
        mask = torch.tensor(mask, dtype=torch.uint8)

        return mask

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index
        Returns:
            tuple: (image, target), với target chứa mask và labels.
        """
        # Lấy tên file ảnh
        img_file = self.files[idx]
        img_path = os.path.join(self.root, self.split, img_file)

        # Tìm tệp XML tương ứng (giả định cùng tên với file ảnh, khác phần mở rộng)
        xml_file = img_file.rsplit('.', 1)[0] + '.xml'
        xml_path = os.path.join(self.root, self.split, xml_file)

        assert os.path.exists(xml_path), f"Không tìm thấy file XML: {xml_path}"

        # Đọc ảnh
        img = Image.open(img_path).convert("RGB")

        # Phân tích cú pháp file XML
        target = self.parse_xml(xml_path)
        mask = to_pil_image(target)

        # Áp dụng transform nếu cần
        if self.transforms:
            img,mask = self.transforms(img, mask)

        return img, mask
    
    @classmethod
    def decode_target(cls, mask):
        """Giải mã mask thành ảnh RGB"""
        return cls.cmap[mask]
