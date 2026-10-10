# seg_filter

分割结果过滤工具集：基于投影几何，把分割图按每个像素的投影面积加权处理。

## 投影面积表生成

`scripts/distance2area.py` 完整链路：

```
project_distance_table.csv ──load──> [x|y] 投影点
        ↓ compute_projected_area（斜对角四边形 + 边界镜像）
  投影面积表（192×320）
        ├──> project_area_table.csv     文本表
        ├──> project_area_table.bin     二进制表（见下方格式协议）
        └──> project_area_hot_table.png plasma 热力图（与表同分辨率）
```

运行（任意目录均可，路径按脚本位置解析）：

```bash
python3 toolkit/seg_filter/scripts/distance2area.py
```

## 投影面积表 bin 格式协议

`project_area_table.bin` 为带头的固定二进制格式，读写两端必须遵循同一约定。

| 偏移 | 大小 | 内容 |
|---|---|---|
| 0 | 4 | 魔数 `'JORP'` |
| 4 | 4 | 版本号 `uint32` 小端 = **3** |
| 8 | 4 | 宽度 `uint32` 小端 = 320 |
| 12 | 4 | 高度 `uint32` 小端 = 192 |
| 16 | `w*h*4` | `float32` 小端，**行主序**（`height` 行 × `width` 列） |

- 文件总大小固定 = `16 + 320*192*4 = 245776` 字节
- 无效像素存 **IEEE 754 `+inf`**（字节 `0x7F800000`），读取端用 `isinf` 判断
- 版本号标识载荷格式：dtype、字节序、行列顺序、布局变更必须升版本（读写两端用同一常量，见 `distance2area.py` 顶部 `_AREA_BIN_*`）
- 魔数/版本/宽高不匹配时读取端应拒绝加载

### Python 读写接口

```python
from distance2area import save_area_table_bin, load_area_table_bin

save_area_table_bin(area_table, "project_area_table.bin")  # area_table: (rows, cols) float
area = load_area_table_bin("project_area_table.bin")       # -> (192, 320) float32, 无效像素为 +inf
```

### C/C++ 读取示例

```c
#include <cstdio>
#include <cmath>
#include <cstdint>

#pragma pack(push, 1)
struct AreaBinHeader {
    char     magic[4];   // 'JORP'
    uint32_t version;    // 3
    uint32_t width;      // 320
    uint32_t height;     // 192
};
#pragma pack(pop)

FILE* f = fopen("project_area_table.bin", "rb");
AreaBinHeader h;
fread(&h, sizeof(h), 1, f);              // 小端 x86 直接读
// 校验 h.magic == 'JORP'、h.version == 3
std::vector<float> area(h.width * h.height);
fread(area.data(), sizeof(float), area.size(), f);
// 无效像素: std::isinf(area[i * h.width + j])
```
