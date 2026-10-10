import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import struct

# 路径基于脚本自身位置解析，避免从任意工作目录运行时找不到 ../table
_script_dir = Path(__file__).resolve().parent
project_distance_table_csv = (_script_dir / "../table/project_distance_table.csv").resolve()
project_area_table_csv = (_script_dir / "../table/project_area_table.csv").resolve()
project_area_table_bin = (_script_dir / "../table/project_area_table.bin").resolve()
project_area_hot_table = (_script_dir / "../table/project_area_hot_table.png").resolve()

# 投影面积表 bin 格式协议（见 README.md）：
# 头部 16 字节 = 魔数 'JORP' + 版本 + 宽度 + 高度（uint32 小端），
# 载荷 = float32 小端、行主序（height 行 × width 列），无效像素存 IEEE 754 +inf。
# 版本号标识载荷格式；dtype/布局变更必须升版本，读写两端共用以下常量。
_AREA_BIN_MAGIC = b"JORP"
_AREA_BIN_VERSION = 3
_AREA_BIN_DTYPE = "<f4"


def _parse_point(cell):
    """解析单个单元格为 (x, y) 投影点。

    投影点格式为 [x|y]（x 横向距离，y 纵向距离），无效像素为 [inf|inf]。
    兼容不带括号的 inf 写法，返回 (inf, inf)。
    """
    parts = cell.strip("[]").split("|")
    if len(parts) == 1:
        v = float(parts[0])
        if not np.isfinite(v):
            return (np.inf, np.inf)
        raise ValueError(f"单元格 {cell!r} 是单个数值，不是有效的 [x|y] 投影点")
    if len(parts) != 2:
        raise ValueError(f"单元格 {cell!r} 不是有效的 [x|y] 格式")
    return (float(parts[0]), float(parts[1]))


def load_distance_table(csv_path):
    """从投影距离表 CSV 读取每像素的投影点坐标。

    投影距离是像素在投影平面上的投影点相对光心的坐标 [x|y]：
    x 为横向距离，y 为纵向距离，与投影表格式一致（无效像素为 [inf|inf]）。
    返回 shape=(rows, cols, 2) 的 float 数组，无效像素为 [inf, inf]。
    """
    table = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            cells = [c.strip() for c in line.split(",") if c.strip()]
            if not cells:
                continue
            try:
                pts = [_parse_point(c) for c in cells]
            except ValueError as e:
                raise ValueError(f"csv 第 {line_no} 行存在无法解析的数值: {e}") from e
            table.append(np.asarray(pts, dtype=float))
    if not table:
        raise ValueError(f"距离表为空: {csv_path}")
    if len({row.shape[0] for row in table}) != 1:
        raise ValueError("距离表每行列数不一致")
    return np.array(table)


def compute_projected_area(coord_table):
    """按四个斜对角像素的投影点围成四边形，计算每像素的投影面积。

    中间像素 (i, j) 的投影面积 = 四个斜对角像素 (i-1, j-1)、(i-1, j+1)、
    (i+1, j+1)、(i+1, j-1) 的投影点围成四边形的面积（鞋带公式）。

    边界像素缺少的对角像素用存在的对角像素镜像补全（镜像中心为中间像素
    的投影点）：
    - 缺两个对角（边像素）：优先左右镜像（缺左/右对角时）或上下镜像
      （缺上/下对角时），只翻转 x 或 y 一个坐标；
    - 缺三个对角（角像素）：仅有的对角先做左右镜像、再上下镜像、
      最后对角镜像（点对称，两个坐标都翻转）。

    自身、所用对角像素或镜像源像素无效（inf）时，投影面积记为 inf。
    返回 shape=(rows, cols) 的 float 数组。
    """
    rows, cols, _ = coord_table.shape
    i_all = np.arange(rows)[:, None]
    j_all = np.arange(cols)[None, :]
    cx = coord_table[..., 0]
    cy = coord_table[..., 1]

    def grab(di, dj):
        """取 (i+di, j+dj) 处的投影点，越界时取边界值。"""
        return coord_table[np.clip(i_all + di, 0, rows - 1),
                           np.clip(j_all + dj, 0, cols - 1)]

    # 左右镜像（x 关于中心翻转）、上下镜像（y 关于中心翻转）
    def lr(src):
        return np.stack([2.0 * cx - src[..., 0], src[..., 1]], axis=-1)

    def ud(src):
        return np.stack([src[..., 0], 2.0 * cy - src[..., 1]], axis=-1)

    # 四个对角像素的投影点；出界处用存在的对角像素镜像补全。
    # inf 坐标参与乘减会产生 nan 与 RuntimeWarning，无效值随后被掩码剔除
    with np.errstate(invalid="ignore"):
        # 左上角 (i-1,j-1)：界内取原值；角点 (0,0) 仅剩右下角做对角镜像；
        # 顶行缺左上/右上两角，用左下角上下镜像；左列缺左上/左下两角，
        # 用右上角左右镜像
        tl_in = ((i_all >= 1) & (j_all >= 1))[..., None]
        tl00 = ((i_all == 0) & (j_all == 0))[..., None]
        tl_row0 = (i_all == 0)[..., None]
        tl = np.where(tl_in, grab(-1, -1),
             np.where(tl00, 2.0 * coord_table - grab(1, 1),
               np.where(tl_row0, ud(grab(1, -1)), lr(grab(-1, 1)))))

        # 右上角 (i-1,j+1)：角点 (0,cols-1) 仅剩左下角做对角镜像；
        # 顶行用右下角上下镜像；右列缺右上/右下两角，用左上角左右镜像
        tr_in = ((i_all >= 1) & (j_all <= cols - 2))[..., None]
        tr0c = ((i_all == 0) & (j_all == cols - 1))[..., None]
        tr_row0 = (i_all == 0)[..., None]
        tr = np.where(tr_in, grab(-1, 1),
             np.where(tr0c, 2.0 * coord_table - grab(1, -1),
               np.where(tr_row0, ud(grab(1, 1)), lr(grab(-1, -1)))))

        # 右下角 (i+1,j+1)：角点 (rows-1,cols-1) 仅剩左上角做对角镜像；
        # 底行用右上角上下镜像；右列缺右上/右下两角，用左下角左右镜像
        br_in = ((i_all <= rows - 2) & (j_all <= cols - 2))[..., None]
        brlc = ((i_all == rows - 1) & (j_all == cols - 1))[..., None]
        br_rowL = (i_all == rows - 1)[..., None]
        br = np.where(br_in, grab(1, 1),
             np.where(brlc, 2.0 * coord_table - grab(-1, -1),
               np.where(br_rowL, ud(grab(-1, 1)), lr(grab(1, -1)))))

        # 左下角 (i+1,j-1)：角点 (rows-1,0) 仅剩右上角做对角镜像；
        # 底行用左上角上下镜像；左列缺左上/左下两角，用右下角左右镜像
        bl_in = ((i_all <= rows - 2) & (j_all >= 1))[..., None]
        blr0 = ((i_all == rows - 1) & (j_all == 0))[..., None]
        bl_rowL = (i_all == rows - 1)[..., None]
        bl = np.where(bl_in, grab(1, -1),
             np.where(blr0, 2.0 * coord_table - grab(-1, 1),
               np.where(bl_rowL, ud(grab(-1, -1)), lr(grab(1, 1)))))

        # 鞋带公式: A = 0.5 * |Σ x_k*y_{k+1} - Σ y_k*x_{k+1}|
        sum1 = (tl[..., 0] * tr[..., 1] + tr[..., 0] * br[..., 1]
                + br[..., 0] * bl[..., 1] + bl[..., 0] * tl[..., 1])
        sum2 = (tl[..., 1] * tr[..., 0] + tr[..., 1] * br[..., 0]
                + br[..., 1] * bl[..., 0] + bl[..., 1] * tl[..., 0])
        quad_area = 0.5 * np.abs(sum1 - sum2)

    corners_valid = (np.isfinite(tl).all(axis=-1) & np.isfinite(tr).all(axis=-1)
                     & np.isfinite(br).all(axis=-1) & np.isfinite(bl).all(axis=-1))
    center_valid = np.isfinite(coord_table).all(axis=-1)
    valid = corners_valid & center_valid
    return np.where(valid, quad_area, np.inf)


def save_area_table(area_table, csv_path):
    """将投影面积表保存为 CSV。

    格式与投影表一致：每行 320 列、每格 [value]（面积标量），
    无效像素保存为 [inf]。area_table 为 shape=(rows, cols) 的 float 数组。
    """
    with open(csv_path, "w", encoding="utf-8") as f:
        for row in area_table:
            f.write(",".join(f"[{v}]" for v in row) + "\n")


def save_area_table_bin(area_table, bin_path):
    """将投影面积表保存为二进制 bin 文件（见 README.md 的格式协议）。

    头部 16 字节：魔数 'JORP' + 版本 + 宽度 + 高度（uint32 小端）；
    载荷：float32 小端、行主序，无效像素（inf）存 IEEE 754 +inf。
    area_table 为 shape=(rows, cols) 的 float 数组。
    """
    rows, cols = area_table.shape
    header = struct.pack("<4sIII", _AREA_BIN_MAGIC, _AREA_BIN_VERSION, cols, rows)
    payload = np.ascontiguousarray(area_table, dtype=_AREA_BIN_DTYPE)
    with open(bin_path, "wb") as f:
        f.write(header)
        f.write(payload.tobytes(order="C"))


def load_area_table_bin(bin_path):
    """读取投影面积表 bin 并校验头部（魔数/版本/宽高）。

    返回 shape=(height, width) 的 float32 数组，无效像素为 +inf。
    """
    with open(bin_path, "rb") as f:
        header = f.read(16)
    magic, version, cols, rows = struct.unpack("<4sIII", header)
    if magic != _AREA_BIN_MAGIC:
        raise ValueError(f"bin 魔数错误: {magic!r} != {_AREA_BIN_MAGIC!r}")
    if version != _AREA_BIN_VERSION:
        raise ValueError(f"bin 版本不支持: {version} != {_AREA_BIN_VERSION}")
    data = np.fromfile(bin_path, dtype=_AREA_BIN_DTYPE, offset=16)
    if data.size != rows * cols:
        raise ValueError(f"bin 数据长度不符: {data.size} != {rows * cols} ({rows}x{cols})")
    return data.reshape(rows, cols)


def save_area_hot_map(area_table, png_path):
    """根据投影面积表生成 plasma 热力图并保存为 PNG。

    输出图像与投影表同分辨率：每像素对应表格一格，不绘制坐标轴。
    面积动态范围大（随距离平方增长），先 log10 归一化再映射到 plasma
    色带；无效像素（inf）用热力最高的颜色（plasma 顶端亮黄色）表示。
    area_table 为 shape=(rows, cols) 的 float 数组。
    """
    with np.errstate(divide="ignore"):
        log_area = np.log10(area_table)      # inf -> inf, 0 -> -inf
    finite = np.isfinite(log_area)
    if not finite.any():
        raise ValueError("投影面积表没有有效像素")
    vmin = log_area[finite].min()
    vmax = log_area[finite].max()
    log_area[~finite & np.isfinite(area_table)] = vmin   # 面积<=0 用最低热度
    norm = np.clip((log_area - vmin) / (vmax - vmin), 0.0, 1.0)
    norm[np.isinf(area_table)] = 1.0         # inf 像素用热力最高的颜色
    rgb = plt.cm.plasma(norm)                # (rows, cols, 4)
    plt.imsave(png_path, rgb)


if __name__ == "__main__":
    dist = load_distance_table(project_distance_table_csv)
    area = compute_projected_area(dist)
    save_area_table(area, project_area_table_csv)
    save_area_table_bin(area, project_area_table_bin)
    save_area_hot_map(area, project_area_hot_table)
    print(f"投影距离表: {project_distance_table_csv}")
    print(f"投影面积表: {project_area_table_csv}")
    print(f"投影面积表(bin): {project_area_table_bin}")
    print(f"投影面积热力图: {project_area_hot_table}")
    print(f"有效像素: {int(np.isfinite(area).sum())}, 无效像素: {int(np.isinf(area).sum())}")
