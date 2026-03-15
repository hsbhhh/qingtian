import cooler
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

mcool_file = r'Data\chroms\4DNFIJYULXT7.mcool'
refGene_file = r"Data\chroms\refGene.txt"
my_gene_file = r"Data\feature_genename.txt"

# =========================
# 1. 读取你的基因列表
# =========================
my_gene = pd.read_csv(my_gene_file, names=['gene'])
my_gene['gene'] = my_gene['gene'].astype(str).str.strip()
target_genes = set(my_gene['gene'].tolist())

# =========================
# 2. 读取 refGene
# refGene 常见 16 列
# =========================
ref_cols = [
    "bin", "name", "chrom", "strand",
    "txStart", "txEnd", "cdsStart", "cdsEnd",
    "exonCount", "exonStarts", "exonEnds",
    "score", "name2", "cdsStartStat", "cdsEndStat", "exonFrames"
]

ref = pd.read_csv(refGene_file, sep='\t', header=None, names=ref_cols)

# 基因符号列是 name2
ref['name2'] = ref['name2'].astype(str).str.strip()
ref['chrom'] = ref['chrom'].astype(str).str.strip()

# 只保留你的基因
ref = ref[ref['name2'].isin(target_genes)].copy()

# 只保留标准染色体
valid_chroms = [f'chr{i}' for i in range(1, 23)] + ['chrX', 'chrY']
ref = ref[ref['chrom'].isin(valid_chroms)].copy()

# 坐标转 int
ref['txStart'] = ref['txStart'].astype(int)
ref['txEnd'] = ref['txEnd'].astype(int)

# =========================
# 3. 每个基因选一个代表转录本
# 这里用“最长转录本”策略
# =========================
ref['txLen'] = ref['txEnd'] - ref['txStart']
ref = ref.sort_values(['name2', 'txLen'], ascending=[True, False])
gene_anno = ref.drop_duplicates(subset=['name2'], keep='first').copy()

# 保持和你的 gene list 顺序一致
gene_order = my_gene['gene'].tolist()
gene_anno = gene_anno.set_index('name2')
gene_anno = gene_anno.reindex(gene_order)
gene_anno = gene_anno.dropna(subset=['chrom', 'txStart', 'txEnd']).reset_index()
gene_anno = gene_anno.rename(columns={'index': 'gene', 'name2': 'gene'})

print(f"Genes in your list: {len(gene_order)}")
print(f"Genes matched in refGene: {len(gene_anno)}")

# =========================
# 4. 打开 mcool，选 25kb 分辨率
# 论文测试 5kb~25kb 都稳定；25kb 很合适
# =========================
res = 25000
c = cooler.Cooler(mcool_file + f"::/resolutions/{res}")

# 进一步过滤到 cooler 里实际存在的染色体
gene_anno = gene_anno[gene_anno['chrom'].isin(c.chromnames)].copy()
gene_anno = gene_anno.reset_index(drop=True)

# gene -> bin 区间（按 gene body）
gene_anno['start_bin'] = gene_anno['txStart'] // res
gene_anno['end_bin'] = (gene_anno['txEnd'] - 1) // res

print(f"Genes kept after chromosome filtering: {len(gene_anno)}")

# =========================
# 5. 用二维前缀和快速计算 gene-gene contact
#    C[i, j] = 基因i所在bin块 与 基因j所在bin块 的Hi-C总和
# =========================
n = len(gene_anno)
C = np.zeros((n, n), dtype=np.float32)

def block_sum(prefix, r1, r2, c1, c2):
    """
    在二维前缀和矩阵上求子矩阵和，区间均为闭区间
    """
    total = prefix[r2, c2]
    if r1 > 0:
        total -= prefix[r1 - 1, c2]
    if c1 > 0:
        total -= prefix[r2, c1 - 1]
    if r1 > 0 and c1 > 0:
        total += prefix[r1 - 1, c1 - 1]
    return total

for chrom in gene_anno['chrom'].unique():
    idx = gene_anno.index[gene_anno['chrom'] == chrom].tolist()
    if len(idx) == 0:
        continue

    chrom_size = int(c.chromsizes[chrom])
    region = f"{chrom}:0-{chrom_size}"

    print(f"Processing {chrom}, genes={len(idx)} ...")

    # 读该染色体的 cis Hi-C 矩阵
    mat = c.matrix(balance=True).fetch(region)
    mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    # 二维前缀和
    prefix = mat.cumsum(axis=0).cumsum(axis=1)

    max_bin = mat.shape[0] - 1

    # 提取该染色体基因的 bin 范围
    sub = gene_anno.loc[idx, ['start_bin', 'end_bin']].copy()
    sub['start_bin'] = sub['start_bin'].clip(lower=0, upper=max_bin)
    sub['end_bin'] = sub['end_bin'].clip(lower=0, upper=max_bin)

    # 对称填充
    for a_pos, i in enumerate(idx):
        s1 = int(sub.iloc[a_pos]['start_bin'])
        e1 = int(sub.iloc[a_pos]['end_bin'])
        if s1 > e1:
            continue

        for b_pos in range(a_pos, len(idx)):
            j = idx[b_pos]
            s2 = int(sub.iloc[b_pos]['start_bin'])
            e2 = int(sub.iloc[b_pos]['end_bin'])
            if s2 > e2:
                continue

            val = block_sum(prefix, s1, e1, s2, e2)
            C[i, j] = val
            C[j, i] = val

# =========================
# 6. SVD 压到 5 维
# 论文最终就是 5 维 condensed Hi-C features
# =========================
svd = TruncatedSVD(n_components=5, random_state=42)
hic_5d = svd.fit_transform(C)

# 可选：标准化，便于和别的 omics 特征拼接
scaler = StandardScaler()
hic_5d_scaled = scaler.fit_transform(hic_5d)

# =========================
# 7. 保存结果
# =========================
feat_df = pd.DataFrame(hic_5d_scaled, columns=[f'HiC_{i+1}' for i in range(5)])
feat_df.insert(0, 'gene', gene_anno['gene'].values)

feat_df.to_csv(r'Data\chroms\my_gene_hic_5d_features.csv', index=False)
np.save(r'Data\chroms\my_gene_contact_matrix.npy', C)

print("Done.")
print("Feature shape:", feat_df.shape)
print("Saved to: Data\\chroms\\my_gene_hic_5d_features.csv")
