import numpy as np
from Bio import PDB, SeqIO
from scipy import stats
import warnings
import math
import json
import io
from collections import Counter
import os

# 忽略 Biopython 的 PDB 解析警告
warnings.filterwarnings('ignore')


class ConservationCalculator:
    """
    保守性计算（CONSTRUCT-like 序列层）：
    - 输入：MSA (AF3 json / .a3m / .fasta)
    - 输出：每个位点的“替换率” (site-specific rate)，数值越小表示越保守
      （用 Shannon entropy 或其 Z-score 近似 Rate4Site）
    """

    def __init__(self, msa_file, query_len, min_coverage=0.3):
        """
        :param msa_file: MSA 文件路径 (.json / .a3m / .fasta)
        :param query_len: query 序列长度（通常等于结构中的标准 AA 残基数）
        :param min_coverage: 每列最小非 gap 覆盖比例，低于此值视为低置信列
        """
        self.msa_file = msa_file
        self.query_len = query_len
        self.min_coverage = min_coverage
        self.msa_matrix = self._load_msa()
        self._report_msa_stats()

    # ---------- MSA 读取部分 ----------

    def _load_msa(self):
        """智能加载 MSA: 优先支持 .a3m/.fasta，也兼容 AF3 data.json"""
        if self.msa_file.endswith('.json'):
            return self._load_from_json()
        else:
            return self._load_from_fasta()

    def _load_from_json(self):
        """
        适配 AF3 的 data.json 结构：
        root -> sequences list -> protein -> unpairedMsa (A3M 字符串)
        """
        print(f"[Info] 读取 AF3 JSON 文件: {self.msa_file}")
        with open(self.msa_file, 'r') as f:
            data = json.load(f)

        msa_sequences = []
        found_msa = False

        if 'sequences' in data:
            for seq_info in data['sequences']:
                if 'protein' in seq_info:
                    prot_data = seq_info['protein']
                    if 'unpairedMsa' in prot_data:
                        msa_content = prot_data['unpairedMsa']
                        print("[Info] 找到 'unpairedMsa' 字段，解析 A3M 字符串...")

                        with io.StringIO(msa_content) as handle:
                            for record in SeqIO.parse(handle, "fasta"):
                                seq_str = str(record.seq).upper()
                                # A3M 清洗：保留大写 + '-'
                                clean_seq = "".join(
                                    [c for c in seq_str if c.isupper() or c == '-']
                                )
                                if len(clean_seq) == self.query_len:
                                    msa_sequences.append([ord(c) for c in clean_seq])
                        found_msa = True
                        break  # 只处理第一条链

        # 兼容 raw feature JSON（根键为 'msa'）
        if not found_msa and 'msa' in data:
            print("[Info] JSON 中找到 'msa' 数组，直接使用。")
            arr = np.array(data['msa'], dtype=np.int16)
            if arr.shape[1] != self.query_len:
                print(f"[Warning] msa 长度({arr.shape[1]}) != query_len({self.query_len})，将截断。")
                min_len = min(arr.shape[1], self.query_len)
                arr = arr[:, :min_len]
            return arr

        if not msa_sequences:
            available_keys = list(data.keys())
            extra = (data['sequences'][0].keys()
                     if 'sequences' in data and len(data['sequences']) > 0 else [])
            raise ValueError(
                f"JSON 解析失败：未在 sequences->protein 中找到 'unpairedMsa'。\n"
                f"根键: {available_keys}, sequences[0] 的键: {list(extra)}"
            )

        arr = np.array(msa_sequences, dtype=np.int16)
        print(f"[Info] AF3 MSA 载入成功，矩阵维度: {arr.shape}")
        return arr

    def _load_from_fasta(self):
        """从 FASTA/A3M 文件加载（专门为 UniRef50 .a3m 定制）"""
        print(f"[Info] 读取 FASTA/A3M: {self.msa_file}")
        seqs = []
        with open(self.msa_file, "r") as f:
            for record in SeqIO.parse(f, "fasta"):
                seq_str = str(record.seq).upper()
                # 对 A3M：小写字母代表插入，删掉；保留大写和 '-'
                clean_seq = "".join([c for c in seq_str if c.isupper() or c == '-'])
                if len(clean_seq) == self.query_len:
                    seqs.append([ord(c) for c in clean_seq])
        arr = np.array(seqs, dtype=np.int16)
        print(f"[Info] FASTA/A3M MSA 载入成功，矩阵维度: {arr.shape}")
        return arr

    # ---------- MSA 质量报告（用于 sanity check） ----------

    def _report_msa_stats(self):
        if self.msa_matrix.size == 0:
            print("[Warning] MSA 为空，将返回全 0 得分。")
            return

        num_seqs, seq_len = self.msa_matrix.shape
        GAP_ASCII = 45  # '-'
        GAP_TOKEN = 21  # 兼容数字编码的 gap

        non_gap = (self.msa_matrix != GAP_ASCII) & (self.msa_matrix != GAP_TOKEN)
        col_cov = non_gap.sum(axis=0) / num_seqs
        mean_cov = float(col_cov.mean())
        frac_low_cov = float((col_cov < self.min_coverage).mean())

        print(f"[MSA-Stats] num_seqs = {num_seqs}, seq_len = {seq_len}")
        print(f"[MSA-Stats] 平均列覆盖率 = {mean_cov:.3f}, "
              f"低覆盖列比例(<{self.min_coverage}) = {frac_low_cov:.3f}")

        # 粗略 Neff：unique sequences 数
        try:
            uniq_rows = np.unique(self.msa_matrix, axis=0)
            neff = uniq_rows.shape[0]
        except Exception:
            neff = num_seqs
        print(f"[MSA-Stats] 粗略 Neff (unique sequences) = {neff}")

        # 抽样估计 pairwise identity（主要是 debug，用来确认不是全 Isolate）
        max_sample = min(200, num_seqs)
        if num_seqs >= 2:
            idx = np.random.choice(num_seqs, size=max_sample, replace=False)
            sub = self.msa_matrix[idx]
            identities = []
            for i in range(len(sub)):
                for j in range(i + 1, len(sub)):
                    a = sub[i]
                    b = sub[j]
                    mask = (a != GAP_ASCII) & (a != GAP_TOKEN) & (b != GAP_ASCII) & (b != GAP_TOKEN)
                    if mask.sum() == 0:
                        continue
                    same = (a[mask] == b[mask]).sum()
                    identities.append(same / mask.sum())
            if identities:
                avg_id = float(np.mean(identities))
                print(f"[MSA-Stats] 抽样平均 pairwise identity ≈ {avg_id:.3f}")
            else:
                print("[MSA-Stats] 无法计算 pairwise identity（可能全是 gap）")
        else:
            print("[MSA-Stats] 仅有 1 条序列，无法估计 pairwise identity。")

    # ---------- 熵 → 近似 Rate4Site ----------

    def calculate_site_rates(self):
        """
        计算每个位点的“替换率” (site-specific rate)，数值越小越保守：
        - 先计算 Shannon entropy
        - 再做 Z-score 标准化（可选），保持“越小越保守”的方向
        """
        if self.msa_matrix.size == 0:
            return np.zeros(self.query_len, dtype=np.float32)

        num_seqs, seq_len = self.msa_matrix.shape
        rates = np.zeros(seq_len, dtype=np.float32)

        GAP_ASCII = 45
        GAP_TOKEN = 21

        for col_idx in range(seq_len):
            column = self.msa_matrix[:, col_idx]
            valid_mask = (column != GAP_ASCII) & (column != GAP_TOKEN)
            valid_residues = column[valid_mask]

            coverage = valid_residues.size / num_seqs
            if coverage < self.min_coverage or valid_residues.size == 0:
                # 覆盖太低或全 gap：设为 0（后续会被 Z-score 拉到中间）
                rates[col_idx] = 0.0
                continue

            counts = Counter(valid_residues.tolist())
            total = float(valid_residues.size)
            entropy = 0.0
            for count in counts.values():
                p = count / total
                entropy -= p * math.log2(p)

            # 这里先用 entropy 作为 raw rate（越大越不保守）
            rates[col_idx] = entropy

        # 对 rates 做 Z-score，便于不同蛋白之间比较
        mean = float(np.mean(rates))
        std = float(np.std(rates))
        if std == 0.0:
            print("[Warning] 所有位点的 entropy 完全相同，返回原始 entropy。")
            return rates.astype(np.float32)

        z_rates = (rates - mean) / std  # 越小越保守
        return z_rates.astype(np.float32)


class ConstructAlgo:
    """
    CONSTRUCT-like 结构层算法：
    - 输入：结构 (PDB/mmCIF) + site-specific rate（来自 MSA）
    - 输出：一组 hotspot 残基索引（0-based，对应结构中的标准 AA 残基顺序）
    """

    def __init__(self, structure_file, msa_file):
        # 1. 解析结构
        if structure_file.endswith('.cif'):
            self.parser = PDB.MMCIFParser(QUIET=True)
        else:
            self.parser = PDB.PDBParser(QUIET=True)

        self.structure = self.parser.get_structure("prot", structure_file)
        self.residues = self._get_standard_residues()
        self.n_res = len(self.residues)
        print(f"[Info] 结构加载成功，标准 AA 残基数: {self.n_res}")

        if self.n_res == 0:
            raise ValueError("结构中未找到任何标准氨基酸残基。")

        # 2. 从 MSA 计算 site rates（替换率，越小越保守）
        self.calc = ConservationCalculator(msa_file, self.n_res)
        self.site_rates = self.calc.calculate_site_rates()  # shape (L,)

        # 3. 提取每个残基的坐标（侧链质心 / CA）
        self.coords = self._get_center_of_mass()

        # 4. 防御性长度对齐（一般 AF2 PDB 和 MSA 长度一致，这里是双保险）
        min_len = min(len(self.site_rates), len(self.coords))
        if min_len < len(self.coords):
            print(f"[Warning] MSA 长度({len(self.site_rates)}) < 结构长度({len(self.coords)})，截断结构和残基列表到 {min_len}。")
            self.coords = self.coords[:min_len]
            self.residues = self.residues[:min_len]
        if min_len < len(self.site_rates):
            print(f"[Warning] 结构长度({len(self.coords)}) < MSA 长度({len(self.site_rates)})，截断 site_rates 到 {min_len}。")
            self.site_rates = self.site_rates[:min_len]

        self.n = min_len

    def _get_standard_residues(self):
        """提取所有标准氨基酸残基（不区分链）"""
        res_list = []
        for model in self.structure:
            for chain in model:
                for res in chain:
                    if PDB.is_aa(res, standard=True):
                        res_list.append(res)
        return res_list

    def _get_center_of_mass(self):
        """计算每个残基的侧链质心（无侧链时用 CA）"""
        coords = []
        for res in self.residues:
            atoms = [a for a in res if a.name not in ['N', 'C', 'O', 'CA']]
            if not atoms:
                atoms = [a for a in res if a.name == 'CA']
            if atoms:
                coords.append(np.mean([a.get_coord() for a in atoms], axis=0))
            else:
                coords.append(np.array([0., 0., 0.], dtype=np.float32))
        return np.array(coords, dtype=np.float32)

    def run(self, return_scores=False):
        """
        执行简化版 CONSTRUCT：
        - 用 Shannon entropy 的 Z-score 近似 site-specific rate（越小越保守）
        - 在 3D 结构上做距离加权平滑，得到 spatial rate
        - 扫描窗口半径 r = 1 .. 20 Å
        - 每个 r 下：
          * 按 spatial rate 取最保守的前 10% 位点
          * 求其几何中心 anchor
          * 用 t-test 比较这些位点到 anchor 的距离 vs 其他位点
        - 选择 -log10(p) 最大的半径及对应 Top10% 作为 hotspot 集合

        :param return_scores: 如果为 True，额外返回 best_radius, spatial_rates(best) 和 raw site_rates
        :return: hotspot_indices (list)，如果 return_scores=True，返回 (hotspots, best_radius, best_spatial_rates, site_rates)
        """
        n = self.n
        if n == 0:
            print("[Error] 无有效残基，无法运行 CONSTRUCT。")
            return [] if not return_scores else ([], None, None, None)

        print("[Info] 预计算残基间距离矩阵...")
        dist_mat = np.linalg.norm(self.coords[:, None, :] - self.coords[None, :, :], axis=-1)

        best_log_p = -np.inf
        best_radius = None
        best_hotspots = []
        best_spatial_rates = None

        print("[Info] 扫描空间窗口 (1–20 Å)...")
        eps = 1e-6

        for radius in range(1, 21):
            spatial_rates = np.zeros(n, dtype=np.float32)

            # 1. 对每个残基做距离加权平滑（越小越保守）
            for i in range(n):
                neighbors = np.where(dist_mat[i] <= radius)[0]
                if neighbors.size == 0:
                    spatial_rates[i] = self.site_rates[i]
                    continue

                dists = dist_mat[i, neighbors]
                # 距离越近权重越大：w = 1 / (d + eps)
                weights = 1.0 / (dists + eps)
                S_neighbors = self.site_rates[neighbors]
                spatial_rates[i] = np.sum(weights * S_neighbors) / np.sum(weights)

            # 2. 取最保守的前 10%：rate 越小越保守
            k = max(5, int(n * 0.10))
            top_idx = np.argsort(spatial_rates)[:k]  # 最小的 k 个

            # 3. anchor = 这些 hotspot 的几何中心
            anchor = np.mean(self.coords[top_idx], axis=0)
            dists_to_anchor = np.linalg.norm(self.coords - anchor, axis=1)

            group_hot = dists_to_anchor[top_idx]
            group_rest = np.delete(dists_to_anchor, top_idx)

            if group_rest.size == 0 or group_hot.size < 3:
                continue

            # 4. 双尾 t-test，再转成单尾检验：假设 hotspot 更靠近 anchor => mean_hot < mean_rest
            t, p_two = stats.ttest_ind(group_hot, group_rest, equal_var=False)
            if np.isnan(t) or np.isnan(p_two):
                continue

            if t < 0:
                p_one = p_two / 2.0
            else:
                p_one = 1.0 - p_two / 2.0

            if p_one <= 0:
                continue

            log_p = -math.log10(p_one)
            print(f"[Radius {radius:2d} Å] k={k}, "
                  f"mean_hot_dist={group_hot.mean():.2f}, "
                  f"mean_rest_dist={group_rest.mean():.2f}, "
                  f"-log10(p)={log_p:.2f}")

            if log_p > best_log_p:
                best_log_p = log_p
                best_radius = radius
                best_hotspots = top_idx
                best_spatial_rates = spatial_rates

        print("\n[Result] 最佳半径:", best_radius, " 对应 -log10(p) =", best_log_p)
        hotspots_list = list(best_hotspots)

        if return_scores:
            return hotspots_list, best_radius, best_spatial_rates, self.site_rates
        return hotspots_list


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="CONSTRUCT-like hotspot detection using UniRef MSA (.a3m) and AlphaFold PDB."
    )
    parser.add_argument(
        "--cif",
        required=True,
        help="Path to structure file (.pdb or .cif). "
             "例如: /home2/s439820/GVP_AMR/Antibiogram_PATRIC.MetaPrism.gene_protein_selected.pdb/A0A023UEV3.pdb"
    )
    parser.add_argument(
        "--msa",
        required=True,
        help="Path to MSA file (.a3m, .fasta or AF3 data.json). "
             "例如: /home2/s439820/GVP_AMR/msa_results/A0A023UEV3.uniref.a3m"
    )
    parser.add_argument(
        "--save_scores",
        action="store_true",
        help="如果指定，则额外保存 spatial_rates 和 site_rates 到 .npy 文件。"
    )
    args = parser.parse_args()

    struct_path = args.cif
    msa_path = args.msa

    if not os.path.exists(struct_path) or not os.path.exists(msa_path):
        print(f"Error: 文件不存在.\n  结构: {struct_path}\n  MSA: {msa_path}")
        exit(1)

    try:
        algo = ConstructAlgo(struct_path, msa_path)
        hotspots, best_radius, spatial_rates, site_rates = algo.run(return_scores=True)

        print("\n" + "=" * 40)
        print("CONSTRUCT-like 分析完成")
        print("=" * 40)
        print(f"热点残基索引 (0-based):\n{hotspots}")
        print(f"热点数量: {len(hotspots)}")
        if best_radius is not None:
            print(f"最佳窗口半径: {best_radius} Å")

        # 保存 hotspot mask
        base, ext = os.path.splitext(struct_path)
        hotspot_path = base + "_hotspots.npy"
        np.save(hotspot_path, np.array(hotspots, dtype=np.int32))
        print(f"\n[Success] 热点索引已保存至: {hotspot_path}")

        # 可选：保存分数（便于后续 GVP 中使用连续权重）
        if args.save_scores and spatial_rates is not None and site_rates is not None:
            spatial_path = base + "_spatial_rates.npy"
            raw_path = base + "_site_rates.npy"
            np.save(spatial_path, spatial_rates.astype(np.float32))
            np.save(raw_path, site_rates.astype(np.float32))
            print(f"[Success] spatial_rates 已保存至: {spatial_path}")
            print(f"[Success] site_rates 已保存至: {raw_path}")

        print("\n提示：后续在 GVP 模型里，你可以：")
        print("  - 直接加载 *_hotspots.npy，将这些 index 作为 binary mask；")
        print("  - 或者加载 *_spatial_rates.npy，将其转成连续权重，对口袋残基增强 attention。")

    except Exception as e:
        print(f"\n[Error] 运行失败: {e}")
        import traceback
        traceback.print_exc()
