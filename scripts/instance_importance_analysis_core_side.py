"""
Predict capacities and visualize core/side and aggregated feature importances.
Integrates strict "microscopic" core feature extraction to ensure SHAP reference
baselines are chemically perfect, utilizing custom model loading and scaling.
"""

import itertools
import os
from functools import partial
from typing import Dict, List, Set, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import skfp
import yaml
from rdkit import Chem
from rdkit.Chem import Draw, EState, rdDepictor, rdMolDescriptors, rdmolops
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.EState import AtomTypes
from rdkit.Chem.EState.AtomTypes import _rawD
from skfp.fingerprints import ECFPFingerprint, EStateFingerprint, FunctionalGroupsFingerprint

estate_type_names = [name for name, smarts in _rawD]


# =============================================================================
# 1. CUSTOM DESCRIPTOR CLASS (Whole & Core Methods)
# =============================================================================
class CustomDescriptor:
    def __init__(self):
        self.desc = {
            "diameter": skfp.descriptors.diameter,
            "diameter_filtered": self.calculate_diameter_endings,
            "num_heteroatoms": rdMolDescriptors.CalcNumHeteroatoms,
            "num_rotatable_bonds": rdMolDescriptors.CalcNumRotatableBonds,
            "mol_wt": rdMolDescriptors.CalcExactMolWt,
            "o%": partial(self.calculate_percentage, idx=8),
            "n%": partial(self.calculate_percentage, idx=7),
            "c%": partial(self.calculate_percentage, idx=6),
        }

    @staticmethod
    def calculate_percentage(mol: Chem.Mol, idx: int) -> float:
        num_oxygen = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == idx)
        return num_oxygen / mol.GetNumAtoms()

    def calculate_diameter_endings(self, mol: Chem.Mol) -> int:
        regular_diameter = skfp.descriptors.diameter(mol)
        patterns = ["O=*", "C#N", "Cl-*", "Br-*", "[NH2]", "[HO]"]
        if mol is None:
            return 0

        rdDepictor.Compute2DCoords(mol)
        dist_matrix = rdmolops.GetDistanceMatrix(mol)
        max_dist = -1

        for pattern_str in patterns:
            pattern = Chem.MolFromSmarts(pattern_str)
            matches = mol.GetSubstructMatches(pattern)
            if len(matches) < 2:
                continue

            for m1, m2 in itertools.combinations(matches, 2):
                heavy1 = [i for i in m1 if mol.GetAtomWithIdx(i).GetAtomicNum() > 1]
                heavy2 = [i for i in m2 if mol.GetAtomWithIdx(i).GetAtomicNum() > 1]
                for i in heavy1:
                    for j in heavy2:
                        d = dist_matrix[i][j]
                        if d > max_dist:
                            max_dist = d

        return max_dist if max_dist >= 0 else regular_diameter

    def fit_transform(self, smiles: list) -> np.ndarray:
        descriptors = np.zeros((len(smiles), len(self.desc.keys())))
        for i, s in enumerate(smiles):
            mol = Chem.MolFromSmiles(s)
            if mol is None:
                continue
            for j, (name, func) in enumerate(self.desc.items()):
                descriptors[i, j] = func(mol)
        return descriptors

    def calculate_core_descriptors(self, mol: Chem.Mol, core_match: set) -> list:
        if not mol or not core_match:
            return [0] * len(self.desc)

        core_set = set(core_match)
        num_core_atoms = len(core_set)

        try:
            dist_matrix = rdmolops.GetDistanceMatrix(mol)
            core_dists = [dist_matrix[i][j] for i, j in itertools.combinations(core_match, 2)]
            regular_diameter = max(core_dists) if core_dists else 0
        except Exception:
            regular_diameter = 0

        patterns = ["O=*", "C#N", "Cl-*", "Br-*", "[NH2]", "[HO]"]
        max_dist = -1
        for pattern_str in patterns:
            pattern = Chem.MolFromSmarts(pattern_str)
            matches = mol.GetSubstructMatches(pattern)
            core_pattern_matches = [m for m in matches if set(m).issubset(core_set)]
            if len(core_pattern_matches) < 2:
                continue

            for m1, m2 in itertools.combinations(core_pattern_matches, 2):
                heavy1 = [i for i in m1 if mol.GetAtomWithIdx(i).GetAtomicNum() > 1]
                heavy2 = [i for i in m2 if mol.GetAtomWithIdx(i).GetAtomicNum() > 1]
                for i in heavy1:
                    for j in heavy2:
                        d = dist_matrix[i][j]
                        if d > max_dist:
                            max_dist = d

        diameter_filtered = max_dist if max_dist >= 0 else regular_diameter
        num_heteroatoms = sum(1 for idx in core_set if mol.GetAtomWithIdx(idx).GetAtomicNum() not in (1, 6))

        rot_pattern = Chem.MolFromSmarts("[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]")
        rot_matches = mol.GetSubstructMatches(rot_pattern)
        num_rotatable_bonds = sum(1 for m in rot_matches if set(m).issubset(core_set))

        mol_wt = sum(mol.GetAtomWithIdx(idx).GetMass() for idx in core_set)
        o_pct = sum(1 for idx in core_set if mol.GetAtomWithIdx(idx).GetAtomicNum() == 8) / num_core_atoms
        n_pct = sum(1 for idx in core_set if mol.GetAtomWithIdx(idx).GetAtomicNum() == 7) / num_core_atoms
        c_pct = sum(1 for idx in core_set if mol.GetAtomWithIdx(idx).GetAtomicNum() == 6) / num_core_atoms

        return [regular_diameter, diameter_filtered, num_heteroatoms, num_rotatable_bonds, mol_wt, o_pct, n_pct, c_pct]

    def get_feature_names_out(self) -> list:
        return list(self.desc.keys())


# =============================================================================
# 2. MICROSCOPIC CORE EXTRACTION UTILITIES
# =============================================================================
def get_core_ecfp_counts(mol: Chem.Mol, core_match: set, radius: int = 2, nBits: int = 2048, return_bit_info=False):
    bit_info = {}
    rdMolDescriptors.GetHashedMorganFingerprint(mol, radius=radius, nBits=nBits, fromAtoms=list(core_match), bitInfo=bit_info)

    filtered_counts = np.zeros(nBits, dtype=int)
    filtered_bit_info = {}

    for bit_id, occurrences in bit_info.items():
        valid_occurrences = []
        for center_idx, bit_radius in occurrences:
            env_atoms = {center_idx}
            if bit_radius > 0:
                env_bonds = Chem.FindAtomEnvironmentOfRadiusN(mol, bit_radius, center_idx)
                for bond_idx in env_bonds:
                    bond = mol.GetBondWithIdx(bond_idx)
                    env_atoms.add(bond.GetBeginAtomIdx())
                    env_atoms.add(bond.GetEndAtomIdx())

            if env_atoms.issubset(core_match):
                filtered_counts[bit_id] += 1
                valid_occurrences.append((center_idx, bit_radius))

        if valid_occurrences:
            filtered_bit_info[bit_id] = tuple(valid_occurrences)

    if return_bit_info:
        return filtered_counts, filtered_bit_info
    return filtered_counts


def get_core_estate_sums(
    whole_mol: Chem.Mol,
    core_mol: Chem.Mol,
    whole_core_match: tuple,
    core_mol_match: tuple,
) -> np.ndarray:
    """
    1. Independently computes EState indices and EState groups for `core_mol`.
    2. Compares each core atom's group in `core_mol` against its group in `whole_mol`.
    3. If the atom is assigned to a different group in `whole_mol` (e.g., aaCH -> aasC
       at attachment points), reassigns its core EState value to the `whole_mol` group.
    """
    if AtomTypes.esPatterns is None:
        AtomTypes.BuildPatts()

    # 1. Independently compute EState indices and atom groups for the core molecule
    es_core = EState.EStateIndices(core_mol)
    types_core = AtomTypes.TypeAtoms(core_mol)

    # 2. Compute atom groups for the whole molecule
    types_whole = AtomTypes.TypeAtoms(whole_mol)

    name_to_bin = {name: i for i, (name, _) in enumerate(AtomTypes.esPatterns)}
    estate_sums = np.zeros(len(AtomTypes.esPatterns), dtype=np.float64)

    # 3. Compare core vs. whole groups for every matched atom in the core
    for whole_idx, core_idx in zip(whole_core_match, core_mol_match):
        core_group = types_core[core_idx]
        whole_group = types_whole[whole_idx]

        # If the atom's group changed in the whole molecule, reassign to whole_group
        if core_group != whole_group:
            assigned_groups = []
        else:
            assigned_groups = core_group
        # Add the independent core EState value to the assigned group bin(s)
        for t_name in assigned_groups:
            bin_idx = name_to_bin[t_name]
            estate_sums[bin_idx] += es_core[core_idx]

    return estate_sums


# def get_core_estate_sums(mol: Chem.Mol, core_match: set = None) -> np.ndarray:
#     """Calculates the 79-bin EState sums using RDKit's official TypeAtoms logic."""
#     if AtomTypes.esPatterns is None:
#         AtomTypes.BuildPatts()
#
#     es_indices = EState.EStateIndices(mol)
#     atom_types = AtomTypes.TypeAtoms(mol)
#
#     name_to_bin = {name: i for i, (name, _) in enumerate(AtomTypes.esPatterns)}
#     estate_sums = np.zeros(len(AtomTypes.esPatterns), dtype=np.float64)
#
#     for atom_idx, types_for_atom in enumerate(atom_types):
#         if core_match is not None and atom_idx not in core_match:
#             continue
#
#         for t_name in types_for_atom:
#             bin_idx = name_to_bin[t_name]
#             estate_sums[bin_idx] += es_indices[atom_idx]
#
#     return estate_sums


def get_core_functional_groups_counts(mol: Chem.Mol, core_match: set) -> np.ndarray:
    # Relying on global intersection to filter side chains
    fg = FunctionalGroupsFingerprint(count=True)
    return fg.transform([Chem.MolToSmiles(mol)])[0]


# =============================================================================
# 3. FEATURE GENERATION & SHAP BASELINE
# =============================================================================
def generate_features(smiles: str, config: dict) -> pd.DataFrame:
    ecfp_config = config.get("ecfp", {})
    estate_config = config.get("estate", {})
    fg_config = config.get("functional_groups", {})

    ecfp = ECFPFingerprint(fp_size=ecfp_config.get("size", 2048), radius=ecfp_config.get("radius", 2), include_chirality=True, count=ecfp_config.get("count", True))
    estate = EStateFingerprint(variant=estate_config.get("variant", "sum"))
    functional_groups = FunctionalGroupsFingerprint(count=fg_config.get("count", True))
    descriptor = CustomDescriptor()

    smiles_list = [smiles]
    ecfp_features = ecfp.transform(smiles_list)
    estate_features = estate.transform(smiles_list)
    descriptor_features = descriptor.fit_transform(smiles_list)
    fg_features = functional_groups.transform(smiles_list)

    ecfp_feature_names = [f"ecfp_{i}" for i in range(ecfp_features.shape[1])]
    estate_feature_names = [f"estate_{i}" for i in range(estate_features.shape[1])]
    descriptor_features_names = descriptor.get_feature_names_out()
    fg_feature_names = [f"functional_groups_{i}" for i in range(fg_features.shape[1])]

    feature_names = ecfp_feature_names + estate_feature_names + descriptor_features_names + fg_feature_names
    features = np.concatenate([ecfp_features, estate_features, descriptor_features, fg_features], axis=1)

    df = pd.DataFrame(features, columns=feature_names)
    features_to_keep = config.get("features_to_keep", [])
    if features_to_keep:
        df = df[features_to_keep]
    return df


def generate_core_reference_features(smiles_list: list, core_smarts: str, config: dict) -> pd.DataFrame:
    ecfp_config = config.get("ecfp", {})
    n_bits = ecfp_config.get("size", 1024)
    radius = ecfp_config.get("radius", 2)

    core_pattern = Chem.MolFromSmarts(core_smarts)
    descriptor = CustomDescriptor()
    all_core_features = []
    feature_names = []

    core_mol = Chem.MolFromSmiles(core_smarts)
    core_mol_match = core_mol.GetSubstructMatch(core_pattern)

    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        matches = mol.GetSubstructMatches(core_pattern)
        if not matches:
            continue
        core_match = set(matches[0])

        ecfp_array = get_core_ecfp_counts(mol, core_match, radius, n_bits)
        estate_array = get_core_estate_sums(mol, core_mol, matches[0], core_mol_match)
        # get_core_estate_sums(Chem.MolFromSmiles(core_smarts), core_match=None)
        desc_list = descriptor.calculate_core_descriptors(mol, core_match)
        fg_array = get_core_functional_groups_counts(mol, core_match)

        combined = np.concatenate([ecfp_array, estate_array, desc_list, fg_array])
        ecfp_feature_names = [f"ecfp_{i}" for i in range(len(ecfp_array))]
        estate_feature_names = [f"estate_{i}" for i in range(len(estate_array))]
        descriptor_features_names = descriptor.get_feature_names_out()
        fg_feature_names = [f"functional_groups_{i}" for i in range(len(fg_array))]

        feature_names = ecfp_feature_names + estate_feature_names + descriptor_features_names + fg_feature_names
        all_core_features.append(combined)

    baseline_array = np.min(all_core_features, axis=0).reshape(1, -1)

    dummy_df = generate_features(smiles_list[0], config)

    df = pd.DataFrame(baseline_array, columns=feature_names)
    df = df[list(dummy_df.columns)]
    return df


# =============================================================================
# 4. PREDICTION & MODEL IO
# =============================================================================
def load_model_and_scaler(config_path: str = "brr_best_model/config.yaml"):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    model = joblib.load(cfg["model"])
    scaler = joblib.load(cfg["scaler"])
    numerical_features = pd.read_csv(cfg["numerical_features"], header=None).iloc[:, 0].tolist()
    dataset_cfg = cfg["dataset"]
    if "features_to_keep" in dataset_cfg:
        dataset_cfg["features_to_keep"] = pd.read_csv(dataset_cfg["features_to_keep"], header=None).iloc[:, 0].tolist()
    return cfg, model, scaler, numerical_features, dataset_cfg


def predict_and_explain(smiles: str, model, scaler, numerical_features: List[str], dataset_cfg: dict, core_baseline_df: pd.DataFrame, explain_method="shap"):
    feats = generate_features(smiles, dataset_cfg)
    non_scaled = feats.copy()

    if numerical_features:
        feats[numerical_features] = scaler.transform(feats[numerical_features].values)

    y_pred = model.predict(feats.values)

    if explain_method == "coefficient" and hasattr(model, "coef_"):
        importance = pd.DataFrame({"feature": feats.columns.tolist(), "importance": model.coef_[0] * feats.values, "feature_values": non_scaled.iloc[0].values})
        return float(y_pred[0]), importance
    else:
        # Instead of recalculating via hardcoded train_smarts, we use the pure intersection baseline
        train_data = core_baseline_df.copy()

        # Apply your exact logic: ensure baseline feature counts never exceed the target molecule
        # for f in train_data.columns:
        #     if train_data[f].iloc[0] > 0 and non_scaled[f].iloc[0] == 0:
        #         train_data.loc[0, f] = 0

        # Scale the baseline using the loaded scaler
        if numerical_features:
            train_data[numerical_features] = scaler.transform(train_data[numerical_features].values)
        try:
            explainer = shap.LinearExplainer(model, train_data)
            shap_values = explainer.shap_values(feats.to_numpy())
        except Exception:
            explainer = shap.TreeExplainer(model, feature_perturbation="interventional", data=train_data)
            shap_values = explainer.shap_values(feats.to_numpy())

        y_pred_core = model.predict(train_data.values)

        shap_exp = shap.Explanation(values=shap_values, data=non_scaled.to_numpy(), feature_names=list(non_scaled.columns))
        return float(y_pred[0]), float(y_pred_core[0]), shap_exp


# =============================================================================
# 5. VISUALIZATION & UTILITIES
# =============================================================================
def compute_mcs_smarts(smiles_list: List[str], timeout: int = 10) -> Tuple[str, Dict[int, Tuple[int, ...]]]:
    # mols_with_core = [Chem.MolFromSmiles(s) for s in smiles_list]
    # mcs_res = rdFMCS.FindMCS(mols_with_core, timeout=timeout)
    mols = [Chem.MolFromSmiles(s) for s in smiles_list[:-1]]
    # smarts = mcs_res.smartsString or ""

    smarts = "N#Cc(cc1)ccc1-[n](cc1)c(cc2)c1[n]2-c(cc1)ccc1C#N"

    matches = {}
    if smarts:
        patt = Chem.MolFromSmarts(smarts)
        for i, m in enumerate(mols):
            ms = m.GetSubstructMatches(patt)
            matches[i] = ms[0] if ms else tuple()
    return smarts, matches


def make_rect(match):
    cx = match.group(1)
    cy = match.group(2)
    rx = float(match.group(3))
    ry = float(match.group(4))
    atom_class = match.group(5)
    style_str = match.group(6)

    width = rx * 1.5
    height = ry * 1.5
    x = float(cx) - (width / 2)
    y = float(cy) - (height / 2)
    return f"<rect x='{x}' y='{y}' width='{width}' height='{height}' rx='3' class='{atom_class}' style='{style_str}' />"


def replace_circles_with_rectangles(svg: str) -> str:
    import re

    pattern = r"<ellipse\s+cx='([^']+)'\s+cy='([^']+)'\s+rx='([^']+)'\s+ry='([^']+)'\s+class='([^']+)'\s+style='([^']+)'\s*/>"
    svg = re.sub(pattern, make_rect, svg)
    svg = re.sub(r"style='[^']+#([0-9A-Fa-f]{8})[^']*'", lambda m: m.group(0).replace(m.group(1), m.group(1)[:6]), svg)
    return svg


# def visualize_core_side(smiles: str, core_atoms: Set[int], annot: str = '', width: int = 400, height: int = 300) -> str:
#     mol = Chem.MolFromSmiles(smiles)
#     rdDepictor.Compute2DCoords(mol)
#     core_set = set(core_atoms)
#
#     atom_colors, bond_colors = {}, {}
#     highlight_atoms, highlight_bonds = list(core_set), []
#     core_color = (0.2, 0.6, 1.0, 0.4)
#
#     for a in core_set: atom_colors[a] = core_color
#
#     for bond in mol.GetBonds():
#         a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
#         if a1 in core_set and a2 in core_set:
#             highlight_bonds.append(bond.GetIdx())
#             bond_colors[bond.GetIdx()] = core_color
#
#     drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
#     drawer.drawOptions().prepareMolsBeforeDrawing = True
#     drawer.DrawMolecule(mol, highlightAtoms=highlight_atoms, highlightAtomColors=atom_colors,
#                         highlightBonds=highlight_bonds, highlightBondColors=bond_colors, legend=annot)
#     drawer.FinishDrawing()
#     return drawer.GetDrawingText()


def visualize_core_side(smiles: str, core_atoms: Set[int], annot: str = "", width: int = 400, height: int = 300, core_color=(0.204, 1, 0.941, 0.4), side_color=(1.0, 0.4, 0.4, 0.4)) -> str:
    mol = Chem.MolFromSmiles(smiles)
    rdDepictor.Compute2DCoords(mol)
    # n_atoms = mol.GetNumAtoms()
    core_set = set(core_atoms)
    # all_atoms = set(range(n_atoms))
    # side_set = all_atoms - core_set

    atom_colors = {}
    highlight_atoms = []
    highlight_bonds = []
    bond_colors = {}

    for a in core_set:
        atom_colors[a] = core_color
        highlight_atoms.append(a)
    # for a in side_set:
    #     atom_colors[a] = side_color
    #     highlight_atoms.append(a)

    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtomIdx()
        a2 = bond.GetEndAtomIdx()
        if a1 in highlight_atoms or a2 in highlight_atoms:
            highlight_bonds.append(bond.GetIdx())
            if a1 in core_set and a2 in core_set:
                bond_colors[bond.GetIdx()] = core_color
            # else:
            #     bond_colors[bond.GetIdx()] = (0.6, 0.2, 0.6, 0.4)

    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    drawer.drawOptions().prepareMolsBeforeDrawing = True
    drawer.DrawMolecule(mol, highlightAtoms=list(set(highlight_atoms)), highlightAtomColors=atom_colors, highlightBonds=list(set(highlight_bonds)), highlightBondColors=bond_colors, legend=annot)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    return replace_circles_with_rectangles(svg)


def validate_core_ecfp_extraction(mol: Chem.Mol, filtered_bit_info: dict, out_path: str):
    if not filtered_bit_info:
        return

    # Pick top 4 bits that were found
    bits_to_draw = list(filtered_bit_info.keys())[:4]
    tpls = [(mol, bit, filtered_bit_info) for bit in bits_to_draw]

    try:
        img = Draw.DrawMorganBits(tpls, molsPerRow=min(4, len(tpls)), legends=[f"Strict Core Bit {b}" for b in bits_to_draw])

        # Check if RDKit returned a raw SVG string or a PIL Image object
        if isinstance(img, str):
            # Change the file extension to .svg and write the text
            svg_path = out_path.replace(".png", ".svg")
            with open(svg_path, "w") as f:
                f.write(img)
        else:
            # It's a PIL Image object, use .save()
            img.save(out_path)

    except Exception as e:
        print(f"Skipping validation draw due to: {e}")


def validate_feature_differences(core_df: pd.DataFrame, whole_df: pd.DataFrame, out_path: str) -> pd.DataFrame:
    """
    Compares the 1-row core baseline DataFrame against the 1-row whole molecule DataFrame.
    Returns a DataFrame containing only the features that have different values, and saves it to a CSV.
    """
    # Extract the single row from both DataFrames
    core_row = core_df.iloc[0]
    whole_row = whole_df.iloc[0]

    differences = []

    for col in core_df.columns:
        val_core = core_row[col]
        val_whole = whole_row[col]

        # Use np.isclose to handle minor floating-point differences in percentages/weights
        if not np.isclose(val_core, val_whole, atol=1e-5):
            differences.append({"Feature": col, "Core_Value": val_core, "Whole_Molecule_Value": val_whole, "Difference (Whole - Core)": val_whole - val_core})

    diff_df = pd.DataFrame(differences)

    if not diff_df.empty:
        # Sort by the absolute difference so the biggest changes are at the top
        diff_df = diff_df.sort_values(by="Difference (Whole - Core)", key=abs, ascending=False)
        diff_df.to_csv(out_path, index=False)

    return diff_df


def plot_custom_shap_bar(shap_exp: shap.Explanation, whole_df: pd.DataFrame, core_df: pd.DataFrame, out_path: str, top_n: int = 10):
    """
    Plots a custom SHAP bar chart for the top N features.
    Feature names are black and right-aligned.
    Values and differences are gray and left-aligned directly next to the names.
    SHAP values are displayed at the ends of the bars.
    """
    if len(shap_exp.values.shape) > 1:
        shap_vals = shap_exp.values[0]
    else:
        shap_vals = shap_exp.values

    feature_names = whole_df.columns.tolist()

    non_zero_indices = np.where(~np.isclose(shap_vals, 0, atol=1e-8))[0]
    abs_shap_non_zero = np.abs(shap_vals[non_zero_indices])
    sorted_indices = non_zero_indices[np.argsort(abs_shap_non_zero)]
    top_n_actual = min(top_n, len(sorted_indices))

    # If a molecule somehow has absolutely 0 feature importance across the board, skip plotting
    if top_n_actual == 0:
        print(f"Skipping SHAP plot for {out_path} - no non-zero features found.")
        return

    top_indices = sorted_indices[-top_n_actual:]

    plot_shap_vals = []
    colors = []

    core_row = core_df.iloc[0]
    whole_row = whole_df.iloc[0]

    for idx in top_indices:
        s_val = shap_vals[idx]
        plot_shap_vals.append(s_val)
        # Standard SHAP colors: Crimson for positive, dodger blue for negative
        # colors.append("#ff0051" if s_val > 0 else "#008bfb")
        colors.append("#138c4a" if s_val > 0 else "#c72127")

    fig_height = 0.5 + (0.5 * top_n_actual)

    fig, ax = plt.subplots(figsize=(10, fig_height))
    plt.subplots_adjust(left=0.4)

    y_positions = range(top_n_actual)
    bars = ax.barh(y_positions, plot_shap_vals, color=colors)

    plt.xlabel("SHAP value (impact on model output)", fontsize=12)
    plt.axvline(0, color="black", linewidth=1, linestyle="-", zorder=0)

    ax.set_yticks(y_positions)
    ax.set_yticklabels([])
    ax.tick_params(axis="y", length=0)

    # Text anchor point (distance from the y-axis in axes fraction)
    # -0.22 ensures the gray text doesn't cross over the y-axis into the graph
    anchor_x = -0.33

    def fmt(n):
        return f"{n:.2f}".rstrip("0").rstrip(".")

    def format_feature_names(f: str):
        functional_group_mapping = [
            "aliphatic carboxylic acids",
            "aliphatic hydroxyl groups",
            "aliphatic hydroxyl groups excluding tert-OH",
            "N functional groups attached to aromatics",
            "Aromatic carboxylic acide",
            "aromatic nitrogens",
            "aromatic amines",
            "aromatic hydroxyl groups",
            "carboxylic acids COO",
            "carboxylic acids COO2",
            "carbonyl O",
            "carbonyl O, excluding COOH",
            "thiocarbonyl",
            "C(OH)CCN-Ctert-alkyl or  C(OH)CCNcyclic",
            "Imines",
            "Tertiary amines",
            "Secondary amines",
            "Primary amines",
            "hydroxylamine groups",
            "XCCNR groups",
            "tert-alicyclic amines (no heteroatoms, not quinine-like bridged N)",
            "H-pyrrole nitrogens",
            "thiol groups",
            "aldehydes",
            "alkyl carbamates (subject to hydrolysis)",
            "alkyl halides",
            "Allylic oxidation sites excluding steroid dienone",
            "amides",
            "amidine groups",
            "anilines",
            "aryl methyl sites for hydroxylation",
            "azide groups",
            "azo groups",
            "barbiturate groups",
            "Benzene rings",
            "benzodiazepines with no additional fused rings",
            "Bicyclic",
            "diazo groups",
            "dihydropyridines",
            "epoxide rings",
            "esters",
            "ether oxygens (including phenoxy)",
            "furan rings",
            "guanidine groups",
            "halogens",
            "hydrazine groups",
            "hydrazone groups",
            "imidazole rings",
            "imide groups",
            "isocyanates",
            "isothiocyanates",
            "ketones",
            "ketones excluding diaryl, a,b-unsat. dienones, heteroatom on Calpha",
            "beta lactams",
            "cyclic esters (lactones)",
            "methoxy groups -OCH3",
            "morpholine rings",
            "nitriles",
            "nitro groups",
            "nitro benzene ring substituents",
            "non-ortho nitro benzene ring substituents",
            "nitroso groups, excluding NO2",
            "oxazole rings",
            "oxime groups",
            "para-hydroxylation sites",
            "phenols",
            "phenolic OH excluding ortho intramolecular Hbond substituents",
            "phosphoric acid groups",
            "phosphoric ester groups",
            "piperdine rings",
            "piperzine rings",
            "primary amides",
            "primary sulfonamides",
            "pyridine rings",
            "quaternary nitrogens",
            "thioether",
            "sulfonamides",
            "sulfone groups",
            "terminal acetylenes",
            "tetrazole rings",
            "thiazole rings",
            "thiocyanates",
            "thiophene rings",
            "unbranched alkanes of at least 4 members (excludes halogenated alkanes)",
            "urea groups",
        ]
        from rdkit.Chem.EState.AtomTypes import _rawD

        estate_names_mapping = [name for name, smarts in _rawD]
        estate_names_mapping = {f"estate_{i}": n for i, n in enumerate(estate_names_mapping)}
        functional_group_mapping = {f"functional_groups_{i}": name for i, name in enumerate(functional_group_mapping)}
        desc = ["diameter", "diameter_filtered", "num_heteroatoms", "num_rotatable_bonds", "mol_wt", "o%", "n%", "c%"]
        if f in desc:
            dict_desc = ["Global diameter", "End-to-end group diameter", "Number of heteroatoms", "Number of rotatable bonds", "Mol Weight", "O%", "N%", "C%"]
            dict_desc = dict(zip(desc, dict_desc))
            return dict_desc[f]
        if f.startswith("estate_"):
            suffix = f.removeprefix("estate_")
            return estate_names_mapping[f]
            # return f'EState {suffix}'

        if f.startswith("ecfp_"):
            suffix = f.removeprefix("ecfp_")
            return f"ECFP {suffix}"

        if f.startswith("functional_groups_"):
            suffix = f.removeprefix("functional_groups_")
            if functional_group_mapping is None:
                return f"Functional group {suffix}"

            mapped_name = functional_group_mapping.get(suffix) or functional_group_mapping.get(f) or functional_group_mapping.get(int(suffix)) if suffix.isdigit() else None
            return mapped_name.capitalize() if mapped_name else f"Functional group {suffix}"

        return f.capitalize()

    for i, idx in enumerate(top_indices):
        feat_name = feature_names[idx]
        val = whole_row[feat_name]
        core_val = core_row[feat_name]
        diff = val - core_val

        feat_name_display = format_feature_names(feat_name)

        # Format the value string (e.g., " = 2 (0)" or " = 711.9 (+15)")
        if round(diff, 2) == 0:
            val_str = f" = {fmt(val)} (0)"
        else:
            sign = "+" if diff > 0 else ""
            val_str = f" = {fmt(val)} ({sign}{fmt(diff)})"

        # Draw Feature Name (Black, Right-aligned from anchor)
        ax.text(anchor_x, i, feat_name_display, ha="right", va="center", color="black", fontsize=12, transform=ax.get_yaxis_transform())

        # Draw Value & Difference (Gray, Left-aligned from anchor)
        ax.text(anchor_x, i, val_str, ha="left", va="center", color="#888888", fontsize=12, transform=ax.get_yaxis_transform())

    # --- ADD SHAP VALUES TO THE ENDS OF THE BARS ---
    x_span = max(plot_shap_vals) - min(plot_shap_vals)
    pad = x_span * 0.02 if x_span > 0 else 0.1

    for bar, s_val, color in zip(bars, plot_shap_vals, colors):
        if s_val > 0:
            ha = "left"
            x_pos = s_val + pad
            text_str = f"+{s_val:.2f}"
        else:
            ha = "right"
            x_pos = s_val - pad
            text_str = f"{s_val:.2f}"

        ax.text(x_pos, bar.get_y() + bar.get_height() / 2, text_str, va="center", ha=ha, color=color, fontsize=12)

    # Despine the plot for a cleaner look
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    # Adjust x-axis limits slightly to ensure the new text isn't cut off
    xlims = plt.xlim()
    plt.xlim(xlims[0] - (pad * 5), xlims[1] + (pad * 5))

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


# =============================================================================
# 6. MAIN EXECUTION PIPELINE
# =============================================================================
def main(smiles_list_all: List[str], output_dir: str = "outputs_brr", config_path: str = "brr_best_model/config.yaml"):
    os.makedirs(output_dir, exist_ok=True)

    # Load Model, Config, and Scalers
    print("Loading model and configuration...")
    cfg, model, scaler, numerical_features, dataset_cfg = load_model_and_scaler(config_path)

    print("Computing MCS...")
    smarts, matches = compute_mcs_smarts(smiles_list_all)
    print(f"Core SMARTS: {smarts}")

    print("Generating pure core reference baseline...")
    # smiles_list = smiles_list_all
    core_baseline_df = generate_core_reference_features(smiles_list_all, smarts, dataset_cfg)

    for i, smi in enumerate(smiles_list_all):
        outdir = os.path.join(output_dir, f"smiles_{i}")
        os.makedirs(outdir, exist_ok=True)
        mol = Chem.MolFromSmiles(smi)
        core_atoms = set(matches.get(i, ()))

        whole_features = generate_features(smi, dataset_cfg)
        diff_csv_path = os.path.join(outdir, "core_vs_whole_differences.csv")
        diff_df = validate_feature_differences(core_baseline_df, whole_features, diff_csv_path)
        print(f"Molecule {i}: Found {len(diff_df)} features differing between strict core baseline and whole molecule.")

        # --- Feature Generation, Scaling, Prediction & SHAP ---
        y, y_core, expl = predict_and_explain(smi, model, scaler, numerical_features, dataset_cfg, core_baseline_df, explain_method="shap")

        with open(os.path.join(outdir, "prediction.txt"), "w") as f:
            f.write(f"{y}\n{y_core}")

        shap_plot_path = os.path.join(outdir, "custom_shap_bar.svg")
        plot_custom_shap_bar(expl, whole_features, core_baseline_df, shap_plot_path, top_n=10)

        # --- Visualizations ---
        try:
            svg = visualize_core_side(smi, core_atoms, annot=f"pred: {y}")
            with open(os.path.join(outdir, "mol_core_side.svg"), "w") as f:
                f.write(svg)
        except Exception as e:
            print("Failed to draw core/side for", smi, e)

        if core_atoms:
            _, bit_info = get_core_ecfp_counts(mol, core_atoms, return_bit_info=True)
            validate_core_ecfp_extraction(mol, bit_info, os.path.join(outdir, "ecfp_validation.png"))

        print(f"Processed Molecule {i} - Outputs saved to {outdir}")


if __name__ == "__main__":
    smiles = pd.read_csv("../notebooks/lab_smiles.csv")["smiles"].tolist()
    print(len(smiles))
    config_file = "../notebooks/rf_best_model/config.yaml"
    main(smiles, output_dir="../notebooks/outputs_rf", config_path=config_file)
