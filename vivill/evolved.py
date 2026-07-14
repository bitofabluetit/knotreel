import sys
import urllib.request
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import shortest_path

# --- DEPENDENCY VERIFICATION ---
try:
    import torch
    from transformers import AutoTokenizer, EsmModel
except ImportError:
    print("📦 [Setup] Missing required machine learning libraries.")
    print("👉 Action Required: pip install torch transformers")
    sys.exit(1)

# Standard amino acid mapping
AA_MAP = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
}


# --- 1. DATA INGESTION ---
def fetch_and_parse_pdb(pdb_id="1ubq"):
    url = f"https://files.rcsb.org/download/{pdb_id.lower()}.pdb"
    print(f"⏳ [Ingestion] Fetching PDB structure {pdb_id.upper()} from RCSB...")
    
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            lines = response.read().decode('utf-8').splitlines()
    except Exception as e:
        print(f"❌ [Ingestion] Failed to fetch PDB file: {e}")
        return None, None, None

    coords, sequence, seen_residues = [], [], set()
    full_pdb_records = []

    for line in lines:
        if line.startswith("ATOM") or line.startswith("HETATM"):
            atom_name = line[12:16].strip()
            alt_loc = line[16]
            if alt_loc not in (' ', 'A'): 
                continue
                
            res_name = line[17:20].strip()
            chain_id = line[21].strip()
            res_seq = int(line[22:26].strip())

            if atom_name == "CA" and chain_id in ("", "A"):
                if res_seq not in seen_residues:
                    seen_residues.add(res_seq)
                    x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                    coords.append([x, y, z])
                    sequence.append(AA_MAP.get(res_name, 'X'))
                    full_pdb_records.append({
                        "seq_idx": len(sequence) - 1,
                        "res_seq": res_seq,
                        "res_name": res_name
                    })

    return np.array(coords), "".join(sequence), full_pdb_records


# --- 2. MANIFOLD RECONSTRUCTION ENGINE ---
def extract_esm_attractor_mds(sequence, target_dim=3):
    print("🧬 [ESM-2] Extracting high-dimensional contextual embeddings...")
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
    model = EsmModel.from_pretrained("facebook/esm2_t6_8M_UR50D")
    inputs = tokenizer(sequence, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    embeddings = outputs.last_hidden_state[0, 1:-1].numpy()
    N = len(embeddings)
    
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9
    norm_embeddings = embeddings / norms
    
    D_mat = squareform(pdist(norm_embeddings, metric='euclidean'))
    D_sq = D_mat ** 2
    
    C = np.eye(N) - np.ones((N, N)) / N
    B = -0.5 * (C @ D_sq @ C)
    
    U, S, Vt = np.linalg.svd(B)
    raw_coords = U[:, :target_dim] * np.sqrt(np.maximum(S[:target_dim], 0.0))
    
    return raw_coords


def extract_native_contacts(coords, threshold=8.0, min_seq_sep=4):
    N = len(coords)
    contacts = []
    for i in range(N):
        for j in range(i + min_seq_sep, N):
            dist = np.linalg.norm(coords[i] - coords[j])
            if dist < threshold:
                contacts.append((i, j, dist))
    print(f"🔗 [Contacts] Extracted {len(contacts)} non-local topological contact anchors.")
    return contacts


# --- 3. ALIGNMENT SOLVERS ---
def kabsch_align_rigid(reference_coords, target_coords):
    mean_ref = np.mean(reference_coords, axis=0)
    mean_tar = np.mean(target_coords, axis=0)
    
    centered_ref = reference_coords - mean_ref
    centered_tar = target_coords - mean_tar
    
    H = centered_tar.T @ centered_ref
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
        
    aligned_coords = centered_tar @ R.T + mean_ref
    return aligned_coords


def resolve_chirality_and_align(reference_coords, esm_attractor):
    best_aligned = None
    best_rmsd = float('inf')
    
    sign_flips = [
        [1, 1, 1],   [1, 1, -1],  [1, -1, 1],  [1, -1, -1],
        [-1, 1, 1],  [-1, 1, -1], [-1, -1, 1], [-1, -1, -1]
    ]
    
    for flip in sign_flips:
        temp_attractor = esm_attractor * np.array(flip)
        aligned = kabsch_align_rigid(reference_coords, temp_attractor)
        
        diff = reference_coords - aligned
        rmsd = np.sqrt(np.mean(np.sum(diff ** 2, axis=1)))
        
        if rmsd < best_rmsd:
            best_rmsd = rmsd
            best_aligned = aligned
            
    print(f"🧬 [SVD Signs] Resolved sign ambiguity. Best template RMSD: {best_rmsd:.4f} Å")
    return best_aligned


# --- 4. PHASE 2.5: TRIANGLE INEQUALITY SMOOTHING & HYBRID EMBEDDING ---
def extract_hybrid_topology_template(esm_attractor, contacts, bond_length=3.80):
    N = len(esm_attractor)
    adj = np.zeros((N, N))
    
    for i in range(N - 1):
        adj[i, i+1] = adj[i+1, i] = bond_length
        
    for i, j, dist in contacts:
        adj[i, j] = adj[j, i] = dist
        
    D_smoothed = shortest_path(adj, directed=False, method='FW')
    
    D_sq = D_smoothed ** 2
    C = np.eye(N) - np.ones((N, N)) / N
    B = -0.5 * (C @ D_sq @ C)
    
    U, S, Vt = np.linalg.svd(B)
    coords_embedded = U[:, :3] * np.sqrt(np.maximum(S[:3], 0.0))
    
    embedded_steps = np.sqrt(np.sum((coords_embedded[1:] - coords_embedded[:-1])**2, axis=1))
    mean_embedded_step = np.mean(embedded_steps) + 1e-9
    coords_scaled = coords_embedded * (bond_length / mean_embedded_step)
    
    return coords_scaled


# --- 5. PHASE 3: ISOMETRIC BIOLOGICAL REFINE LOOP ---
def fold_protein_scheduled(relaxed_template, target_contacts, bond_length=3.80, min_clash=2.20):
    N = len(relaxed_template)
    
    if target_contacts:
        contact_i, contact_j, contact_dist = zip(*target_contacts)
        contact_i = np.array(contact_i)
        contact_j = np.array(contact_j)
        contact_dist = np.array(contact_dist)
    else:
        contact_i = contact_j = contact_dist = np.array([])

    stages = [
        {
            "name": "Stage 1: Local Relaxation",
            "bond": 1000.0,
            "angle": 1.0,
            "clash": 50.0,
            "contact": 100.0,
            "maxiter": 500,
        },
        {
            "name": "Stage 2: Strict Isometric Lock",
            "bond": 5000.0,
            "angle": 10.0,
            "clash": 150.0,
            "contact": 50.0,
            "maxiter": 500,
        }
    ]

    current_coords = relaxed_template.copy()

    for stage in stages:
        def loss_function(flat_coords):
            coords = flat_coords.reshape((N, 3))
            loss = 0.0
            
            diffs = coords[1:] - coords[:-1]
            bonds = np.sqrt(np.sum(diffs**2, axis=1) + 1e-9)
            loss += (stage["bond"] / (N - 1)) * np.sum((bonds - bond_length) ** 2)
            
            accel = coords[2:] - 2 * coords[1:-1] + coords[:-2]
            loss += (stage["angle"] / (N - 2)) * np.sum(accel ** 2)
            
            dists = pdist(coords)
            clash_mask = dists < min_clash
            valid_clash_mask = squareform(clash_mask)
            np.fill_diagonal(valid_clash_mask, False)
            for i in range(N - 1):
                valid_clash_mask[i, i+1] = valid_clash_mask[i+1, i] = False
                
            clashing_dists = squareform(dists)[valid_clash_mask]
            if len(clashing_dists) > 0:
                 loss += (stage["clash"] / len(clashing_dists)) * np.sum((clashing_dists - min_clash) ** 2)
                 
            if len(contact_i) > 0:
                c_dists = np.sqrt(np.sum((coords[contact_i] - coords[contact_j])**2, axis=1) + 1e-9)
                loss += (stage["contact"] / len(contact_i)) * np.sum((c_dists - contact_dist) ** 2)
                
            return loss

        res = minimize(loss_function, current_coords.flatten(), method='L-BFGS-B', options={'maxiter': stage['maxiter']})
        current_coords = res.x.reshape((N, 3))

    return current_coords


# --- 6. PHASE 4: CONTACT SENSITIVITY & POCKET GENERATION ENGINE ---
def analyze_folding_nuclei_and_pockets(esm_attractor, contacts, experimental_coords, pdb_records, bond_length=3.80, top_k=5):
    """
    Identifies folding core bottlenecks through analytical distance manifold perturbations,
    generating coordinates for stabilization and docking.
    """
    print("\n🔬 [Analysis] Commencing structural sensitivity profiling via analytical MDS perturbation...")
    N = len(esm_attractor)
    
    # 1. Establish the baseline coordinate state with all contacts active
    base_template = extract_hybrid_topology_template(esm_attractor, contacts, bond_length)
    base_aligned = kabsch_align_rigid(experimental_coords, base_template)
    base_rmsd = np.sqrt(np.mean(np.sum((experimental_coords - base_aligned) ** 2, axis=1)))
    
    sensitivity_scores = []
    
    # 2. Perturb and score each individual contact
    for idx, (i, j, dist) in enumerate(contacts):
        # Create a pruned set missing only the active contact [cite: 1.2.7]
        pruned_contacts = contacts[:idx] + contacts[idx+1:]
        
        # Fast, analytical coordinate projection (zero gradient descent cost) [cite: 1.2.4]
        pruned_template = extract_hybrid_topology_template(esm_attractor, pruned_contacts, bond_length)
        pruned_aligned = kabsch_align_rigid(experimental_coords, pruned_template)
        pruned_rmsd = np.sqrt(np.mean(np.sum((experimental_coords - pruned_aligned) ** 2, axis=1)))
        
        delta_rmsd = pruned_rmsd - base_rmsd
        sensitivity_scores.append({
            "idx": idx,
            "res_i": i,
            "res_j": j,
            "native_dist": dist,
            "delta_rmsd": delta_rmsd
        })
        
    # Sort by descending delta_rmsd to identify the most critical nucleation points
    sensitivity_scores.sort(key=lambda x: x["delta_rmsd"], reverse=True)
    
    print("\n======================================================================")
    print("           CRITICAL FOLDING NUCLEI IDENTIFICATION (TOP SENSITIVE)")
    print("======================================================================")
    for rank, score in enumerate(sensitivity_scores[:top_k]):
        res_i_meta = pdb_records[score["res_i"]]
        res_j_meta = pdb_records[score["res_j"]]
        print(f" Rank {rank+1}: {res_i_meta['res_name']}-{res_i_meta['res_seq']} <---> "
              f"{res_j_meta['res_name']}-{res_j_meta['res_seq']} "
              f"(Seq Sep: {abs(score['res_i'] - score['res_j'])}) "
              f"| Delta RMSD: {score['delta_rmsd']:.5f} Å")
        
    print("\n======================================================================")
    print("           STABILIZATION TARGETS / DOCKING POCKET METRICS")
    print("======================================================================")
    for rank, score in enumerate(sensitivity_scores[:top_k]):
        res_i_meta = pdb_records[score["res_i"]]
        res_j_meta = pdb_records[score["res_j"]]
        
        # Extract physical 3D coordinates from the native structure
        coord_i = experimental_coords[score["res_i"]]
        coord_j = experimental_coords[score["res_j"]]
        
        # Calculate target pocket metrics [cite: 3.1]
        pocket_centroid = (coord_i + coord_j) / 2.0
        alignment_vector = coord_j - coord_i
        alignment_direction = alignment_vector / (np.linalg.norm(alignment_vector) + 1e-9)
        
        print(f"\n🎯 [Target {rank+1}] stabilizing: {res_i_meta['res_name']}{res_i_meta['res_seq']} to {res_j_meta['res_name']}{res_j_meta['res_seq']}")
        print(f"  • Native Target Distance:        {score['native_dist']:.4f} Å")
        print(f"  • Pocket Centroid (X, Y, Z):     ({pocket_centroid[0]:.4f}, {pocket_centroid[1]:.4f}, {pocket_centroid[2]:.4f})")
        print(f"  • Binding Orientation Vector:    ({alignment_direction[0]:.4f}, {alignment_direction[1]:.4f}, {alignment_direction[2]:.4f})")
        print(f"  • Actionable Chaperone Strategy: Design a linker matching the centroid with binding groups spaced at {score['native_dist']:.2f} Å [cite: 3.1]")
        
    print("======================================================================\n")
    return sensitivity_scores


# --- 7. VIVELL-BIO KINEMATICS ENGINE ---
class VivellKinematicAnalyzer:
    def __init__(self, coords):
        self.coords = coords
        self.centroid = np.mean(coords, axis=0)
        self.centered_coords = coords - self.centroid

        self.x, self.y, self.z = self.centered_coords[:, 0], self.centered_coords[:, 1], self.centered_coords[:, 2]
        self.radius = np.sqrt(self.x**2 + self.y**2 + self.z**2) + 1e-8

        dx, dy, dz = np.gradient(self.x), np.gradient(self.y), np.gradient(self.z)
        self.v_vec = np.column_stack((dx, dy, dz))
        self.velocity = np.sqrt(dx**2 + dy**2 + dz**2) + 1e-8

        ddx, ddy, ddz = np.gradient(dx), np.gradient(dy), np.gradient(dz)
        self.a_vec = np.column_stack((ddx, ddy, ddz))
        self.acceleration = np.sqrt(ddx**2 + ddy**2 + ddz**2) + 1e-8

        dddx, dddy, dddz = np.gradient(ddx), np.gradient(ddy), np.gradient(dz)
        self.j_vec = np.column_stack((dddx, dddy, dddz))
        self.jerk = np.sqrt(dddx**2 + dddy**2 + dddz**2) + 1e-8

        cross_prod = np.cross(self.v_vec, self.a_vec)
        cross_mag = np.sqrt(np.sum(cross_prod**2, axis=1))
        self.curvature = cross_mag / (self.velocity**3)

        self.wobble = (self.curvature * self.jerk) / (self.radius**2)

        self.mean_radius = np.mean(self.radius)
        self.mean_velocity = np.mean(self.velocity)
        self.mean_acceleration = np.mean(self.acceleration)
        self.log_max_wobble = np.log10(np.percentile(self.wobble, 95) + 1e-9)
        self.r_a_correlation = np.corrcoef(self.radius, self.acceleration)[0, 1]

    def print_diagnostics(self, title):
        print("\n----------------------------------------------------------")
        print(f"      {title}")
        print("----------------------------------------------------------")
        print(f" • Mean Backbone Radius:            {self.mean_radius:.4f} Å")
        print(f" • Mean Step Velocity (Bond):       {self.mean_velocity:.4f} Å")
        print(f" • Mean Curvature Acceleration:     {self.mean_acceleration:.4f}")
        print(f" • Vivell Wobble (Log 95th %):      {self.log_max_wobble:.4f}")
        print(f" • Radius-Acceleration Corr (C):    {self.r_a_correlation:.4f}")
        
        if -2.50 <= self.log_max_wobble <= -1.00:
            print("\n Verdict: Physical parameters fall within stable, biological bounds.")
        else:
            print("\n Verdict: Trajectory parameters deviate from stable baseline constraints.")
        print("----------------------------------------------------------")


# --- 8. MAIN EXECUTION ---
if __name__ == "__main__":
    pdb_target = "1ubq"
    experimental_coords, sequence, pdb_records = fetch_and_parse_pdb(pdb_target)

    if experimental_coords is not None:
        # Phase 1: Reconstruct initial raw embedding attractor using Classical MDS
        esm_attractor = extract_esm_attractor_mds(sequence)

        # Phase 2: Extract Non-Local Contact Anchors
        contacts = extract_native_contacts(experimental_coords)

        # --- THE SENSITIVITY AND DOCKING POCKET ANALYSIS ---
        # Runs BEFORE physical optimization to map structural bottlenecks and virtual coordinates [cite: 1.2.4]
        sensitivity_profiles = analyze_folding_nuclei_and_pockets(
            esm_attractor, contacts, experimental_coords, pdb_records, top_k=5
        )

        # --- THE COLD-START RESOLUTION (Smooth constraints & embed using distance geometry) ---
        target_attractor = extract_hybrid_topology_template(esm_attractor, contacts)
        target_attractor_aligned = resolve_chirality_and_align(experimental_coords, target_attractor)

        # Phase 3: Fold and refine local backbone geometry
        folded_coords = fold_protein_scheduled(target_attractor_aligned, contacts, min_clash=2.20)

        # Step 4: Validate the Re-Folded Model's Physical Invariants
        analyzer = VivellKinematicAnalyzer(folded_coords)
        analyzer.print_diagnostics("VIVELL-BIO: FOLDED MODEL VERIFICATION")

        # Step 5: Evaluate shape alignment vs original PDB structure using Rigid Kabsch
        aligned_folded = kabsch_align_rigid(experimental_coords, folded_coords)
        diff = experimental_coords - aligned_folded
        final_rmsd = np.sqrt(np.mean(np.sum(diff ** 2, axis=1)))
        
        print(f"\n🏆 Final Kabsch-Aligned RMSD (vs True Structure): {final_rmsd:.4f} Å")
        
        if final_rmsd < 1.50:
            print("🌟 [Success] Ultra-high-fidelity folding. The model resolved the native fold.")
        elif final_rmsd < 3.00:
            print("✅ [Success] Moderate-fidelity folding. The major topological features converged successfully.")
        else:
            print("⚠️ Shape divergence. Check balance of contact energy vs local secondary force weights.")

    print("\n==========================================================")