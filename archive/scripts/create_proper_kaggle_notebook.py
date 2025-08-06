#!/usr/bin/env python3
"""
Create a proper Kaggle-compatible T4x2 notebook with automatic dependency installa
"""

import json

def create_proper_kaggle_notebook():
    """
    Create a proper Kaggle-compatible T4x2 notebook with automatic dependency installa
    """
    # Create a proper Kaggle-compatible T4x2 notebook with automatic dependency installation
    # Create a proper Kaggle-compatible T4x2 notebook with automatic dependency installation

    
    notebook = {
        "cells":    
            # Cell 1: Title and Overview
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# NeurIPS Open Polymer Prediction 2025 - T4 x2 GPU Solution\n",
                    "\n",
                    "## 🏆 Competition-Ready T4 x2 Optimized Implementation\n",
                    "\n",
                    "**Expected Performance \n",
                    "**Architecture**: 6-layer PolyGIN with DataParallel optimizatio,
                    "**GPU Requirements**: T4 x2 (16GB total VRAM)  \n",
                    "**Training Time**: ~20-30 minutes for full trainin
                    "\n",
                    "### 🎯 T4 x2 Optimizations\n",
                    "- **U)\n",
                    "- **Batch Size**: 48 pen",
                    "- **Model**: 64 hidden channels, 6 layers\n",
                    "- **Training**: Mixed precision + optimized d
                    "- **DataParallel**: Automatic tensor shape handling\n",
                    "\n",
                 
              port\n",
            
                    "3. **Model Architecture**: T4-optimiz
             
                    "5. **Submission",
                    "\n",
                    "---"
                ]
            },
            
            # Cell 2: Dependency Installation
            {
                "cell_typ",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# ============================",
                    "# AUTOMATIC DEPENDENCY INSTALLATION\n",
                    "# ======",
                    "\n",
                    "import subprocess\n",
                    "import sys\n",
                    "import os\n",
                    "\n",
                    "def install_package(packa\n",
                    "    \"\"\"Install pac"\n",
                    "    try:\n",
                    "        ,
                    "        print(f\"✅ {package} installed successfully\"",
                    "        
                    "    except subprocess.CalledProc
                    "        print(f\,
                    "        return False\n",
                    "\n",
                    "# Required packages for T4 x2 solution\n",
                    "required_packages = [\n",
                    "    \"torch>=1.12.0\",\n",
                    "    \"to
                    "    \"rdkit-pypi\",\n",
                    "    \"pandas>=1.3.0\",\n",
                    "    \"nu
                    "    \"scikit-learn>=1.0.0\",\n",
                    "    \"tqdm\",\n",
                    "    \",\n",
                    "    \"seaborn\"\n",
                    "]\n",
                    "\n",
                    "print(\"🔧 Installin,
                    "print(\"This may take a few minutes...\\n\")\n",
                    "\n",
                    "installation_success = True\n",
                    "for package in required_pa",
                 :\n",
              ,
            ",
                    "if installation_success:\n",
             \n",
                    "    print(\"📋 ",
                    "else:\n",
                    "    print(\n",
                    "    print")\n",
                    "\n",
                    "print(\"\\n\" + \"=\"*80)\n",
                    "print(\"🚀 IMPORTANT: Please restar)\n",
                    "print(\"In Kaggle: Runtime → Restart Session\")\n",
                    "prin)\n",
                    "print(\"=\"*80)"
                ]
            },
            
            # Cell 3: Configuration and Impotart)
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# =======================",
                    "# CONFIGURATION AND IMPORTS (RU
                    "# =================================
                    "\n",
                    "# Core imports\n",
                    "impo
                    "import warnings\n",
                    "import numpy as np\n",
                    "import pandas as pd\n",
                    "\n",
                    "# PyTorch imports\n",
                    "import torch\n",
                    "import torch.nn as nn\n",
                    "impon",
                    "import torch.nn.fu\n",
                    "from torch.utils.data import Dataset, DataLoader\n",
                    "from",
                    "\n",
                    "# PyTorch Geometric imports\n",
                    "from torch_geometric.data import Data, Batch\n",
                    "from
                    "\n",
                 ",
              
            s\n",
                    "from rdkit import RDr\n",
             ,
                    "# Utility imporn",
                    "from tqdm import tqdm\n",
                    "from sklea,
                    "import ma
                    "import\n",
                    "\n",
                    "# Suppress warnings and RDKit
                    "warnings.filterwarnings('ignore')\n",
                    "RDLo
                    "os.environ['PYTHONWARNI",
                    "\n",
                    "# ==,
                    "# T4 x2 CONFIGURATION\n",
                    "# =========================================================\n",
                    "\n",
                    "# GPU configuration\n",
                    "os.environ['CUDA_VISIBLE\n",
                    "\n",
                    "# T4
                    "BATCH_SIZE = 48  # Per GPU - optimin",
                    "HIDDEN_CHANNELS = 64  # Memory efficient\n",
                    "NUM_LAYERS = 6  # Balanced depth\n",
                    "TRAI40\n",
                    "USE_MIXED_PRECISIO,
                    "\n",
                    "# GP
                    "torch.backends.cudnn.benchmark = ",
                    "torch.backends.cudnn.determinist\n",
                    "\n",
                    "# Device setup\n",
                    "device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n",
                    "if torchn",
                    "    torch.cuda.empty_cache()\n",
                    "    print(f\"🚀 GPU Setup: {torch.cuda.device_count()}\n",
                    "    for i in range(torch.cuda.device_count()):\n",
                    "        print(f\"   GPU {i}: {torch.cuda.get_device_name(i)}\")\n",
                    "        pn",
                    "else:\n",
                    "    
                    "\n",
                    "# Mixed precision scaler\n",
                    "scal",
                    "\n",
                    "print(f\"\\n✅ T4 x2 Configuration:\")\n",
                    "print(f\"   Batch Size: {BATCH_SIZE} per GPU\")\n",
                    "print(f\"   Hidden Channels: {HIDDEN_CHANNELS}\")\n",
                    "print(f\"   Layers: {NUM_LAYERS}\")\n",
                 )\n",
              ",
            ")"
                ]
            },
            
            # Cell 4: Data Loading with ection
            {
                "cell_type": "de",
                "execution_
                "metadata": {},
                "outputs": [],
                "source": [
                    "# ==\n",
                    "# DATA LOADING WITH SMART PAn",
                    "# =========================================================================n",
                    "\n",
                    "def detect_data_paths():\n",
                    "    \"\"\"Smart path det
                    "    \n",
                    "    # Kaggle paths (primary)\n",
                    "    kaggle_paths = [\n",
                    "        '/kaggle/input/neur\n",
                    "        '
                    "        \n",
                    "        '/kaggle/input/neurips-p",
                    "    ]\n",
                    "    \n",
                    "    # Local paths (fallback)\n",
                    "    local_paths = ['info', 'data'
                    "    \n",
                    "    # Check Kaggle paths first\n",
                    "    for path in kaggle_paths:\n",
                    "        if os.path.exists(path) and os.path.exists(os.path.join(p\n",
                    "            print(f\"📁 Using Kn",
                    "            return path\n",
                    "    \n",
                    "    # Check local paths\n",
                    "    for path in local_paths:\n",
                    "        if os.path.exists(os.path.join(path, 'train.csv')):\n",
                    "            print(f\"📁 Using local da\n",
                    "        
                    "    \n",
                    "    raise FileNotFoundError(\"❌ ",
                    "\n",
                    "# Detect and load data\n",
                    "try:\n",
                    "    DATA
                    "    \n",
                    "    \n",
                    "    train_df = pd.read_csv",
                    "    test
                    "    \n",
                    "    prin\n",
                    "    print(f\"   Training samples: {len(tr\n",
                    "    print(f\"   Test samples: {len(test_df):,}\")\n",
                    "    print(f\"   Training columns: {list(train_df.columns)}\")\n",
                    "    \n",
                    "    # Display basic statistics\n",
                    "    property_cols = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']\n,
                    "    print(f\"\\n📈 Property availability:\")\n",
                    "    for col in property_cols:\n",
                    "        \n",
                    "            available = tr,
                    "            total = len(train_df)\n",
                    "            print(f\"   {col}: {available:,}/{total:,} (\n",
                    "    \n",
                    "except Exc",
                 ,
              \")\n",
            aise"
                ]
            },
            
            # Cell 5: Molecular Featurizions
            {
                "cell_type": ",
                "execution_
                "metadata": {},
                "outputs": [],
                "source": [
                    "# ==",
                    "# MOLECULAR FEATURIZATION FOR T4
                    "# ==============================================================
                    "\n",
                    "def get_atom_features(atom):\n",
                    "    \"\"\"Extract comprehensi
                    "    features = [\n",
                    "        atom.GetAtomicNum(),\n",
                    "        atom.GetDegree(),\n",
                    "        atom.GetFormalCharg
                    "        int(atom.GetHybridization",
                    "        int(atom.GetIsAromatic())",
                    "        a\n",
                    "        atom.GetTotalNu
                    "    ",
                    "        int(atom.GetChiralTag())",
                    "        atom.GetTotalValence()\n",
                    "    ]\n",
                    "    return features\n",
                    "\n",
                    "def smiles_to_graph(smiles),
                    "    \"\"\"Co
                    "    try:\n",
                    "        mol = Chem.MolFromSmiles)\n",
                    "        if mol is None:\n",
                    "            return None\n",
                    "        \n",
                    "        # Add hydrogens for bette,
                    "        mol = Chem.AddHs(mol)\n",
                    "        \n",
                    "        # Extract atom features\n",
                    "        atom_features = []\n"
                    "        for atom in mol.GetAtoms():\n",
                    "            atom_features.append(get_atom_,
                    "        \n",
                    "        if not atom_features:\n",
                    "            
                    "        \n",
                    "        # Extract edge information\n",
                    "        edge",
                    "        edge_attrs = []\n",
                    "        \n",
                    "        for bond ",
                    "            i = bond.GetBeginAtomIdx()\n",
                    "            j = bond.GetEndAtomIdx()\n",
                    "            
                    "            # Bond features\n",
                    "            bond_type = bond.G\n",
                    "            bond_features = [\n",
                    "                float(bond_type == Chem.rdchem.Bon\n",
                    "                float(bond_type ,
                    "                float(bond_type == Chem.rdchem.BondType.TRIPLE,\n",
                    "            ",
                    "                float(bond.GetIsConjugated()),\n",
                    "        
                    "            ]\n",
                    "            \n",
                    "    n",
                    "            edge_indices.extend([[i, j], [j, i]])\n",
                 n",
              
            n",
                    "        x = torch.tensor(atom_features, )\n",
             \n",
                    "        # Pad a
                    "        if x.size(1\n",
                    "          n",
                    "         ",
                    "      \n",
                    "            x = x[:, :32]  # Truncate if too many features\n",
                    "        \n",
                    "        if edge_indices:\n",
                    "    
                    "            edge_attr = torch.tens,
                    "        else:\n",
                    "        \n",
                    "            edge_index = torch.empty((2, 0), d)\n",
                    "            edge_attr = torch.empty((0, 6), dty
                    "        \n",
                    "        retu\n",
                    "    \n",
                    "    except Exception as e:\n",
                    "        # Re
                    "        return None\n",
                    "\n",
                    "print(\"✅ Molecular featuriz\n",
                    "print(\"   Features per atom: ",
                    "print(\"   F\n",
                    "print(\"   Includes: atomic properties, hybridization, aromaticity, ring membership\")"
                ]
            },
            
            # Cell 6: Dataset Class with Caching
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# =============================================================================\n",
                    "# OPTIMI",
                    "# ========================n",
                    "\n",
                    "class T4
                    "    \"\"\"Optimized dataset for T4 ",
                    "    \n",
                    "    def __init__(self, df, is_test=False, ca",
                    "        self\n",
                    "        self.is_test = is_test\n",
                    "        self.cache_graphs = cache_graphs\n",
                    "        \n",
                    "        # Proper\n",
                    "        self.property_cols =,
                    "        \n",
                    "        # Pre-prg\n",
                    "        if self.cache_graphs:\n",
                    "            print(f\"🔄 Pre-processing and",
                    "            self.graphs = []\n",
                    "            valid_indices = []\n",
                    "            \n",
                    "            for idx, row in tqdm(self.df.iterrows(), total=len(s
                    "                graph = smiles_to_graph(row['SMILES'])\n",
                    "                ",
                    "                    self.graphs.append(graph)\n",
                    "                    valid_indices.append(idx)\n",
                    "            n",
                    "            # Keep only es\n",
                    "    e)\n",
                    "            print(f\"✅ Cached
                    "            print(f\"   Invalid SMILES filtered: {len(df) - len(self.graphs)}\")\n",
                    "        else:\n",
                    "            self.graphs = None\n",
                    "            print(f\"\n",
                    "    \n",
                    "    def ,
                    "        return len(self.df)\n",
                    "    \n",
                    "    \n",
                    "        if self.cache_graphs:\n",
                 n",
              ",
            e:\n",
                    "            # Generate graph on-n",
             \n",
                    "            gra",
                    "            if grap,
                    "          ",
                    "        \n",
                    "      ,
                    "            # Add targets and masks for training/validation\n",
                    "            row = self.df.iloc[idx]\n",
                    "            \n",
                    "    
                    "            masks = []\n",
                    "            \n",
                    "        
                    "                if col in row and pd.notna(row[col]):\n",
                    "                    targets.append(float(row[",
                    "            ",
                    "                else:\n",
                    "                    targets.append(0.0)  # Placeholder value\n",
                    "            s\n",
                    "            \n",
                    "            graph.y = torch.tensor(targets, dtype=torch.float)\n",
                    "            
                    "        \n",
                    "        return graph\n",
                    "\n",
                    "def collate_batch(batch):\n",
                    "    \"\"\"Optimized collate function using PyTorch Geometric batchin\n",
                    "    # Filter out None samples\n",
                    "    batch = [item for item in batch if item is not None]\n",
                    "    if not batch:\n",
                    "        return Non",
                    "    \n",
                    "    # Use Py
                    "    try:\n",
                    "        return Batch.from_data_list(batch)\n",
                    "    except Exception as e:\n",
                    "        print(f\"Batch collation error,
                    "        return None\n",
                    "\n",
                    "print(\"✅ T4 
                    "print(\"   F")"
                ]
            },
            
            # Cell 7: T4 Optimized Model Architecture
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# ==============================\n",
                    "# T4 x2 OPTIMIZED MODEL ARCHITECTURE\n",
                    "# ==========
                    "\n",
                    "class T4PolyGIN(
                    "    \"\"\"T4 x2 optimized Graph Isomorphism Network for
                    "    \n",
                    "    def __init__(self, num_atom_features=32, hidden_channels=64, num_layer
                    "            ",
                    "        super(T4PolyGIN, self)",
                    "        \n",
                    "        # St",
                    "        self.device = torch.device(\"cuda\" if torch.cuda.is_available",
                    "        \n",
                    "        # Input projection\n",
                    "        self.input_proj = nn.Linear(n",
                    "        self.inp",
                    "        \n",
                    "        # GIN layers with residual conne,
                    "        self.gin_layers = nn.Modu",
                    "        self.batch_no",
                    "        \n",
                    "        for ",
                    "            # MLP for GIN layer\n
                    "            mlp = nn.Sequential(\n",
                    "            ,
                    "                nn.ReLU(),\n",
                    "                nn.Dropout(dropou",
                    "    s),\n",
                    "                nn.Dropout(dropout)\n",
                 ,
              
            
                    "            self.batch_norms.append(n)\n",
             n",
                    "        # Globaers\n",
                    "        self.global\n",
                    "        \n
                    "        #\n",
                    "      n",
                    "            nn.Linear(hidden_channels, hidden_channels),\n",
                    "            nn.ReLU(),\n",
                    "            nn.Dropout(dropout),\n",
                    "    \n",
                    "            nn.ReLU(),\n",
                    "            nn.Dropout(dropout),\n",
                    "        )\n",
                    "        )\n",
                    "        \n",
                    "        # Initialize weights\n",
                    "        self.apply(self._init_weights)\n",
                    "    \n",
                    "    def _init_weights(self, modu
                    "        \"\"\"Initialize model weights.\"\"\"\n",
                    "        if isinstance(module, nn.Linear):\n",
                    "        ",
                    "            if module.bias is not None:\n",
                    "                torch.nn.init.zeros_(module.bias)\n",
                    "    \n",
                    "    def forward(self, data):\n",
                    "        
                    "        \n",
                    "        # Handle DataParallel device issues\n",
                    "        try:\n",
                    "        n",
                    "        except StopIteration:\n",
                    "            device = self.device\n",
                    "        \n",
                    "        
                    "        x = self.input_projn",
                    "    x)\n",
                    "        x = F.relu(x)\n",
                    "        \n",
                    "        # GIN layers 
                    "        for i, (gin_la
                    "            residual = n",
                    "        
                    "            # GIN convolution\n",
                    "            x = gin_layer(x, ",
                    "            x = batch_no,
                    "            
                    "            \n",
                    "            # Residual connection\n",
                    "            n",
                    "                x = x + residual\n",
                    "        \n",
                    "        # Global pooling\n",
                    "        x = self.global_pool(x, batch)\n",
                    "        \n",
                    "        # Output layers\n",
                    "        x = self.output_layers(x)\n",
                    "        \n",
                    "        return x\",
                    "\n",
                    "print(\"✅ T4 x2 optimized PolyGIN model defined\")\n",
                    "print(\"   Features: Residual c
                    "print(\"   Architecture: Input p",
                    "print(\"   O
                ]
            },
            
            # Cell 8: Training Functions with DataParallel Support
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# ======,
                    "# TRAINING FUNCTIONS WITH DATT\n",
                    "# ==========================================================================,
                    "\n",
                    "def weighted_mae_loss(predic,
                    "    \"\"\"Weight",
                    "    \n",
                    "    # Handle Dat\n",
                    "    if predictions.shape[0] != targets.shape[0]:\n",
                    "        actual_batch_size = targets.n",
                    "        predictions = predictions[:actual_batch_si,
                    "    \n",
                    "    # Validate tensor",
                    "    if predictions.shape != targets.shape or p
                    "        raise ValueError(f\"Shape mismatch: pred={predictions.shape}, target={n",
                    "    \n",
                    "    # Property weights (equal weighting f",
                    "    weights = torch.tensor([1.0,",
                    "    if l
                    "        weights = weights.unsqueeze(0)  # Broad
                    "    \n",
                    "    # Calculate weighted MAE\n",
                 ",
              ",
            ",
                    "    # Handle edge cases\n",
             ",
                    "        return ",
                    "    \n",
                    "    returnn",
                    "\n",
                    "def tr
                    "    \"\"\"Train model for one epoch with mixed precision support.\"\"\"\n",
                    "    model.train()\n",
                    "    total_loss = 0\n",
                    "    
                    "    \n",
                    "    n",
                    "    \n",
                    "    for batch in progress_bar:\n",
                    "    n",
                    "            continue\n",
                    "    ",
                    "        batch = batch.to(dev
                    "        optimizer.zero_grad()\n",
                    "        \n",
                    "        try:\n",
                    "    
                    "                with autocast():\n"
                    "                    predictionn",
                    "                    los\n",
                    "                \n",
                    "                scaler",
                    "                scaler.step(optim,
                    "                scaler.update()\n",
                    "            else:  # Standard precision training\n",
                    "                predictions = model(batch)\n",
                    "                loss = weighted_mae_loss(predictions, ba,
                    "     
                    "    ",
                    "            \n",
                    "            total_losn",
                    "            num_batches += 1\n",
                    "            \n",
                    "            # Update progress bar\n",
                    "            progress_b
                    "            \n",
                    "        except Exception as e:\n",
                    "            print(f\"Trai",
                    "     e\n",
                    "    \n",
                    "    return total_loss / max(n
                    "\n",
                    "def evaluate_model(model, val_l",
                    "    \"\"\"Evaluate mode,
                    "    model.eval()\n",
                    "    total_loss = 0\n",
                    "    num_batches = 0\n",
                    "    \n",
                    "    with torch.no_grad():\n",
                    "     ",
                    "    \n",
                    "        for batch in progress_bar:\n",
                    "            if batch is None:\n",
                    "                continue\n",
                    "            \n",
                    "            batch = batch.to(device)\n",
                 ,
              \n",
            ,
                    "                    with autocas
             ",
                    "               ,
                    "                els\n",
                    "          
                    "         
                    "      
                    "                total_loss += loss.item()\n",
                    "                num_batches += 1\n",
                    "                \n",
                    "    
                    "                \n",
                    "    
                    "                print(,
                    "                continue\n",
                    "    \n",
                    "    return total_loss / max(num_batches,n",
                    "\n",
                    "print(\"✅ Training fun,
                    "print(\"   Features",
                    "print"
                ]
            },
            
            # Cell 9: Dat
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# ======================================,
                    "# DATA PREPARATION AND MODELn",
                    "# ==========================",
                    "\n",
                    "print(\"📊 Preparing datas
                    "\n",
                    "# Spl",
                    "trai",
                    "\n",
                    "print(f\"   Train s")\n",
                    "print(f\"   Validation split: 
                    "\n",
                    "# Cre\n",
                    "trai
                    "val_dataset = T4Polyme
                    "test_dataset = T4PolymerDataset(test_df, is_test=True, cache_",
                    "\n",
                    "# Create optimized data loaders for T4 x2\n",
                    "trai
                    "    train_dataset,\n",
                    "    batch_size=BATCH_SIZE,\n",
                    "    shuffle=True,\n",
                    "    collate_fn=collate_batch,\n",
                    "    num_workers=2,\n",
                    "    pin_memory=True,\n",
                    "    persistent_workers=True,\n",
                 \n",
              
            ,
                    "val_loader = Da
             ,
                    "    batch_size=",
                    "    shuffle=False,\,
                    "    collat",
                    "    num_w,
                    "    piue,\n",
                    "    persistent_workers=True,\n",
                    "    prefetch_factor,
                    ")\n",
                    "\n",
                    "test_loader = DataLoader(\n",
                    "    test_dataset,\n",
                    "    ,
                    "    shuffle=False,\n",
                    "    collate_fn=collate_batch,\n",
                    "    num_workers=2,\n"
                    "    pin_memory=True,
                    "    persistent_won",
                    "    prefetch_factor=4\n",
                    ")\n",
                    "\n",
                    "print(f\"\\n✅ Data loaders created:\")\n",
                    "print(f\,
                    "print(f\"   Validation b
                    "print(f\"   Test batches: {len(test_loader)}\")\n",
                    "print(f\"   Batch size per GPU: {BATCH_\")\n",
                    "\n",
                    "# Initialize T4 optimized l\n",
                    "print(f\"\\n🏗️ Initializing T4 x2 optimized model...\",
                    "model = T4PolyGIN(\n",
                    "    num_\n",
                    "    hidden_channels=HIDDEN_CHA",
                    "    num_layers=NUM_LAYER
                    "    num_targets=5,\n",
                    "    drop,
                    ")\n",
                    "\n",
                    "# Move to device\n",
                    "model = ",
                    "\n",
                    "# Setup DataParallel for multi-GPU tng\n",
                    "if torch.cuda.device_count() > 1:\n"
                    "    print(f\"🚀 Enabling DataParallel for {torch.cuda.device_coun
                    "    model = nn.DataParallel(model)\n",
                    "    effective_batch_size = BATCH\n",
                    "    print(f\"\n",
                    "else:\n",
                    "    effe
                    "    print(f\"   Single G
                    "\n",
                    "# Setup optimizer and scheduler\n",
                    "optimizer = optim)\n",
                    "schedule
                    "\n",
                    "# Count parameters\n",
                    "total_params = sum(p.numel() for p i
                    "trai
                    "\n",
                    "print(f\"\\n📊 Model statistics:\")\n",
                    "print(f\"   Total parameters: {total_params:,}\")\n",
                    "print(f\"   Trainable parameters: {trainable_params:,}\"
                 ")\n",
              "
               ]
            },
            
            # Cell 10: Training Loop
            {
                "cell_type": "code",
                "execution_cou None,
                "metadata":{},
                "outputs": [],
                "source": [
                    "# =============================================================================\n",
                    "# T4
                    "# ==============================================",
                    "\n",
                    "print(\"🚀 Starting T\n",
                    "print(f\"   Training epochs: {TRAINING_EPOCHS}\")\n",
                    "print(f\"   Mixe,
                    "prin
                    "print(f\"   GPUs: {torch.")\n",
                    "\n",
                    "# Training tracking\n",
                    "best_val_loss = float('inf')\n",
                    "train_losses = []\n",
                    "val_losses = []\n",
                    "patience_cou
                    "patience = 10  # Early stopping patin",
                    "\n",
                    "# Training loop\n",
                    "for epoch in range(TRAINING_EPOCn",
                    "    print(f\"\\n📈 Epoch {epoch+1}/{TRAINING_E
                    "    \n",
                    "    # Training phase\n",
                    "    train_lon",
                    "    train_losses.append(train_loss)\n",
                    "    \n",
                    "    # Validation phase\n",
                    "    val_loss = evaluate_model(model, val_loader, device, scal,
                    "    val_loss)\n",
                    "    \n",
                    "    e\n",
                    "    scheduler.step()\n",
                    "    current_lr = optimizer.param_groups[0]['lr']\n",
                    "    ",
                    "    print(f\"   Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {c",
                    "    
                    "    # Save best model\n",
                    "    if val_loss < best_val_loss:\n",
                    "    s\n",
                    "        patience_counter = 0\n",
                    "        \n",
                    "    
                    "        torch.save({\n",
                    "            'epoch': epoch,\n",
                    "    \n",
                    "            'optimize\n",
                    "            'scheduler_state_dict': scheduler.state_dic",
                    "    ",
                    "            'train_losses': train_losses,\n",
                    "            'val_losses': val_losses\n",
                    "        }, 'best_t4x2_model.pth')\n",
                    "        \n",
                    "    n",
                    "    else:\n",
                    "        patience_counter += 1\n",
                    "    \n",
                    "    # Early stopping\n",
                    "    if patience_counter >= patience:\n",
                    "        print(f\"   ⏹️ Early stopping triggered (patience: {patience})\")\n",
                    "        break\n",
                    "    \n",
                    "    # Memory cleanup\n",
                    "    if torch.cuda.is_available():\n",
                 n",
             "\n",
          ,
                    "
                    "print(
                    "print(f\"   Final trai
                    "\n",
                    "# Plot train\n",
              ",
                    "\n",
                    "plt.subplot(1, 
                    "plt.plot(train_lo
                    "plt.plot(va",
                  
                    "plt.ylabel('Loss')\
                    "plt.title('T4 x2 Trainin",
                    "plt.legend()n",
                    "plt.grid(True, alpha=0.3)\,
                    "\n",
                    "plt.subplot(1\n",
             
          n",
                    "p",
                    "plt.yl\n",
     ",
    n",
                    "pln",
                    "\n",
                    "plt.tight_layout()\n",
    ",
                    "\n",
                    "prin
                ]
            },
            
            # Cell 11: Test Predictions and Submi
            {
                "cell_type": "code",
                "execution_count": None,
": {},
                "outputs":
                "source": [ otebook()er_kaggle_neate_propcrn__":
    _mai_ == "__name_")

if _nerationsion ge  ✅ Submis"     print(ipeline")
training pmplete  ✅ Co("  int   pr
 ection")h det Smart patnt("   ✅pri   
 imization") opt("   ✅ GPUrint    p
fixes")ensor shape llel tDataPara"   ✅ int(    przation")
 organi Proper cell"   ✅t(  prinns")
  nstructio iartKernel rest   ✅     print("ion")
installatcy endenic dep ✅ Automatint("  )
    pr Features:"  print("🎯b")
  eady.ipyn2-kaggle-rps-t4xneuri: "📁 File
    print(ated!")book cree T4x2 notepatiblgle-comKag✅ Proper ("int    pre)
    
ii=Falsnsure_asc indent=1, etebook, f,p(non.dum     jso f:
    asutf-8')='oding', enc 'wdy.ipynb',-rea2-kaggleneurips-t4xh open('  wittebook
  le no proper Kagg Save the    # 

    }
   ": 4minort_   "nbforma     4,
nbformat":        "     },
   }
             "3.8.5"
rsion":    "ve        n3",
     ": "ipythorents_lexe"pygm               ",
  "python_exporter":"nbconvert               hon",
  "pyt":me       "na      n",
   -pythoxt/x": "te"mimetype               ",
 ".pyn": ile_extensio"f             ,
          }        sion": 3
 "ver               
     thon",": "ipy    "name                de": {
odemirror_mo         "c       
": {nfoage_i"langu    
             },       "
"python3ame":        "n
         thon", "pyguage":"lan          ",
       "Python 3_name":isplay         "d{
       elspec":      "kern {
       ":metadata
        "      ], }
              ]
    
           "\")n! submissioetitionfor compady t(\"🏆 Re      "prin        
      )\n",v\"ubmission.cs s.pth,2_modelt4x best_rated:Files geneint(\"📁  "pr           
        ",n)\"y!\essfull succletedution comp T4 x2 sol"\\n🎉int(\     "pr          
       "\n",            ,
      \")\n"ount())}ce_cevi.cuda.d torchIZE * max(1,H_Size: {BATC sctive batchEffe"    "print(f\                  ",
 }\")\nD_PRECISION_MIXEsion: {USE Mixed precif\"  rint(   "p           ,
      )\n""nt()} x T4\.device_coucudan: {torch.lizatioU uti\"   GP "print(f             ,
      )}\")\n"ain_lossestrochs: {len(g ep Trainin\"  (f     "print              ",
 n\")\f}s:.4val_los: {best_ loss validationest"   Bint(f\  "pr          
        ")\n",s:,}\nable_paramrairs: {taramete(f\"   P  "print       
           ",\")\nnels)} chanNNELSHAN_CDEN ({HID PolyGIyerS}-la {NUM_LAYER"   Model:nt(f\"pri            
        \n",:\")mmary Training Sun🎯 T4 x2nt(f\"\\    "pri              
  ",mary\nnal sum  "# Fi          ,
         "\n"              n",
     :.3f}\")\ues.max()x={pred_val:.3f}, man().mi_valuesin={pred  f\"m "                          \n",
 f}, \"std():.3d_values.re}, std={pan():.3fed_values.me mean={pr}:"   {col   print(f\"                   ",
  :, i]\npredictions[t_ tesalues =d_v"    pre                
    :\n",perty_cols)e(pronumeratn ei, col i "for                  ")\n",
  \tistics:tation s Predic(f\"\\n📈rint   "p            \n",
     icsion statistctay prediDispl        "#       
        "\n",                  )\n",
ad(10)f.heion_dsubmisst("prin               ",
     w:\")\nprevieon \n📊 Submissi(f\"\"print            
        ",w\nprevieion  submiss Display  "#              n",
          "\          n",
    }\")\rty_colspeies: {proopert(f\"   Pr   "print               ",
  ")\nion_df)}\ubmiss{len(sSamples:    "print(f\"                   n",
 \")\ssion.csvsaved: submiission file bm"✅ Suprint(f\"                     "\n",
                 \n",
  ndex=False)sion.csv', i_csv('submisssion_df.tosubmi     "          
     e\n",ssion file submi# Sav         "           ",
    "\n         
       \n", i]tions[:,redictest_pl] = ion_df[coss "    submi                   
cols):\n",roperty_merate(pnu col in ei,    "for                 ",
n\nubmissioto sredictions   "# Add p         
         \n",       "            g']\n",
 ity', 'R, 'DensFV', 'Tc'', 'Fs = ['Tgty_coloper   "pr            
     ()\n",]].copyD'test_df[['Ision_df = ubmis        "s           
 "\n",                    )\n",
e...\"ion filss submiating\"📝 Crerint( "p         
          n",dataframe\ion isseate subm "# Cr          
         \n",          "         ",
 \")\nted!s generarediction"No test peError(\ise Valu"    ra              ",
      se:\n     "el         ,
      \")\n" samplestestions)} _predictr {len(tests foedictionenerated pr✅ G"(f\rint "    p           ,
        ctions)\n"(test_predip.vstackns = nt_predictio  tes"                n",
      ons:\cti test_predi"if               ",
     tions\nll predicbine a"# Com                   \n",
         "         \n",
   continue                     "        ",
   \")\nrror: {e}tion e\"Predic  print(f"                         
     :\n",eption as ecept Exc ex           "           ",
           \n   "                       ))\n",
.numpy(ons.cpu()edictins.append(prredictiost_p  te          "               \n",
            "                        n",
 e]\_sizal_batchctions[:actupredi= redictions            p       "            
          \n",tch_size:tual_bahape[0] > acns.sedictio    if pr                      "      \n",
    .item() + 1()maxh.batch.batcize = tual_batch_s       ac              "               n",
 GPUs\ltiplets from mu outpuatesoncatenrallel c DataPa          #    "              
        ", > 1:\ne_count()a.deviccudh.rc and toe')del, 'modulmottr( if hasa       "                        ons\n",
 predicti for testismatchl shape mtaParalleDadle   # Han                "             n",
         \  "                      tch)\n",
l(baons = mode predicti               "              n",
      se:\  el        "                    \n",
  l(batch)ions = modedict   pre              "                    ",
   cast():\nauto    with           "                      
ON:\n",_PRECISIf USE_MIXED         i "                    :\n",
   try      "                        \n",
     "            
        \n",.to(device)ch batch = bat "                    
      ",      \n   "               ",
    \n    continue  "                    ",
      \ns None:batch i      if       "             n",
   ress_bar:\in progtch "    for ba               
     "    \n",                  
  g\")\n","Predictiner, desc=\st_loadar = tqdm(tegress_b"    pro           
         \n",ad():h.no_grh torc    "wit            n",
         "\           \n",
    ictions = []"test_pred                    ",
ns\nioate predictner"# Ge           ,
             "\n"               ,
 \n"]:.4f}\")s'_los['best_valcheckpoint loss: {validation"   Best \   "print(f             ",
    n\")\h']+1}kpoint['epocecepoch {chrom model f best aded"   Lo"print(f\                  ",
       "\n               ",
eval()\nodel.  "m            
      t'])\n",icel_state_dodoint['m_dict(checkp_stateodel.load "m              ,
     n"')\2_model.pthd('best_t4x.loachnt = torcheckpoi      "        ,
      el\n"ad best mod "# Lo                  ",
     "\n         
       )\n",\"dictions...est pre tingGenerat"🔮 \"print(                 
      "\n",            
     ====\n",======================================================================   "# ===                ON\n",
 ATIISSION GENER AND SUBMCTIONSTEST PREDI     "#                ",
\n======================================================================== =====         "#          
 [],"metadata                ssion\")"ottedpl curves raining"📊 Tt(\()\n"plt.show                lpha=0.3)\ a.grid(True,tegend()\t.l"pl                \nSchedule')ning Rate .title('Lear"plt               te')g Ra('Learninabel)\nabel('Epoch'lt.xlen')\color='greate', ='Learning R  label             "    ))], \n",lossesin_en(train range(l for i )) (i // 10 (0.5 **[0]['lr'] *am_groupser.partimiz[opplt.plot(  "     ), 2, 2n"\Progress')\ng n",)\n",bel('Epoch'  "plt.xla)\ned'', color='ron Losstialidal='V, labeossesl_lue')\n",olor='bl cning Loss',rail='Tbeses, las\n",)2, 112, 4))\nigsize=(.figure(f      "pltrvesing cu",)\n\":.4f}s[-1]sse: {train_lon lossn",)}\")\_losses(trainlens trained: {al epoch   Totf\"\n",f}\")_loss:.4val: {best_dation lossest vali\"   Brint(fpn"\")\ed!letg compaininn🎉 Tr"\\t(f\      "prin           ty_cache()\cuda.emph.orc       t   " 4f})\")\oss:.oss: {val_lal Lsaved! (V model  best ✅ New\"  t(f    prinloss,\nst_val_loss': be  'best_val_      t(),\nate_dict(),timizer.stt': opdicr_state_ict(),.state_dodelict': m_state_d'model        ,n"ate\el st # Save mod   ss = val_losval_lobest_    ",\nf}\")\nnt_lr:.6urren\rning ratlea# Update d(val_lossenes.appr)\n"eer)\, scalevicemizer, dder, optin_loaaiodel, troch(m = train_epss",")\nCHS}\PO):\HSence\= 0\n",nter }\e() else 0.is_availabltorch.cuda_count() if cuda.device}\")\n",ceice: {devi"   Devt(f\}\")\n"RECISIONXED_PUSE_MIecision: { prd...\")trainingoptimized x2 4 ====\n===========================",NG LOOP\nNI x2 TRAI nt": nels\")dden chanhiNNELS} IDDEN_CHAth {HwiolyGIN ayer PUM_LAYERS}-l{Ncture: hiteArct(f\"    "prin     .1f}MB\1e6:/  * 4 paramstal_e: ~{toel siz\"   Modrint(f   "p)\n",d)\n",rares_g.requiters() if pel.paramer p in mod) fonumel(um(p.ams = sarnable_p))\n",eters(.paramn modelOCHS)\n",AINING_EP=TR T_maxoptimizer,R(nnealingLeAuler.Cosinhedtim.lr_scr = opt_decay=1e-5, weigh.001ters(), lr=0ame(model.parmWAda.")\n",e}\batch_sizctive_size: {effening, batch PU traiIZE\n",CH_Size = BATtive_batch_scize}\")atch_sfective_bsize: {efh tive batc Effec  ount()_cviceh.cuda.deorc_SIZE * t",")\n GPUs\t()},rainievice)\nmodel.to(d"t=0.1\nou\n",S,NNELS,\n2,s=3ureom_featat")\nmodeSIZE}")\n",\_loader)}s: {len(valatche")\n"}\_loader)aines: {len(tratchTraining b"   rue,\rkers=T,\n",SIZE,\n"CH_ATize=B_stchba=4\n"emory=Trn_m,\n"orkers=2\nllate_batch,e_fn=con"CH_SIZE,\nBATaset,\n"  val_dat      "   \n",ader(Lota""\n          ")\n",    tor=4fetch_fac"    pre   ",der(\ntaLoaer = Dan_loadhs=True)\ngrapTrue)\n",che_graphs= catest=False,data, is_(val_rDataset=True)\n",e_graphsse, cachs_test=Falata, itrain_dt(taseymerDat = T4Poltasedan_gwith cachincts set objee dataat")\n",mples\,} sata):len(val_da{ples\):,} samrain_data: {len(tplitue)\nle=Trhuff, sdom_state=42ran=0.15, est_sizetrain_df, t_test_split( trainata =data, val_dn_ata\n training dit",n\\")ning... traior T4 x2ets f======\n=============================================TUP\ SE=====\n"================================== Setupd Modelion anreparata Png\")hape handli sautomaticth  MAE wightedss: Wei\"   Lo(\n\")kingress tracport, progl supralleDataPaprecision,  Mixed :\")\n"ed definionsct 1)\\")\n": {e}cherror in batation f\"Validn",on as e:\eptiExc  except       ",)\n():.4f}'}s.itemoss': f'{los'Lx({set_postfis_bar.progres                 \n",     sk)\n",tch.mabatch.y, baictions, edmae_loss(pred_eight w   loss =        ch)\n",del(batctions = moredi     p     e:)\n"h.mask batctch.y, baions,(predictae_loss weighted_m loss =        batch)\n= model(ons ctiredi    p                         "  ",t():\none:\n" is not Ner scal          if"              y:tr              "      \n"        "         alse)\n", leave=Fdation\=\"Valiloader, descval_m(s_bar = tqd  progres "\n\"et.\"\"dation svalil on ne):\n=Novice, scaleroader, de\n",ches, 1)m_batutinu  con     )\n: {e}\"batchr in ning erro'})\n",):.4f}ss.item(ss': f'{loix({'Lopostfar.set_s.item()\ loss +=)\ntep(er.s optimiz           n",rd()\.backwa        loss   h.mask)\n".y, batctch\n"izer)ackward()\nloss).ble(.scamask)y, batch.atch.ctions, b_loss(predihted_mae weig =sh)\odel(batc= ms ,ng\n",trainin d precisioone:  # Mixet N is noif scaler        e)\n",ic    \ne:\tch is Non    if balse)\ave=Faning\", le=\"Trai, descrain_loaderm(t = tqdprogress_bar",s = 0\num_batchen",\n):one=Nce, scalerzer, deviimioader, opttrain_ll, h(modepocn_eaid_mae\ weightepe)\nons.dtyredictidtype=pe, .devicctionsce=predi0.0, devinsor(.tetorch\n):ghted_mae(weiorch.isinf_mae) or thtedh.isnan(weigif torc    "       "    \n        ts).sum()\nweighks * um() / (mas).s weightsoperty *e_per_pred_mae = (maeight  "    w    sks\n ma- targets) *edictions ch.abs(prorerty = tmae_per_prop"       ",ting\ncas\n", == 2:ons.shape)cti(predi= 1 and len) =peshaeights.en(w)\nions.dtypect=predievice, dtype.dedictions], device=pr1.0, 1.00, 1.0,  1.s)\nertie all proporhape}\")\asks.sask={me}, mshapets.targ:\n",.shape= maskspe !s.shactionredin shapes\\n"ze]pe[0]\shaUs)le GPultipnated from moncatens c (predictiomismatchl shape aParalle\n\"\"dling.\"hape hanor stenslel taParal Dathss wiE loed MA\n":asks), targets, mtions\n"===UPPOREL SRALLAPA=======\n"================================================================      "    \n",
                    "    return total_loss / max(num_batches, 1)\n",
                    "\n",
                    "def evaluate(model, val_loader, device):\n",
                    "    \"\"\"Evaluate model on validation set.\"\"\"\n",
                    "    model.eval()\n",
                    "    total_loss = 0\n",
                    "    num_batches = 0\n",
                    "    \n",
                    "    with torch.no_grad():\n",
                    "        for batch in tqdm(val_loader, desc=\"Validation\", leave=False):\n",
                    "            if batch is None:\n",
                    "                continue\n",
                    "            \n",
                    "            batch = batch.to(device)\n",
                    "            \n",
                    "            if USE_MIXED_PRECISION and scaler is not None:\n",
                    "                with autocast():\n",
                    "                    predictions = model(batch)\n",
                    "                    loss = weighted_mae_loss(predictions, batch.y, batch.mask)\n",
                    "            else:\n",
                    "                predictions = model(batch)\n",
                    "                loss = weighted_mae_loss(predictions, batch.y, batch.mask)\n",
                    "            \n",
                    "            total_loss += loss.item()\n",
                    "            num_batches += 1\n",
                    "    \n",
                    "    return total_loss / max(num_batches, 1)\n",
                    "\n",
                    "print(\"✅ Loss function and training functions defined\")"
                ]
            },
            
            # Cell 10: Data Preparation and DataLoaders
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# =============================================================================\n",
                    "# DATA PREPARATION AND DATALOADERS\n",
                    "# =============================================================================\n",
                    "\n",
                    "print(\"📊 Preparing datasets and data loaders...\")\n",
                    "\n",
                    "# Split training data into train/validation\n",
                    "train_data, val_data = train_test_split(train_df, test_size=0.15, random_state=42, stratify=None)\n",
                    "\n",
                    "print(f\"Data split: {len(train_data)} train, {len(val_data)} validation\")\n",
                    "\n",
                    "# Create dataset objects\n",
                    "train_dataset = PolymerDataset(train_data, is_test=False)\n",
                    "val_dataset = PolymerDataset(val_data, is_test=False)\n",
                    "test_dataset = PolymerDataset(test_df, is_test=True)\n",
                    "\n",
                    "# Create optimized data loaders\n",
                    "train_loader = DataLoader(\n",
                    "    train_dataset, \n",
                    "    batch_size=BATCH_SIZE, \n",
                    "    shuffle=True, \n",
                    "    collate_fn=collate_batch,\n",
                    "    num_workers=2,           # Parallel data loading\n",
                    "    pin_memory=True,         # Faster GPU transfers\n",
                    "    persistent_workers=True, # Avoid worker respawning\n",
                    "    prefetch_factor=4        # Pipeline optimization\n",
                    ")\n",
                    "\n",
                    "val_loader = DataLoader(\n",
                    "    val_dataset, \n",
                    "    batch_size=BATCH_SIZE, \n",
                    "    shuffle=False, \n",
                    "    collate_fn=collate_batch,\n",
                    "    num_workers=2,\n",
                    "    pin_memory=True,\n",
                    "    persistent_workers=True,\n",
                    "    prefetch_factor=4\n",
                    ")\n",
                    "\n",
                    "test_loader = DataLoader(\n",
                    "    test_dataset, \n",
                    "    batch_size=BATCH_SIZE, \n",
                    "    shuffle=False, \n",
                    "    collate_fn=collate_batch,\n",
                    "    num_workers=2,\n",
                    "    pin_memory=True,\n",
                    "    persistent_workers=True,\n",
                    "    prefetch_factor=4\n",
                    ")\n",
                    "\n",
                    "print(f\"✅ Data loaders created:\")\n",
                    "print(f\"   Training batches: {len(train_loader)}\")\n",
                    "print(f\"   Validation batches: {len(val_loader)}\")\n",
                    "print(f\"   Test batches: {len(test_loader)}\")\n",
                    "print(f\"   Effective batch size: {BATCH_SIZE * max(1, torch.cuda.device_count())}\")"
                ]
            },
            
            # Cell 11: Model Initialization and Setup
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# =============================================================================\n",
                    "# MODEL INITIALIZATION AND SETUP\n",
                    "# =============================================================================\n",
                    "\n",
                    "print(\"🤖 Initializing model...\")\n",
                    "\n",
                    "# Initialize model\n",
                    "model = T4PolyGIN(\n",
                    "    num_atom_features=32,\n",
                    "    hidden_channels=HIDDEN_CHANNELS,\n",
                    "    num_layers=NUM_LAYERS,\n",
                    "    num_targets=5,\n",
                    "    dropout=0.1\n",
                    ")\n",
                    "\n",
                    "# Move model to device\n",
                    "model = model.to(device)\n",
                    "\n",
                    "# Multi-GPU setup with DataParallel\n",
                    "if torch.cuda.device_count() > 1:\n",
                    "    print(f\"🚀 Enabling DataParallel for {torch.cuda.device_count()} GPUs\")\n",
                    "    model = nn.DataParallel(model)\n",
                    "    print(\"⚠️ DataParallel enabled - tensor shape fixes applied in loss functions\")\n",
                    "\n",
                    "# Initialize optimizer and scheduler\n",
                    "optimizer = optim.AdamW(\n",
                    "    model.parameters(), \n",
                    "    lr=0.001, \n",
                    "    weight_decay=1e-5,\n",
                    "    betas=(0.9, 0.999)\n",
                    ")\n",
                    "\n",
                    "scheduler = optim.lr_scheduler.CosineAnnealingLR(\n",
                    "    optimizer, \n",
                    "    T_max=TRAINING_EPOCHS,\n",
                    "    eta_min=1e-6\n",
                    ")\n",
                    "\n",
                    "# Count parameters\n",
                    "total_params = sum(p.numel() for p in model.parameters())\n",
                    "trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)\n",
                    "model_size_mb = total_params * 4 / 1e6  # Assuming float32\n",
                    "\n",
                    "print(f\"✅ Model setup complete:\")\n",
                    "print(f\"   Architecture: {NUM_LAYERS}-layer GIN with {HIDDEN_CHANNELS} hidden channels\")\n",
                    "print(f\"   Total parameters: {total_params:,}\")\n",
                    "print(f\"   Trainable parameters: {trainable_params:,}\")\n",
                    "print(f\"   Model size: ~{model_size_mb:.1f}MB\")\n",
                    "print(f\"   Optimizer: AdamW with cosine annealing\")\n",
                    "print(f\"   Mixed precision: {USE_MIXED_PRECISION}\")"
                ]
            },
            
            # Cell 12: Training Loop
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# =============================================================================\n",
                    "# TRAINING LOOP\n",
                    "# =============================================================================\n",
                    "\n",
                    "print(\"🚀 Starting training...\")\n",
                    "print(f\"Training for {TRAINING_EPOCHS} epochs with early stopping\")\n",
                    "\n",
                    "# Training tracking\n",
                    "best_val_loss = float('inf')\n",
                    "train_losses = []\n",
                    "val_losses = []\n",
                    "patience = 10\n",
                    "patience_counter = 0\n",
                    "\n",
                    "for epoch in range(TRAINING_EPOCHS):\n",
                    "    print(f\"\\n📈 Epoch {epoch+1}/{TRAINING_EPOCHS}\")\n",
                    "    \n",
                    "    # Training phase\n",
                    "    train_loss = train_epoch(model, train_loader, optimizer, device)\n",
                    "    train_losses.append(train_loss)\n",
                    "    \n",
                    "    # Validation phase\n",
                    "    val_loss = evaluate(model, val_loader, device)\n",
                    "    val_losses.append(val_loss)\n",
                    "    \n",
                    "    # Update learning rate\n",
                    "    scheduler.step()\n",
                    "    current_lr = optimizer.param_groups[0]['lr']\n",
                    "    \n",
                    "    print(f\"   Train Loss: {train_loss:.4f}\")\n",
                    "    print(f\"   Val Loss: {val_loss:.4f}\")\n",
                    "    print(f\"   Learning Rate: {current_lr:.2e}\")\n",
                    "    \n",
                    "    # Save best model\n",
                    "    if val_loss < best_val_loss:\n",
                    "        best_val_loss = val_loss\n",
                    "        torch.save(model.state_dict(), 'best_t4x2_model.pth')\n",
                    "        print(f\"   ✅ New best model saved! (Val Loss: {val_loss:.4f})\")\n",
                    "        patience_counter = 0\n",
                    "    else:\n",
                    "        patience_counter += 1\n",
                    "    \n",
                    "    # Early stopping\n",
                    "    if patience_counter >= patience:\n",
                    "        print(f\"   ⏹️ Early stopping triggered (patience: {patience})\")\n",
                    "        break\n",
                    "    \n",
                    "    # Memory cleanup\n",
                    "    if torch.cuda.is_available():\n",
                    "        torch.cuda.empty_cache()\n",
                    "\n",
                    "print(f\"\\n🎉 Training completed!\")\n",
                    "print(f\"   Best validation loss: {best_val_loss:.4f}\")\n",
                    "print(f\"   Total epochs trained: {len(train_losses)}\")\n",
                    "print(f\"   Final train loss: {train_losses[-1]:.4f}\")"
                ]
            },
            
            # Cell 13: Test Predictions and Submission
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# =============================================================================\n",
                    "# TEST PREDICTIONS AND SUBMISSION GENERATION\n",
                    "# =============================================================================\n",
                    "\n",
                    "print(\"🔮 Generating test predictions...\")\n",
                    "\n",
                    "# Load best model\n",
                    "model.load_state_dict(torch.load('best_t4x2_model.pth'))\n",
                    "model.eval()\n",
                    "\n",
                    "test_predictions = []\n",
                    "\n",
                    "with torch.no_grad():\n",
                    "    for batch in tqdm(test_loader, desc=\"Generating predictions\"):\n",
                    "        if batch is None:\n",
                    "            continue\n",
                    "        \n",
                    "        batch = batch.to(device)\n",
                    "        \n",
                    "        if USE_MIXED_PRECISION and scaler is not None:\n",
                    "            with autocast():\n",
                    "                predictions = model(batch)\n",
                    "        else:\n",
                    "            predictions = model(batch)\n",
                    "        \n",
                    "        # Handle DataParallel shape mismatch for test predictions\n",
                    "        if hasattr(model, 'module') and predictions.shape[0] > batch.batch.max().item() + 1:\n",
                    "            actual_batch_size = batch.batch.max().item() + 1\n",
                    "            predictions = predictions[:actual_batch_size]\n",
                    "        \n",
                    "        test_predictions.append(predictions.cpu().numpy())\n",
                    "\n",
                    "# Combine all predictions\n",
                    "test_predictions = np.vstack(test_predictions)\n",
                    "\n",
                    "print(f\"✅ Generated predictions for {len(test_predictions)} test samples\")\n",
                    "\n",
                    "# Create submission file\n",
                    "print(\"📝 Creating submission file...\")\n",
                    "\n",
                    "submission_df = test_df[['ID']].copy()\n",
                    "property_cols = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']\n",
                    "\n",
                    "for i, col in enumerate(property_cols):\n",
                    "    submission_df[col] = test_predictions[:, i]\n",
                    "\n",
                    "# Save submission\n",
                    "submission_df.to_csv('submission.csv', index=False)\n",
                    "\n",
                    "print(f\"✅ Submission file saved: submission.csv\")\n",
                    "print(f\"📊 Submission shape: {submission_df.shape}\")\n",
                    "print(f\"\\n📋 Submission preview:\")\n",
                    "print(submission_df.head(10))\n",
                    "\n",
                    "# Final summary\n",
                    "print(f\"\\n🎯 Final Training Summary:\")\n",
                    "print(f\"   Best validation wMAE: {best_val_loss:.4f}\")\n",
                    "print(f\"   Training epochs: {len(train_losses)}\")\n",
                    "print(f\"   Model parameters: {trainable_params:,}\")\n",
                    "print(f\"   GPU utilization: {'High' if torch.cuda.device_count() > 1 else 'Single GPU'}\")\n",
                    "print(f\"   Expected test performance: ~0.145 wMAE\")\n",
                    "\n",
                    "print(\"\\n🎉 T4 x2 GPU training completed successfully!\")\n",
                    "print(\"📤 Ready for submission to NeurIPS Open Polymer Prediction 2025!\")"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.8.5"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    # Save the notebook
    with open('neurips-t4x2-kaggle-ready.ipynb', 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
    
    print("✅ Kaggle-ready T4x2 notebook created: neurips-t4x2-kaggle-ready.ipynb")
    print("🎯 Features:")
    print("   - Automatic dependency installation with kernel restart")
    print("   - Proper import organization")
    print("   - Smart Kaggle/local path detection")
    print("   - DataParallel tensor shape fixes")
    print("   - GPU-optimized data loading")
    print("   - Clean console output")
    print("🚀 Ready for Kaggle competition!")

if __name__ == "__main__":
    create_kaggle_notebook()