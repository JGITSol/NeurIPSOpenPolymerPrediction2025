# =============================================================================
# FINAL INDEX FIX - Replace the PolymerDataset class with this fixed version
# =============================================================================

class PolymerDataset(Dataset):
    def __init__(self, df, is_test=False):
        self.df = df.reset_index(drop=True)  # Reset indices first!
        self.is_test = is_test
        self.property_cols = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
        
        print(f"Processing {len(self.df)} samples...")
        self.graphs = []
        self.valid_data = []  # Store valid rows directly instead of indices
        
        for idx, row in tqdm(self.df.iterrows(), total=len(self.df)):
            graph = smiles_to_graph(row['SMILES'])
            if graph is not None:
                self.graphs.append(graph)
                self.valid_data.append(row)  # Store the actual row data
        
        # Create new dataframe from valid data
        if self.valid_data:
            self.df = pd.DataFrame(self.valid_data).reset_index(drop=True)
        else:
            self.df = pd.DataFrame(columns=self.df.columns)
        
        print(f"Valid samples: {len(self.graphs)}")
    
    def __len__(self):
        return len(self.graphs)
    
    def __getitem__(self, idx):
        graph = self.graphs[idx].clone()
        
        if not self.is_test:
            row = self.df.iloc[idx]
            targets, masks = [], []
            
            for col in self.property_cols:
                if pd.notna(row[col]):
                    targets.append(float(row[col]))
                    masks.append(1.0)
                else:
                    targets.append(0.0)
                    masks.append(0.0)
            
            # CRITICAL: Ensure proper tensor shapes (1D tensors)
            graph.y = torch.tensor(targets, dtype=torch.float)  # Shape: (5,)
            graph.mask = torch.tensor(masks, dtype=torch.float)  # Shape: (5,)
        
        return graph

print("✅ Fixed PolymerDataset class - no more index errors!")