# Datasets

HELM supports **9 datasets** across three domains: remote sensing, medical imaging, and microscopy. Each dataset has a label hierarchy defined in its YAML config under `configs/dataset/`.

## Overview

| Dataset | Domain | Leaf Classes | Total Nodes | Intermediate | Max Depth | Config |
|---------|--------|:------------:|:-----------:|:------------:|:---------:|--------|
| UCM | Remote Sensing | 17 | 30 | 13 | 3 | `dataset=ucm` |
| AID | Remote Sensing | 17 | 30 | 13 | 3 | `dataset=aid` |
| MLRSNet | Remote Sensing | 60 | 95 | 35 | 5 | `dataset=mlrsnet` |
| DFC-15 | Remote Sensing | 8 | 17 | 9 | 3 | `dataset=dfc_15` |
| ChestX-ray8 | Medical | 20 | 28 | 8 | 3 | `dataset=chestxray8` |
| NIH ChestXray | Medical | 15 | 22 | 7 | 3 | `dataset=nihchestxray` |
| PadChest | Medical | 19 | 34 | 15 | 6 | `dataset=padchest` |
| MuReD | Medical | 20 | 34 | 14 | 4 | `dataset=mured` |
| HPA | Microscopy | 28 | 34 | 6 | 2 | `dataset=hpa` |

- **Leaf Classes**: labels the model predicts (original dataset labels)
- **Total Nodes**: leaf + intermediate nodes used in hierarchical mode (`hmlc`)
- **Max Depth**: longest path from a root to a leaf node

---

## Remote Sensing

### UCM (UC Merced Land Use)

Multi-label land use classification from aerial imagery. Images are 256x256 pixels at 1-foot resolution.

- **Source**: HuggingFace (`UC_Merced_LandUse_Multilabel`)
- **Config**: `configs/dataset/ucm.yaml`

```
Artificial Surfaces
├── Urban Fabric
│   ├── buildings
│   └── mobile-home
├── Industrial, Commercial, and Transport Units
│   ├── airplane
│   ├── cars
│   ├── court
│   ├── dock
│   ├── ship
│   └── storage tanks
├── Road and Rail Networks and Associated Land
│   └── pavement
└── Mine, Dump, and Construction Sites
    └── bare-soil
Agricultural Areas
└── Arable Land
    └── field
Forest and Semi-Natural Areas
├── Forests
│   └── trees
└── Shrub and/or Herbaceous Vegetation Associations
    ├── chaparral
    └── grass
Water Bodies
├── Inland Waters
│   └── water
└── Marine Waters
    ├── sea
    └── sand
```

---

### AID (Aerial Image Dataset)

Multi-label aerial scene classification with 600x600 pixel images. Shares the same hierarchy structure as UCM.

- **Source**: HuggingFace (`AID_Multilabel`)
- **Config**: `configs/dataset/aid.yaml`

```
Artificial Surfaces
├── Urban Fabric
│   ├── buildings
│   └── mobile-home
├── Industrial, Commercial, and Transport Units
│   ├── airplane, cars, court, dock, ship, tanks
├── Road and Rail Networks and Associated Land
│   └── pavement
└── Mine, Dump, and Construction Sites
    └── bare-soil
Agricultural Areas
└── Arable Land → field
Forest and Semi-Natural Areas
├── Forests → trees
└── Shrub/Herbaceous → chaparral, grass
Water Bodies
├── Inland Waters → water
└── Marine Waters → sea, sand
```

---

### MLRSNet

The largest and most complex dataset, with 60 classes and a 5-level hierarchy. Contains ~109k satellite images.

- **Source**: HuggingFace (`MLRSNet`)
- **Config**: `configs/dataset/mlrsnet.yaml`

```
Artificial surfaces
├── Urban fabric
│   ├── Continuous urban fabric
│   │   └── dense residential area
│   ├── Discontinuous urban fabric
│   │   ├── sparse residential area
│   │   ├── mobile home
│   │   └── terrace
│   └── Green urban areas
│       └── park
├── Industrial, commercial and transport units
│   ├── Industrial or commercial units
│   │   ├── factory, greenhouse, containers
│   │   ├── tanks, transmission tower, wind turbine
│   ├── Road and rail networks
│   │   ├── bridge, crosswalk, freeway, intersection
│   │   ├── overpass, parking lot, parkway, pavement
│   │   ├── railway, railway station, road, roundabout
│   ├── Port areas
│   │   ├── dock, harbor, ships
│   └── Airports
│       ├── airport, airplane, runway
├── Mine, dump and construction sites
│   └── Mineral extraction sites → bare soil
├── Artificial, non-agricultural vegetated areas
│   └── Sport and leisure facilities
│       ├── baseball diamond, basketball court
│       ├── football field, golf course, stadium
│       ├── swimming pool, tennis court, track, trail
└── Buildings
    ├── buildings, cars
Agricultural areas
└── Arable land
    └── Non-irrigated arable land → field
Forest and semi-natural areas
├── Forests → forest, trees
├── Shrub/herbaceous vegetation
│   └── Sclerophyllous vegetation → chaparral, grass
└── Open spaces with little or no vegetation
    ├── Beaches, dunes, and sand plains → beach, sand
    ├── Bare rock → mountain
    ├── Sparsely vegetated areas → desert
    └── Glaciers and perpetual snow → snow, snowberg
Wetlands → wetland
Water bodies
├── Inland waters
│   ├── Water courses → river
│   └── water bodies → lake, water
└── Marine waters → sea
Natural landforms → gully, island
Atmospheric conditions → cloud
```

---

### DFC-15 (Data Fusion Contest 2015)

Semantic labeling of very high-resolution satellite imagery. The simplest hierarchy with 8 classes.

- **Config**: `configs/dataset/dfc_15.yaml`

```
Artificial surfaces
├── Urban fabric → building
├── Industrial, commercial and transport units
│   ├── clutter
│   └── car
├── Road and rail networks → impervious
└── Port areas → boat
Forest and semi-natural areas
├── Forests → tree
└── Shrub/herbaceous vegetation → vegetation
Water bodies → water
```

---

## Medical Imaging

### ChestX-ray8

Large-scale chest X-ray dataset with 20 pathological findings organized by body system.

- **Config**: `configs/dataset/chestxray8.yaml`

```
Respiratory System Diseases
├── Lung Parenchymal Diseases
│   ├── Infiltration
│   ├── Nodule
│   ├── Mass
│   ├── Consolidation
│   ├── Pneumonia
│   ├── Emphysema
│   ├── Fibrosis
│   └── Atelectasis
└── Pleural and Surrounding Structures
    ├── Effusion
    ├── Pneumothorax
    ├── Pleural Thickening
    └── Subcutaneous Emphysema
Circulatory System Diseases
└── Cardiovascular Conditions
    ├── Cardiomegaly
    ├── Tortuous Aorta
    ├── Calcification of the Aorta
    └── Edema
Digestive and Thoracic Conditions → Hernia
Injury, Poisoning, and Certain Other External Causes
├── Pneumomediastinum
└── Pneumoperitoneum
No Finding → No Finding
```

---

### NIH ChestXray (ChestX-14)

NIH Clinical Center chest X-ray dataset with 15 thoracic disease labels.

- **Config**: `configs/dataset/nihchestxray.yaml`

```
Respiratory System Diseases
├── Lung Parenchymal Diseases
│   ├── Atelectasis, Consolidation, Infiltration
│   ├── Edema, Emphysema, Fibrosis
│   ├── Pneumonia, Nodule, Mass
└── Pleural and Surrounding Structures
    ├── Effusion, Pneumothorax, Pleural_Thickening
Circulatory System Diseases
└── Cardiovascular Conditions → Cardiomegaly
Digestive and Thoracic Conditions → Hernia
No Finding → No Finding
```

---

### PadChest

Spanish multi-label chest X-ray dataset with the **deepest hierarchy** (up to 6 levels).

- **Config**: `configs/dataset/padchest.yaml`

```
normal → normal
abnormal
├── pulmonary abnormality
│   ├── opacity
│   │   ├── infiltrates
│   │   │   ├── interstitial pattern → ground glass pattern
│   │   │   └── alveolar pattern → consolidation
│   │   └── atelectasis
│   │       ├── lobar atelectasis
│   │       └── laminar atelectasis
│   ├── bullas
│   └── pleural thickening
│       ├── apical pleural thickening
│       └── calcified pleural thickening
├── pulmonary nodules and masses
│   ├── mass → pulmonary mass
│   ├── nodule
│   ├── hilar enlargement
│   │   ├── adenopathy
│   │   └── vascular hilar enlargement
│   └── granuloma
├── cardiac abnormality
│   ├── aortic disease
│   │   ├── aortic atheromatosis
│   │   └── aortic elongation
│   │       ├── descendent aortic elongation
│   │       ├── aortic button enlargement
│   │       └── supra aortic elongation
│   └── cardiomegaly
└── bone lesion
```

---

### MuReD (Multimodal Retinal Disease)

Fundus image dataset for multi-label retinal disease classification. Contains ~1.3k images across 20 conditions.

- **Config**: `configs/dataset/mured.yaml`

```
Retinal Disorders
├── Vascular Retinopathies
│   ├── DR    (Diabetic Retinopathy)
│   ├── DN    (Diabetic Nephropathy)
│   ├── BRVO  (Branch Retinal Vein Occlusion)
│   ├── CRVO  (Central Retinal Vein Occlusion)
│   └── HTR   (Hypertensive Retinopathy)
├── Degenerative Retinal Disorders
│   ├── ARMD  (Age-Related Macular Degeneration)
│   ├── Macular Disorders
│   │   ├── MH   (Macular Hole)
│   │   └── CSR  (Central Serous Retinopathy)
│   ├── Retinoschisis and Related Conditions
│   │   ├── RS   (Retinoschisis)
│   │   └── TSLN (Tessellation)
│   └── Choroidal Disorders
│       └── CNV  (Choroidal Neovascularization)
└── Structural Retinal Changes
    ├── LS   (Laser Scars)
    ├── ASR  (Abnormal Subretinal)
    └── CRS  (Chorioretinal Scars)
Optic Nerve and Disc Disorders
├── Optic Disc Anomalies
│   ├── ODC  (Optic Disc Cupping)
│   └── ODP  (Optic Disc Pallor)
├── Optic Disc Edema → ODE
└── Refractive Disorders → MYA (Myopia)
Other Retinal and Optic Nerve Disorders
└── Rare Diseases → OTHER
Normal Retina → NORMAL
```

---

## Microscopy

### HPA (Human Protein Atlas)

Fluorescence microscopy images showing protein subcellular localization patterns. The hierarchy is shallow (2 levels) but has the most leaf classes (28).

- **Source**: Kaggle / HuggingFace
- **Config**: `configs/dataset/hpa.yaml`

```
Nucleus-Related Structures
├── Nucleoplasm
├── Nuclear membrane
├── Nucleoli
├── Nucleoli fibrillar center
├── Nuclear speckles
└── Nuclear bodies
Cytoplasm-Related Structures
├── Cytosol
├── Cytoplasmic bodies
├── Intermediate filaments
├── Actin filaments
└── Focal adhesion sites
Organelles
├── Endoplasmic reticulum
├── Golgi apparatus
├── Mitochondria
├── Peroxisomes
├── Endosomes
└── Lysosomes
Microtubule-Related Structures
├── Microtubules
├── Microtubule ends
├── Cytokinetic bridge
├── Mitotic spindle
├── Microtubule organizing center
└── Centrosome
Membrane-Related Structures
├── Plasma membrane
└── Cell junctions
Other Structures
├── Aggresome
├── Lipid droplets
└── Rods & rings
```

---

## How Hierarchies Work in HELM

Each dataset hierarchy defines a tree of label relationships:

1. **Leaf nodes** are the original dataset labels (what the model predicts)
2. **Intermediate nodes** are parent categories added by HELM
3. In `hmlc` mode, activating a leaf automatically activates all its ancestors
4. The hierarchy graph is used by GraphSAGE to propagate information between related labels

For example, in UCM, if an image contains `airplane`, the extended label vector also activates `Industrial, Commercial, and Transport Units` and `Artificial Surfaces`.

### Adding a New Dataset

1. Create `configs/dataset/your_dataset.yaml`:
```yaml
name: YourDatasetName
folder_name: your_dataset
num_classes: 5  # number of leaf labels

hierarchy:
  Category_A:
    Subcategory_A1:
      leaf_label_1: {}
      leaf_label_2: {}
  Category_B:
    leaf_label_3: {}
    leaf_label_4: {}
    leaf_label_5: {}

leaf_labels:
  - leaf_label_1
  - leaf_label_2
  - leaf_label_3
  - leaf_label_4
  - leaf_label_5
```

2. Create a dataset loader in `data/` (see existing loaders for reference)
3. Register it in `data/utils.py` → `DatasetFactory.create_dataset()`

**Important**: leaf and intermediate node names must be unique. If a leaf has the same name as its parent, append `_` to the parent (e.g., `No Finding_: { No Finding: {} }`).
