# 23 bar truss surrogate model

## Project structure

```
.
├── fem_interfaces
│   └── kratos
│       └── Kratos_Struct_Linear_Sudret_Truss.py
├── fem_utilities
│   └── FEM_matrices.py
├── neural_net
│   ├── data_utilities.py
│   ├── loss_functions.py
│   ├── networks.py
│   └── training.py
├── sim_parameters
│   ├── ProjectParameters.json
│   ├── StructuralMaterials.json
│   └── sudret_truss.mdpa
├── utilities
│   └── plot_utilities.py
├── hyper_params.json
├── README.md
└── truss23_femnn.ipynb
```
