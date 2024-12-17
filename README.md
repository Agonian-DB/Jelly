# 1. ModelWIthoutTaichi
- HierarchicalTransformer implementation 
- 3D Unet implementation 
- Simple model test Taichi's inefficiency

# 2. SimulationNew
- taichi_test: new implementations of system simulation
- taichi_record_simple: record of using taichi to train the model.


# 3. ModelWithTaichi

### **3.1. TextureDataset.py**
- **Function**: Acts as the data loader for the project.
- **Purpose**: Loads the texture images and corresponding particle position data for training.
- **How to Use**: 
  - Pass the directory containing generated data to initialize the loader.
  - Handles `.png` images (input) and `.npy` files (position information).

### **3.2. data_generation.py**
- **Function**: Generates dataset based on customizable parameters.
- **Purpose**: Allows users to set the number of textures, spring coefficients, and random assignment of shapes.
- **How to Use**:
  - Import `DataGenerator` and define physical and simulation parameters.
  - Call `generate_dataset()` to create data files.

### **3.3. main.py**
- **Function**: Main script to generate the dataset.
- **Purpose**: Serves as the entry point for data generation.
- **How to Use**:
   Run this script to generate the dataset with default or customized physical parameters:
   ```bash
   python main.py
   ```

### **3.4. model_training.py**
- **Function**: Trains the UNet model using generated data.
- **Purpose**: Runs the training pipeline for predicting spring parameters.
- **How to Use**:
   - Input the directory containing generated data.
   - Defines the UNet architecture and optimizes the model with simulated ground truth data.
   - Logs training loss and saves trained models periodically.

### **3.5. models.py**
- **Function**: Contains the UNet model implementation.
- **Purpose**: Defines the neural network structure to predict spring matrices from input texture images.
- **How to Use**:
   - The `UNet` class can be directly imported and instantiated for model training.

### **3.6. taichiSimulation.py**
- **Function**: Updates jelly particle states based on physical parameters.
- **Purpose**: Runs Taichi-based simulations to model jelly movements using spring indices, rest lengths, and forces.
- **How to Use**:
   - Initialize with particle positions, spring indices, and simulation parameters.
   - Simulates movement and updates positions frame by frame.

### **3.7. utils.py**
- **Function**: Contains helper functions.
- **Purpose**: Handles logging, trial setup, and visualization of predicted vs ground truth spring coefficients.

