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


### **3.2. data_generation.py**
- **Function**: Generates dataset based on customizable parameters.
- **Purpose**: Allows users to set the number of textures, spring coefficients, and random assignment of shapes.


### **3.3. main.py**
- **Function**: Main script to generate the dataset.
- **Purpose**: Serves as the entry point for data generation.


### **3.4. model_training.py**
- **Function**: Trains the UNet model using generated data.
- **Purpose**: Runs the training pipeline for predicting spring parameters.


### **3.5. models.py**
- **Function**: Contains the UNet model implementation.
- **Purpose**: Defines the neural network structure to predict spring matrices from input texture images.


### **3.6. taichiSimulation.py**
- **Function**: Updates jelly particle states based on physical parameters.
- **Purpose**: Runs Taichi-based simulations to model jelly movements using spring indices, rest lengths, and forces.


### **3.7. utils.py**
- **Function**: Contains helper functions.
- **Purpose**: Handles logging, trial setup, and visualization of predicted vs ground truth spring coefficients.

