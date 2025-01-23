# Variational AutoEncoders with Non-Normal Latent space for Drug Response prediction (NNL-VAEDR)

## Overview
This project implements a cancer drug response prediction system using an Autoencoder-based classifier. The system processes input data with various reparameterization methods and supports multiple cancer types.

## Requirements
Ensure the following dependencies are installed:

## Project Structure
```
.
├── model.py                # Defines the AutoencoderClassifier model
├── train.py                # Handles the training and evaluation process
├── Load.py                 # Loads the dataset
├── test.py                 # Handles testing of the model
├── main.py                 # Entry point of the application
```

## Usage
### Running the Script
To execute the script, use the following command:

```bash
python main.py --cancer_type <type> --reparam_method <method> [--is_test_run]
```

### Arguments
- `--cancer_type`: Specifies the type of cancer to analyze. Choices:
  - `breast`
  - `cervical`
  - `uterine`
  
  Example:
  ```
  --cancer_type cervical
  ```

- `--reparam_method`: Specifies the reparameterization method to use. Choices:
  - `normal`
  - `gamma`
  - `log_normal`
  - `uniform`
  - `log_gamma`

  Example:
  ```
  --reparam_method log_gamma
  ```

- `--is_test_run`: (Optional) Set this flag to run the model in test mode. Default is `False`.
  
  Example:
  ```
  --is_test_run
  ```
### Model Training and Testing
- **Training**
  The `train_and_evaluate` function is used to train the model on the provided dataset.
- **Testing**
  The `test` function is executed when the `--is_test_run` flag is provided.

## Example Usage
### Train the Model on Cervical Cancer Data
```bash
python main.py --cancer_type cervical --reparam_method log_gamma
```

### Test the Model with Breast Cancer Data
```bash
python main.py --cancer_type breast --reparam_method normal --is_test_run
```

## Output
During training or testing, the system will output:
- Model performance metrics (e.g., loss, accuracy)
- Logs for different processing steps
