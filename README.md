## An End-to-end Speaker Recognition Method Based on Multi-network Stacking

### Usage

You can run the script from the command line and provide arguments for different modes and parameters. The script accepts the following arguments:

#### Command-Line Arguments

- **`--seed`** (optional):
  Sets the random seed. If not provided, it defaults to `42`.
  Example: `--seed 123`
- **`--mode`** (required):
  Specifies which function to run. The available options are:
  - `main_separate`: Runs the `main_separate()` function.
  - `main`: Runs the `main()` function.
  - `valid`: Runs the `valid()` function. Example: `--mode main`
- **`--dim`** (optional):
  Sets the dimension size for the chosen function. The default value is `26`.
  Example: `--dim 30`
- **`--alpha`** (optional):
  Used only for the `main_separate` function. Sets the alpha value. The default is `None`.
  Example: `--alpha 0.5`
- **`--num_labels`** (optional):
  Sets the number of labels. The default value is `10`.
  Example: `--num_labels 5`

#### Example Commands

1. **Run `main_separate` function with custom parameters:**

   ```
   python main.py --mode main_separate --dim 26 --alpha None --num_labels 8 --seed 42
   ```

2. **Run `main` function with default parameters:**

   ```
   python main.py --mode main
   ```

3. **Run `valid` function with a custom seed:**

   ```
   python main.py --mode valid --seed 42 --dim 26 --num_labels 18
   ```

#### Description of Functions

- **`main_separate(dim, alpha, num_labels)`**:
  Runs the main process with separate configurations, including an optional `alpha` parameter.
- **`main(dim, num_labels)`**:
  Runs the main process with specified `dim` and `num_labels`.
- **`valid(dim, num_labels)`**:
  Runs the validation process with specified `dim` and `num_labels`.
