# Comprehensive Analysis of the Intel-XPU SEM Hybrid Benchmark

## 1. Scope

- Models analysed: **4**.
- Test observations per model: **6000 to 6000**.
- Tasks: discrete consolidation-stress classification, continuous stress regression, and four stress-derived engineering-property estimates.
- Uncertainty: paired non-parametric 95% bootstrap intervals.
- Graphics: 600 dpi PNG and vector PDF.

## 2. Executive summary

- Best classification: **ResNet-50**, macro F1 0.6514, balanced accuracy 0.6713, accuracy 0.6713, MCC 0.6171.
- Best continuous stress model by log-domain R²: **ResNet-50**, log₁₀ R² 0.7595, multiplicative error factor 1.97×.
- Best mean derived-property R²: **ResNet-50**, 0.4838.
- Recorded training time across models: **776.8 min**.

## 3. Classification comparison

|Rank|Model|Accuracy|Balanced accuracy|Macro F1|MCC|Macro F1 95% CI|ECE|Log loss|
|---:|---|---:|---:|---:|---:|---:|---:|---:|
|1|ResNet-50|0.6713|0.6713|0.6514|0.6171|0.6405 to 0.6616|0.2865|2.9577|
|2|ViT-B/16|0.6425|0.6425|0.6180|0.5962|0.6072 to 0.6284|0.3284|4.0469|
|3|MobileNetV2|0.6070|0.6070|0.5867|0.5511|0.5758 to 0.5965|0.3566|3.8975|
|4|EfficientNet-B0|0.6128|0.6128|0.5813|0.5713|0.5706 to 0.5913|0.3707|6.6565|

## 4. Continuous-stress regression

|Model|Physical R²|Physical RMSE (kPa)|MAE (kPa)|Bias (kPa)|log₁₀ R²|Multiplicative factor|
|---|---:|---:|---:|---:|---:|---:|
|ResNet-50|0.9545|439.89|208.86|0.32|0.7595|1.97×|
|ViT-B/16|0.9766|315.34|191.46|115.93|0.7363|2.04×|
|EfficientNet-B0|0.9233|571.29|296.66|76.05|0.5665|2.49×|
|MobileNetV2|0.7314|1068.96|502.32|-156.16|0.5582|2.51×|

## 5. Derived engineering properties

|Model|Mean R²|Mean relative RMSE (%)|Mean raw MAE*|
|---|---:|---:|---:|
|ResNet-50|0.4838|53.88|0.0141|
|ViT-B/16|0.4421|55.98|0.0146|
|MobileNetV2|0.2355|64.81|0.0198|
|EfficientNet-B0|0.2295|65.06|0.0191|

*Mean raw MAE combines differently scaled physical quantities and is descriptive only.

Target-specific leaders:
- **Void ratio, e**: ResNet-50, R²=0.7132, RMSE=0.0954, MAE=0.0563, bias=-0.0351.
- **Permeability, k (m/s)**: ResNet-50, R²=0.2507, RMSE=0.0000, MAE=0.0000, bias=-0.0000.
- **Coefficient of consolidation, cᵥ (m²/s)**: ResNet-50, R²=0.2832, RMSE=0.0000, MAE=0.0000, bias=-0.0000.
- **Volume compressibility, mᵥ (m²/kN)**: ResNet-50, R²=0.6881, RMSE=0.0000, MAE=0.0000, bias=-0.0000.

## 6. Class-level diagnostics

- **ResNet-50**: strongest stress level 6000 kPa (F1=0.9930); weakest 400 kPa (F1=0.3856).
- **ViT-B/16**: strongest stress level 2000 kPa (F1=0.9995); weakest 100 kPa (F1=0.3226).
- **MobileNetV2**: strongest stress level 2000 kPa (F1=0.9980); weakest 100 kPa (F1=0.2348).
- **EfficientNet-B0**: strongest stress level 2000 kPa (F1=1.0000); weakest 100 kPa (F1=0.1576).

## 7. Training and generalisation

|Model|Epochs|Best epoch|Best validation loss|Final train loss|Final validation loss|Final gap|Training time (min)|
|---|---:|---:|---:|---:|---:|---:|---:|
|MobileNetV2|11|6|2.1994|0.0079|3.5580|3.5501|201.1|
|ViT-B/16|12|1|2.9146|0.0076|3.8723|3.8647|231.2|
|EfficientNet-B0|11|3|2.9278|0.0200|4.4730|4.4530|193.7|
|ResNet-50|11|1|3.7757|0.0040|5.0964|5.0924|150.7|

## 8. Scientific interpretation

- All engineering-property targets fixed within each measured stress level: **True**.
- The property outputs are deterministic stress-conditioned values produced from the supplied interpolation table.
- Their apparent predictive performance is therefore inherited from continuous-stress prediction and should not be presented as independent specimen-level property measurement.
- Independent property prediction would require directly measured property targets with genuine within-stress variation.
- Continuous stress spans orders of magnitude, so log-domain R², log RMSE, and multiplicative error factor should accompany physical-domain RMSE and MAE.
- Balanced accuracy and macro F1 should be emphasised if stress-level frequencies are unequal.
- Calibration statistics describe whether confidence values are trustworthy, not only whether predicted classes are correct.
- Conclusions remain limited to the held-out dataset. External batches, instruments, preparation conditions, and magnifications require validation.

## 9. Manuscript-ready outputs

- `figure_01_model_performance_overview`: compact task-level comparison.
- `figure_02_normalised_confusion_matrices`: stress-class error patterns.
- `figure_03_continuous_stress_parity`: measured versus predicted stress.
- `figure_04` and `figure_05`: target-specific engineering-property heatmaps.
- `figure_06_stress_error_by_level`: heteroscedasticity across stress levels.
- `figure_07_merged_training_curves`: convergence and fine-tuning behaviour.
- `figure_08_reliability_diagram`: probability calibration.
- `figure_09_performance_cost`: macro F1 versus training time.
- `table_01` to `table_09`: reusable machine-readable statistics.