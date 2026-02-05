# Model Selection Guide

This guide helps you choose the right model for your intrusion detection task.

## Quick Reference Table

| Model | Best For | Speed | Accuracy | Memory | Multi-Class |
|-------|----------|-------|----------|--------|-------------|
| **Naive Bayes** | Very large datasets, quick baseline | ⚡⚡⚡ | ⭐⭐ | ⚡⚡⚡ | ✓ |
| **Logistic Regression** | Interpretable results, linear relationships | ⚡⚡⚡ | ⭐⭐ | ⚡⚡⚡ | ✓ |
| **Random Forest** | General purpose, feature importance | ⚡⚡ | ⭐⭐⭐ | ⚡⚡ | ✓ |
| **XGBoost** | High accuracy, structured data | ⚡⚡ | ⭐⭐⭐⭐ | ⚡⚡ | ✓ |
| **LightGBM** | Large datasets (3M+ samples) | ⚡⚡⚡ | ⭐⭐⭐⭐ | ⚡⚡⚡ | ✓ |
| **CNN** | Complex patterns, spatial features | ⚡ | ⭐⭐⭐⭐ | ⚡ | ✓ |

Legend: ⚡ = Resource efficiency/Speed, ⭐ = Typical accuracy

## When to Use Each Model

### Naive Bayes
**Use when:**
- You have millions of samples and need fast training
- You want a quick baseline model
- Features are relatively independent
- Memory is constrained

**Avoid when:**
- Features have strong correlations
- You need the highest possible accuracy
- Dataset is small (< 1000 samples)

### Logistic Regression
**Use when:**
- You need interpretable results (coefficients)
- Linear relationships between features and target
- You want fast prediction on large datasets
- Baseline comparison model

**Avoid when:**
- Non-linear relationships dominate
- You need the highest accuracy
- Features have complex interactions

### Random Forest
**Use when:**
- General-purpose classification task
- You need feature importance rankings
- Robust to overfitting is required
- Mixed feature types (continuous + categorical)

**Avoid when:**
- Dataset is extremely large (> 1M samples)
- Real-time prediction speed is critical
- Memory is very limited

### XGBoost
**Use when:**
- You need high accuracy
- Dataset size is moderate (< 1M samples)
- Feature engineering is complete
- You want feature importance

**Avoid when:**
- Dataset is extremely large (use LightGBM instead)
- Very limited computational resources
- Real-time training updates needed

### LightGBM
**Use when:**
- Dataset is very large (1M+ samples)
- You need both speed and accuracy
- Multi-class classification (5+ classes)
- Memory efficiency is important
- Feature importance analysis needed

**Avoid when:**
- Dataset is very small (< 1000 samples)
- Extreme overfitting sensitivity

### CNN
**Use when:**
- Features have spatial or sequential relationships
- You need to learn complex patterns
- Sufficient training data available
- GPU acceleration is available

**Avoid when:**
- Training time is critical constraint
- Interpretability is required
- Dataset is small
- No GPU available and speed is important

## Performance on Sample Data

### Binary Classification (5K samples, 50 features)
```
Random Forest:  87.7% accuracy
XGBoost:        92.5% accuracy
CNN:            91.0% accuracy
```

### Multi-Class (10K samples, 50 features, 5 classes)
```
Naive Bayes:         55.8% accuracy (0.01s training)
Logistic Regression: 54.9% accuracy (0.37s training)
LightGBM:           78.5% accuracy (3.23s training)
```

## Recommended Combinations

### For Production System (Large Dataset)
1. **Primary**: LightGBM - Best balance of speed and accuracy
2. **Fast Baseline**: Naive Bayes - Quick deployment, real-time capable
3. **Fallback**: Random Forest - Robust general-purpose option

### For Research/Analysis
1. **High Accuracy**: XGBoost or LightGBM
2. **Interpretability**: Logistic Regression (for linear insights)
3. **Deep Learning**: CNN (if sufficient data and GPU)

### For Resource-Constrained Environment
1. **Primary**: Naive Bayes - Fastest and most memory-efficient
2. **Secondary**: Logistic Regression - Good speed-accuracy tradeoff
3. **Avoid**: CNN - Too resource-intensive

## Hyperparameter Tuning Priority

If you have limited time for tuning, prioritize in this order:

1. **LightGBM**: `num_leaves`, `learning_rate`, `n_estimators`
2. **XGBoost**: `max_depth`, `learning_rate`, `n_estimators`
3. **Random Forest**: `n_estimators`, `max_depth`
4. **CNN**: `learning_rate`, `batch_size`, `epochs`
5. **Logistic Regression**: `C`, `solver`
6. **Naive Bayes**: `var_smoothing` (minimal tuning needed)

## Training Time Estimates (10K samples)

- Naive Bayes: < 1 second
- Logistic Regression: < 1 second
- Random Forest: 1-5 seconds
- LightGBM: 2-5 seconds
- XGBoost: 5-15 seconds
- CNN: 30-120 seconds (depends on GPU)

## Model File Sizes (Typical)

- Naive Bayes: Very small (< 1 MB)
- Logistic Regression: Small (< 10 MB)
- Random Forest: Large (10-100 MB)
- XGBoost: Medium (5-50 MB)
- LightGBM: Medium (5-50 MB)
- CNN: Large (10-100 MB)

## Conclusion

**For most intrusion detection tasks with large datasets (3M+ samples):**
- Start with **LightGBM** for the best overall performance
- Use **Naive Bayes** as a fast baseline
- Consider **CNN** if you have GPU and time for deep learning

**For smaller datasets or quick experiments:**
- Start with **Random Forest** for reliability
- Try **XGBoost** for maximum accuracy
- Use **Logistic Regression** for interpretability
