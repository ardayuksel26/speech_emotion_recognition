// Auto-migrated from benchmark_result.txt and benchmark_robust_results.txt
// Represents Kelime Bazlı (Word-Level) Model Accuracy Scores in 5-Fold Cross Validation

export const wordBenchmarkClean = [
    { name: "CatBoost", Standard: 95.97 },
    { name: "LightGBM", Standard: 95.85 },
    { name: "XGBoost", Standard: 94.12 },
    { name: "Grad Boost", Standard: 92.68 },
    { name: "Random Forest", Standard: 87.55 },
    { name: "SVM", Standard: 86.57 },
    { name: "MLP (DNN)", Standard: 86.22 },
    { name: "k-NN", Standard: 71.41 }
];

export const wordBenchmarkRobust = [
    { name: "CatBoost", Robust: 98.10 },
    { name: "LightGBM", Robust: 96.37 },
    { name: "XGBoost", Robust: 95.45 },
    { name: "Grad Boost", Robust: 92.56 },
    { name: "Random Forest", Robust: 88.88 },
    { name: "SVM", Robust: 88.36 },
    { name: "MLP (DNN)", Robust: 85.88 },
    { name: "k-NN", Robust: 69.05 }
];

// Combined dataset for multi-bar charts
export const combinedWordBenchmarks = wordBenchmarkClean.map(cleanData => {
    const robustMatch = wordBenchmarkRobust.find(r => r.name === cleanData.name);
    return {
        name: cleanData.name,
        Standard_Accuracy: cleanData.Standard,
        Robust_Accuracy: robustMatch ? robustMatch.Robust : null
    };
});
