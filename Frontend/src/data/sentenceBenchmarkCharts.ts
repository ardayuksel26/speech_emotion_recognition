/**
 * Test/TestsResults/sentence_methods_benchmark_results.txt — VOSK segmentasyonu,
 * sentetik cümle senaryosu (15 cümle / duygu sınıfı).
 * Değerler accuracy ve macro-F1 (oran × 100).
 */
export const sentenceVoskSynthetic15 = [
    { name: 'Grad Boost', accStd: 43.33, accRob: 33.33, f1Std: 37.02, f1Rob: 27.8 },
    { name: 'CatBoost', accStd: 35.0, accRob: 36.67, f1Std: 27.37, f1Rob: 30.52 },
    { name: 'XGBoost', accStd: 28.33, accRob: 35.0, f1Std: 24.61, f1Rob: 30.3 },
    { name: 'CNN1D', accStd: 30.0, accRob: 30.0, f1Std: 17.61, f1Rob: 17.61 },
    { name: 'LightGBM', accStd: 25.0, accRob: 30.0, f1Std: 21.59, f1Rob: 26.44 },
    { name: 'MLP', accStd: 25.0, accRob: 21.67, f1Std: 14.71, f1Rob: 18.5 },
    { name: 'SVM', accStd: 25.0, accRob: 25.0, f1Std: 10.0, f1Rob: 10.0 },
    { name: 'KNN', accStd: 21.67, accRob: 23.33, f1Std: 12.68, f1Rob: 14.49 },
    { name: 'RF', accStd: 20.0, accRob: 20.0, f1Std: 15.28, f1Rob: 15.28 },
    { name: 'DNN', accStd: 18.33, accRob: 18.33, f1Std: 13.4, f1Rob: 13.4 },
] as const;

/**
 * Test/TestsResults/sentence_hq_methods_benchmark_results.txt — HQ cümle seti
 * (10 cümle/duygu, daha uzun ve yoğun kelimeli), VOSK segmentasyonu.
 */
export const sentenceVoskHq = [
    { name: 'RF', accStd: 53.75, accRob: 68.75, f1Std: 47.37, f1Rob: 67.51 },
    { name: 'XGBoost', accStd: 65.0, accRob: 61.25, f1Std: 60.78, f1Rob: 56.14 },
    { name: 'CatBoost', accStd: 63.75, accRob: 63.75, f1Std: 59.46, f1Rob: 59.77 },
    { name: 'Grad Boost', accStd: 56.25, accRob: 63.75, f1Std: 47.79, f1Rob: 56.39 },
    { name: 'LightGBM', accStd: 55.0, accRob: 57.5, f1Std: 51.33, f1Rob: 52.53 },
    { name: 'KNN', accStd: 30.0, accRob: 26.25, f1Std: 19.78, f1Rob: 17.64 },
    { name: 'DNN', accStd: 27.5, accRob: 27.5, f1Std: 20.17, f1Rob: 20.17 },
    { name: 'MLP', accStd: 25.0, accRob: 25.0, f1Std: 11.98, f1Rob: 21.8 },
    { name: 'CNN1D', accStd: 25.0, accRob: 25.0, f1Std: 15.34, f1Rob: 15.34 },
    { name: 'SVM', accStd: 25.0, accRob: 25.0, f1Std: 10.0, f1Rob: 10.0 },
] as const;
