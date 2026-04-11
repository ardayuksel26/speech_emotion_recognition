/**
 * Test/TestsResults/sentence_methods_benchmark_results.txt
 * Sentetik cümle: 15 örnek / duygu. Üç segmentasyon kolonu: VAD (Librosa/SentenceProcessor), VOSK, WhisperX.
 */

/** Tek referans sınıflandırıcı: CatBoost (standart) — yöntem etkisini izole eder. */
export const segmentationCatBoostOnly = [
    { name: 'VAD (Librosa)', acc: 36.67, macroF1: 31.58 },
    { name: 'VOSK', acc: 35.0, macroF1: 27.37 },
    { name: 'WhisperX', acc: 36.67, macroF1: 31.18 },
] as const;

/** Dört tabüler model ortalaması: CatBoost, XGBoost, LightGBM, GradBoost (standart). */
export const segmentationQuadMean = [
    { name: 'VAD (Librosa)', acc: 34.58, macroF1: 29.76 },
    { name: 'VOSK', acc: 32.92, macroF1: 27.65 },
    { name: 'WhisperX', acc: 36.67, macroF1: 31.46 },
] as const;

/** Şematik kısa-süre enerji zarfı (normalize); VAD sezgisini görselleştirir — ölçüm verisi değildir. */
export const vadEnergyIllustration = Array.from({ length: 56 }, (_, i) => {
    const c = 28;
    const envelope = Math.exp(-((i - c) * (i - c)) / 90);
    return { frame: i, energy: 0.04 + 0.88 * envelope };
});
