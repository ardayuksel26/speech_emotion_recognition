/**
 * emotion_performance_metrics.txt / emotion_f1_scores.png
 * Per-class precision, recall, F1 for the Mastermind pipeline (sentence-level).
 */
export const emotionPerformanceMetrics = [
    { emotion: 'Angry',  precision: 100.0, recall: 70.0,  f1: 82.4 },
    { emotion: 'Calm',   precision: 64.0,  recall: 80.0,  f1: 71.1 },
    { emotion: 'Happy',  precision: 53.3,  recall: 80.0,  f1: 64.0 },
    { emotion: 'Sad',    precision: 50.0,  recall: 26.3,  f1: 34.5 },
];

/**
 * mastermind_confusion_matrix_result.txt
 * Rows = actual, Cols = predicted  [Angry, Calm, Happy, Sad]
 */
export const confusionMatrix = {
    labels: ['Angry', 'Calm', 'Happy', 'Sad'],
    matrix: [
        [14, 2, 4, 0],   // Actual Angry
        [0, 16, 2, 2],   // Actual Calm
        [0, 1, 16, 3],   // Actual Happy
        [0, 6, 8, 5],    // Actual Sad
    ],
};

/**
 * validation_robust_metrics.txt / validation_robust_f1_scores.png
 * Robust (noisy) validation: per-class F1 under augmented conditions.
 */
export const validationRobustMetrics = [
    { emotion: 'Angry',  f1_standard: 82.4, f1_robust: 74.1 },
    { emotion: 'Calm',   f1_standard: 71.1, f1_robust: 63.5 },
    { emotion: 'Happy',  f1_standard: 64.0, f1_robust: 57.2 },
    { emotion: 'Sad',    f1_standard: 34.5, f1_robust: 28.9 },
];
