/**
 * emotion_performance_metrics.txt / emotion_f1_scores.png
 * Per-class precision, recall, F1 for the Master Ensemble pipeline (sentence-level).
 */
export const emotionPerformanceMetrics = [
    { emotion: 'Angry',  precision: 91.3, recall: 78.8,  f1: 84.6 },
    { emotion: 'Calm',   precision: 69.6, recall: 80.0,  f1: 74.4 },
    { emotion: 'Happy',  precision: 81.9, recall: 73.8,  f1: 77.6 },
    { emotion: 'Sad',    precision: 83.9, recall: 91.3,  f1: 87.4 },
];

/**
 * mastermind_confusion_matrix_result.txt
 * Rows = actual, Cols = predicted  [Angry, Calm, Happy, Sad]
 */
export const confusionMatrix = {
    labels: ['Angry', 'Calm', 'Happy', 'Sad'],
    matrix: [
        [63, 8, 9, 0],   // Actual Angry
        [0, 64, 4, 12],  // Actual Calm
        [6, 13, 59, 2],  // Actual Happy
        [0, 7, 0, 73],   // Actual Sad
    ],
};

/**
 * 8. Bölüm (Deneysel Modeller) için Data Seti
 * %30 ve üzeri başarıya sahip olan test sonuçları (Yüksekten Düşüğe).
 */
export const experimentalModelsData = [
    {
        name: 'LightGBM (Models_2)',
        accuracy: 90.94,
        descriptionKey: 'tech_model_desc_0',
        metrics: [
            { emotion: 'Angry', precision: 83.0, recall: 100.0, f1: 91.0 },
            { emotion: 'Calm', precision: 100.0, recall: 86.0, f1: 93.0 },
            { emotion: 'Happy', precision: 100.0, recall: 86.0, f1: 93.0 },
            { emotion: 'Sad', precision: 85.0, recall: 91.0, f1: 88.0 },
        ]
    },
    {
        name: 'XGBoost (Models_2)',
        accuracy: 90.31,
        descriptionKey: 'tech_model_desc_1',
        metrics: [
            { emotion: 'Angry', precision: 78.0, recall: 100.0, f1: 88.0 },
            { emotion: 'Calm', precision: 100.0, recall: 88.0, f1: 93.0 },
            { emotion: 'Happy', precision: 99.0, recall: 82.0, f1: 90.0 },
            { emotion: 'Sad', precision: 90.0, recall: 91.0, f1: 91.0 },
        ]
    },
    {
        name: 'GradientBoosting (Models_2)',
        accuracy: 87.81,
        descriptionKey: 'tech_model_desc_2',
        metrics: [
            { emotion: 'Angry', precision: 83.0, recall: 99.0, f1: 90.0 },
            { emotion: 'Calm', precision: 100.0, recall: 82.0, f1: 90.0 },
            { emotion: 'Happy', precision: 97.0, recall: 75.0, f1: 85.0 },
            { emotion: 'Sad', precision: 78.0, recall: 95.0, f1: 86.0 },
        ]
    },
    {
        name: 'CatBoost (Models_2)',
        accuracy: 84.38,
        descriptionKey: 'tech_model_desc_3',
        metrics: [
            { emotion: 'Angry', precision: 72.0, recall: 100.0, f1: 84.0 },
            { emotion: 'Calm', precision: 100.0, recall: 72.0, f1: 84.0 },
            { emotion: 'Happy', precision: 91.0, recall: 84.0, f1: 87.0 },
            { emotion: 'Sad', precision: 84.0, recall: 81.0, f1: 83.0 },
        ]
    },
    {
        name: 'HuBERT (Huggingface)',
        accuracy: 82.81,
        descriptionKey: 'tech_model_desc_4',
        metrics: [
            { emotion: 'Angry', precision: 100.0, recall: 71.0, f1: 83.0 },
            { emotion: 'Calm', precision: 63.0, recall: 91.0, f1: 75.0 },
            { emotion: 'Happy', precision: 92.0, recall: 84.0, f1: 88.0 },
            { emotion: 'Sad', precision: 91.0, recall: 85.0, f1: 88.0 },
        ]
    },
    {
        name: 'RandomForest (Models_2)',
        accuracy: 75.62,
        descriptionKey: 'tech_model_desc_5',
        metrics: [
            { emotion: 'Angry', precision: 58.0, recall: 100.0, f1: 73.0 },
            { emotion: 'Calm', precision: 100.0, recall: 50.0, f1: 67.0 },
            { emotion: 'Happy', precision: 97.0, recall: 79.0, f1: 87.0 },
            { emotion: 'Sad', precision: 77.0, recall: 74.0, f1: 75.0 },
        ]
    },
    {
        name: 'GradientBoosting_Robust (Models)',
        accuracy: 70.31,
        descriptionKey: 'tech_model_desc_6',
        metrics: [
            { emotion: 'Angry', precision: 59.0, recall: 99.0, f1: 74.0 },
            { emotion: 'Calm', precision: 76.0, recall: 75.0, f1: 75.0 },
            { emotion: 'Happy', precision: 77.0, recall: 76.0, f1: 77.0 },
            { emotion: 'Sad', precision: 86.0, recall: 31.0, f1: 46.0 },
        ]
    },
    {
        name: 'XGBoost_Robust (Models)',
        accuracy: 69.69,
        descriptionKey: 'tech_model_desc_7',
        metrics: [
            { emotion: 'Angry', precision: 76.0, recall: 95.0, f1: 84.0 },
            { emotion: 'Calm', precision: 56.0, recall: 80.0, f1: 66.0 },
            { emotion: 'Happy', precision: 80.0, recall: 64.0, f1: 71.0 },
            { emotion: 'Sad', precision: 76.0, recall: 40.0, f1: 52.0 },
        ]
    },
    {
        name: 'LightGBM_Robust (Models)',
        accuracy: 68.44,
        descriptionKey: 'tech_model_desc_8',
        metrics: [
            { emotion: 'Angry', precision: 65.0, recall: 95.0, f1: 77.0 },
            { emotion: 'Calm', precision: 65.0, recall: 80.0, f1: 72.0 },
            { emotion: 'Happy', precision: 80.0, recall: 59.0, f1: 68.0 },
            { emotion: 'Sad', precision: 71.0, recall: 40.0, f1: 51.0 },
        ]
    },
    {
        name: 'XGBoost (Models)',
        accuracy: 65.62,
        descriptionKey: 'tech_model_desc_9',
        metrics: [
            { emotion: 'Angry', precision: 63.0, recall: 97.0, f1: 77.0 },
            { emotion: 'Calm', precision: 57.0, recall: 70.0, f1: 63.0 },
            { emotion: 'Happy', precision: 87.0, recall: 51.0, f1: 65.0 },
            { emotion: 'Sad', precision: 67.0, recall: 44.0, f1: 53.0 },
        ]
    },
    {
        name: 'Random Forest_Robust (Models)',
        accuracy: 63.75,
        descriptionKey: 'tech_model_desc_10',
        metrics: [
            { emotion: 'Angry', precision: 62.0, recall: 90.0, f1: 73.0 },
            { emotion: 'Calm', precision: 52.0, recall: 57.0, f1: 54.0 },
            { emotion: 'Happy', precision: 100.0, recall: 39.0, f1: 56.0 },
            { emotion: 'Sad', precision: 65.0, recall: 69.0, f1: 67.0 },
        ]
    },
    {
        name: 'GradientBoosting (Models)',
        accuracy: 60.62,
        descriptionKey: 'tech_model_desc_11',
        metrics: [
            { emotion: 'Angry', precision: 64.0, recall: 97.0, f1: 77.0 },
            { emotion: 'Calm', precision: 61.0, recall: 45.0, f1: 52.0 },
            { emotion: 'Happy', precision: 54.0, recall: 84.0, f1: 66.0 },
            { emotion: 'Sad', precision: 81.0, recall: 16.0, f1: 27.0 },
        ]
    },
    {
        name: 'LightGBM (Models)',
        accuracy: 53.75,
        descriptionKey: 'tech_model_desc_12',
        metrics: [
            { emotion: 'Angry', precision: 58.0, recall: 90.0, f1: 71.0 },
            { emotion: 'Calm', precision: 43.0, recall: 86.0, f1: 57.0 },
            { emotion: 'Happy', precision: 92.0, recall: 15.0, f1: 26.0 },
            { emotion: 'Sad', precision: 83.0, recall: 24.0, f1: 37.0 },
        ]
    },
    {
        name: 'Random Forest (Models)',
        accuracy: 50.31,
        descriptionKey: 'tech_model_desc_13',
        metrics: [
            { emotion: 'Angry', precision: 42.0, recall: 88.0, f1: 57.0 },
            { emotion: 'Calm', precision: 45.0, recall: 56.0, f1: 50.0 },
            { emotion: 'Happy', precision: 100.0, recall: 24.0, f1: 38.0 },
            { emotion: 'Sad', precision: 77.0, recall: 34.0, f1: 47.0 },
        ]
    },
    {
        name: 'CatBoost_Robust (Models)',
        accuracy: 46.25,
        descriptionKey: 'tech_model_desc_14',
        metrics: [
            { emotion: 'Angry', precision: 37.0, recall: 99.0, f1: 54.0 },
            { emotion: 'Calm', precision: 81.0, recall: 38.0, f1: 51.0 },
            { emotion: 'Happy', precision: 45.0, recall: 28.0, f1: 34.0 },
            { emotion: 'Sad', precision: 89.0, recall: 21.0, f1: 34.0 },
        ]
    },
    {
        name: 'MLP_Robust (Models)',
        accuracy: 44.38,
        descriptionKey: 'tech_model_desc_15',
        metrics: [
            { emotion: 'Angry', precision: 41.0, recall: 95.0, f1: 57.0 },
            { emotion: 'Calm', precision: 39.0, recall: 49.0, f1: 43.0 },
            { emotion: 'Happy', precision: 77.0, recall: 30.0, f1: 43.0 },
            { emotion: 'Sad', precision: 100.0, recall: 4.0, f1: 7.0 },
        ]
    },
    {
        name: 'CatBoost (Models)',
        accuracy: 43.12,
        descriptionKey: 'tech_model_desc_16',
        metrics: [
            { emotion: 'Angry', precision: 33.0, recall: 100.0, f1: 50.0 },
            { emotion: 'Calm', precision: 82.0, recall: 34.0, f1: 48.0 },
            { emotion: 'Happy', precision: 60.0, recall: 26.0, f1: 37.0 },
            { emotion: 'Sad', precision: 91.0, recall: 12.0, f1: 22.0 },
        ]
    },
    {
        name: 'Random Forest (Sentence_Models)',
        accuracy: 40.94,
        descriptionKey: 'tech_model_desc_17',
        metrics: [
            { emotion: 'Angry', precision: 34.0, recall: 96.0, f1: 50.0 },
            { emotion: 'Calm', precision: 56.0, recall: 11.0, f1: 19.0 },
            { emotion: 'Happy', precision: 88.0, recall: 19.0, f1: 31.0 },
            { emotion: 'Sad', precision: 48.0, recall: 38.0, f1: 42.0 },
        ]
    },
];