import React, { useState, useRef, useEffect } from 'react';
import { FaDownload, FaFileCsv, FaFileCode } from 'react-icons/fa';
import { motion, AnimatePresence } from 'framer-motion';
import { AnalysisResult } from '../../types';
import { exportToJSON, exportToCSV } from '../../utils/exportUtils';
import { useTranslation } from 'react-i18next';
import clsx from 'clsx';

interface ExportButtonProps {
    result: AnalysisResult;
    className?: string;
}

export const ExportButton: React.FC<ExportButtonProps> = ({ result, className }) => {
    const { t } = useTranslation();
    const [isOpen, setIsOpen] = useState(false);
    const containerRef = useRef<HTMLDivElement>(null);

    // Close dropdown when clicking outside
    useEffect(() => {
        const handleClickOutside = (event: MouseEvent) => {
            if (containerRef.current && !containerRef.current.contains(event.target as Node)) {
                setIsOpen(false);
            }
        };
        document.addEventListener('mousedown', handleClickOutside);
        return () => document.removeEventListener('mousedown', handleClickOutside);
    }, []);

    const handleExport = (type: 'json' | 'csv') => {
        if (type === 'json') {
            exportToJSON(result);
        } else {
            exportToCSV(result);
        }
        setIsOpen(false);
    };

    return (
        <div className={clsx("relative", className)} ref={containerRef}>
            <button
                onClick={() => setIsOpen(!isOpen)}
                className={clsx(
                    "p-2 rounded-full transition-all duration-200",
                    "hover:bg-gray-100 dark:hover:bg-gray-700",
                    isOpen ? "bg-gray-100 dark:bg-gray-700 text-indigo-500" : "text-gray-500 dark:text-gray-400"
                )}
                title={t('export_results') || "Export Results"}
                aria-label="Export results"
            >
                <FaDownload className="w-5 h-5" />
            </button>

            <AnimatePresence>
                {isOpen && (
                    <motion.div
                        initial={{ opacity: 0, y: 10, scale: 0.95 }}
                        animate={{ opacity: 1, y: 0, scale: 1 }}
                        exit={{ opacity: 0, y: 10, scale: 0.95 }}
                        transition={{ duration: 0.15 }}
                        className="absolute right-0 mt-2 w-48 bg-white dark:bg-gray-800 rounded-xl shadow-xl border border-gray-200 dark:border-gray-700 z-50 overflow-hidden"
                    >
                        <div className="py-1">
                            <button
                                onClick={() => handleExport('json')}
                                className="w-full px-4 py-3 flex items-center gap-3 text-sm text-left hover:bg-gray-50 dark:hover:bg-gray-700/50 transition-colors"
                            >
                                <div className="p-2 bg-yellow-100 dark:bg-yellow-900/30 rounded-lg text-yellow-600 dark:text-yellow-400">
                                    <FaFileCode />
                                </div>
                                <div className="flex flex-col">
                                    <span className="font-medium text-gray-700 dark:text-gray-200">JSON</span>
                                    <span className="text-[10px] text-gray-500">Full analysis data</span>
                                </div>
                            </button>

                            <div className="h-px bg-gray-100 dark:bg-gray-700 mx-2" />

                            <button
                                onClick={() => handleExport('csv')}
                                className="w-full px-4 py-3 flex items-center gap-3 text-sm text-left hover:bg-gray-50 dark:hover:bg-gray-700/50 transition-colors"
                            >
                                <div className="p-2 bg-green-100 dark:bg-green-900/30 rounded-lg text-green-600 dark:text-green-400">
                                    <FaFileCsv />
                                </div>
                                <div className="flex flex-col">
                                    <span className="font-medium text-gray-700 dark:text-gray-200">CSV</span>
                                    <span className="text-[10px] text-gray-500">Word-level spreadsheet</span>
                                </div>
                            </button>
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
};
