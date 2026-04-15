import React, { useState, useRef, useEffect } from 'react';
import { FaDownload, FaFileCsv, FaFileCode } from 'react-icons/fa';
import { motion, AnimatePresence } from 'framer-motion';
import { AnalysisResult } from '../../types';
import { exportToJSON, exportToCSV } from '../../utils/exportUtils';
import { useTranslation } from 'react-i18next';
import { useTheme } from '../../context/ThemeContext';
import clsx from 'clsx';

interface ExportButtonProps {
    result: AnalysisResult;
    className?: string;
}

export const ExportButton: React.FC<ExportButtonProps> = ({ result, className }) => {
    const { t } = useTranslation();
    const { isDark } = useTheme();
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
                    "w-12 h-12 flex items-center justify-center rounded-full transition-all duration-300 !bg-transparent",
                    "hover:scale-110",
                    isOpen ? "text-indigo-500" : "text-slate-500 dark:text-slate-400 hover:text-indigo-600 dark:hover:text-indigo-400"
                )}
                title={t('export_results') || "Export Results"}
                aria-label="Export results"
            >
                <FaDownload className="w-5 h-5" />
            </button>

            <AnimatePresence>
                {isOpen && (
                    <motion.div
                        initial={{ opacity: 0, y: 8, scale: 0.95 }}
                        animate={{ opacity: 1, y: 0, scale: 1 }}
                        exit={{ opacity: 0, y: 8, scale: 0.95 }}
                        transition={{ duration: 0.15 }}
                        className="absolute right-0 mt-2 w-52 rounded-sm z-50 overflow-hidden"
                        style={{
                            background: isDark ? 'rgba(5, 5, 5, 0.98)' : 'rgba(255,255,255,0.97)',
                            border: isDark ? '1px solid rgba(255,255,255,0.1)' : '1px solid rgba(203,213,225,0.8)',
                            boxShadow: isDark
                                ? '0 20px 60px rgba(0,0,0,0.5), 0 0 0 1px rgba(255,255,255,0.05)'
                                : '0 20px 60px rgba(0,0,0,0.12), 0 0 0 1px rgba(255,255,255,0.8)',
                            backdropFilter: 'blur(20px)',
                        }}
                    >
                        <div className="p-1.5">
                            <button
                                onClick={() => handleExport('json')}
                                className="w-full px-3 py-2.5 flex items-center gap-3 text-sm text-left rounded-sm transition-all duration-150 !bg-transparent"
                                style={{
                                    color: isDark ? '#e2e8f0' : '#1e293b',
                                    background: 'transparent',
                                    border: 'none',
                                }}
                                onMouseEnter={e => (e.currentTarget.style.background = isDark ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.04)')}
                                onMouseLeave={e => (e.currentTarget.style.background = 'transparent')}
                            >
                                <div className="w-8 h-8 rounded-sm flex items-center justify-center flex-shrink-0"
                                    style={{ background: isDark ? 'rgba(234,179,8,0.2)' : 'rgba(234,179,8,0.12)', color: '#ca8a04' }}>
                                    <FaFileCode className="text-sm" />
                                </div>
                                <div className="flex flex-col">
                                    <span className="font-bold text-sm">JSON</span>
                                    <span className="text-xs" style={{ color: isDark ? '#94a3b8' : '#64748b' }}>
                                        {t('export_full_data') || 'Full analysis data'}
                                    </span>
                                </div>
                            </button>

                            <div style={{ height: '1px', background: isDark ? 'rgba(255,255,255,0.07)' : 'rgba(0,0,0,0.06)', margin: '2px 4px' }} />

                            <button
                                onClick={() => handleExport('csv')}
                                className="w-full px-3 py-2.5 flex items-center gap-3 text-sm text-left rounded-sm transition-all duration-150 !bg-transparent"
                                style={{
                                    color: isDark ? '#e2e8f0' : '#1e293b',
                                    background: 'transparent',
                                    border: 'none',
                                }}
                                onMouseEnter={e => (e.currentTarget.style.background = isDark ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.04)')}
                                onMouseLeave={e => (e.currentTarget.style.background = 'transparent')}
                            >
                                <div className="w-8 h-8 rounded-sm flex items-center justify-center flex-shrink-0"
                                    style={{ background: isDark ? 'rgba(34,197,94,0.2)' : 'rgba(34,197,94,0.12)', color: '#16a34a' }}>
                                    <FaFileCsv className="text-sm" />
                                </div>
                                <div className="flex flex-col">
                                    <span className="font-bold text-sm">CSV</span>
                                    <span className="text-xs" style={{ color: isDark ? '#94a3b8' : '#64748b' }}>
                                        {t('export_word_data') || 'Word-level spreadsheet'}
                                    </span>
                                </div>
                            </button>
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
};
