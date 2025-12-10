
import React from 'react';
import { useTranslation } from 'react-i18next';
import { motion, AnimatePresence } from 'framer-motion';
import clsx from 'clsx';
// Simple flag icons using ASCII or small SVGs/components would be best, 
// using simple text/emoji for minimize deps or we can use react-icons specific ones if available.
// For now, Emojis are robust and simple for flags.
// 🇺🇸 for EN, 🇹🇷 for TR

export const LanguageSwitcher: React.FC = () => {
    const { i18n } = useTranslation();
    const [isOpen, setIsOpen] = React.useState(false);
    const containerRef = React.useRef<HTMLDivElement>(null);

    const toggleLanguage = (lng: string) => {
        i18n.changeLanguage(lng);
        setIsOpen(false);
        // localStorage is handled by i18next-browser-languagedetector plugin automatically
    };

    // Close on click outside
    React.useEffect(() => {
        const handleClickOutside = (event: MouseEvent) => {
            if (containerRef.current && !containerRef.current.contains(event.target as Node)) {
                setIsOpen(false);
            }
        };
        document.addEventListener('mousedown', handleClickOutside);
        return () => document.removeEventListener('mousedown', handleClickOutside);
    }, []);

    const languages = [
        { code: 'en', label: 'English', flag: '🇺🇸' },
        { code: 'tr', label: 'Türkçe', flag: '🇹🇷' }
    ];

    const currentLang = languages.find(l => l.code === i18n.language) || languages[0];

    return (
        <div className="relative z-50" ref={containerRef}>
            <button
                onClick={() => setIsOpen(!isOpen)}
                className="flex items-center gap-2 px-3 py-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
                aria-label="Change Language"
            >
                <span className="text-xl">{currentLang.flag}</span>
                <span className="text-sm font-medium text-gray-700 dark:text-gray-200 uppercase">{currentLang.code}</span>
            </button>

            <AnimatePresence>
                {isOpen && (
                    <motion.div
                        initial={{ opacity: 0, y: 10, scale: 0.95 }}
                        animate={{ opacity: 1, y: 0, scale: 1 }}
                        exit={{ opacity: 0, y: 10, scale: 0.95 }}
                        transition={{ duration: 0.1 }}
                        className="absolute top-full right-0 mt-2 w-32 bg-white dark:bg-gray-800 rounded-xl shadow-xl border border-gray-200 dark:border-gray-700 overflow-hidden"
                    >
                        {languages.map((lang) => (
                            <button
                                key={lang.code}
                                onClick={() => toggleLanguage(lang.code)}
                                className={clsx(
                                    "w-full flex items-center gap-3 px-4 py-3 text-sm text-left hover:bg-gray-50 dark:hover:bg-gray-700/50 transition-colors",
                                    i18n.language === lang.code && "bg-indigo-50 dark:bg-indigo-900/20 text-indigo-600 dark:text-indigo-400"
                                )}
                            >
                                <span className="text-lg">{lang.flag}</span>
                                <span className="font-medium">{lang.label}</span>
                            </button>
                        ))}
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
};
