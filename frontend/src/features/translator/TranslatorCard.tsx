import React, { useState } from 'react';
import { FaArrowRight, FaMagic, FaCheckCircle, FaCopy, FaCheck, FaExclamationTriangle } from 'react-icons/fa';
import { translateText } from '../../services/api';
import classNames from 'classnames';

export const TranslatorCard: React.FC = () => {
    const [inputText, setInputText] = useState('');
    const [translation, setTranslation] = useState<string | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [isLoading, setIsLoading] = useState(false);
    const [isCopied, setIsCopied] = useState(false);

    const charCount = inputText.length;
    const isLimitExceeded = charCount > 500;

    const handleTranslate = async () => {
        if (!inputText.trim()) {
            setError('Please enter some text to translate!');
            setTranslation(null);
            return;
        }

        setIsLoading(true);
        setError(null);
        setTranslation(null);

        try {
            const result = await translateText(inputText);
            setTranslation(result);
        } catch (err: any) {
            setError(err.message || 'An unexpected error occurred.');
        } finally {
            setIsLoading(false);
        }
    };

    const copyToClipboard = () => {
        if (translation) {
            navigator.clipboard.writeText(translation);
            setIsCopied(true);
            setTimeout(() => setIsCopied(false), 2000);
        }
    };

    return (
        <div className="w-full bg-[#0f0f1a]/80 border border-glass-border rounded-3xl overflow-hidden backdrop-blur-xl shadow-[0_40px_80px_rgba(0,0,0,0.5)] ring-1 ring-white/5 animate-fade-in-up">
            {/* Card Header */}
            <div className="px-6 py-6 border-b border-glass-border bg-black/30">
                <div className="flex items-center justify-center gap-4">
                    <LangBadge flagUrl="https://flagcdn.com/w40/us.png" label="English" type="source" />
                    <FaArrowRight className="text-text-muted text-xl animate-arrow-pulse" />
                    <LangBadge flagUrl="https://flagcdn.com/w40/es.png" label="Spanish" type="target" />
                </div>
            </div>

            {/* Card Body */}
            <div className="p-8">
                <div className="relative mb-6">
                    <textarea
                        value={inputText}
                        onChange={(e) => setInputText(e.target.value)}
                        placeholder="Type or paste your English text here..."
                        rows={4}
                        className="w-full min-h-[140px] p-5 pb-10 bg-black/50 border border-glass-border rounded-2xl text-white font-sans text-base leading-relaxed resize-none transition-all duration-300 focus:outline-none focus:border-primary focus:ring-4 focus:ring-primary/15 placeholder:text-white/50"
                    />
                    <span className={classNames("absolute bottom-3 right-4 text-xs font-medium transition-colors duration-300 pointer-events-none", {
                        "text-danger": isLimitExceeded,
                        "text-text-muted": !isLimitExceeded
                    })}>
                        {charCount} / 500
                    </span>
                </div>

                <button
                    onClick={handleTranslate}
                    disabled={isLoading || !inputText.trim()}
                    className={classNames(
                        "w-full py-4 px-8 bg-gradient-to-br from-primary to-[#4f46e5] border-none rounded-xl text-white font-sans text-lg font-semibold cursor-pointer flex items-center justify-center gap-3 transition-all duration-400 relative overflow-hidden shadow-[0_10px_30px_rgba(99,102,241,0.4)] hover:-translate-y-0.5 hover:shadow-[0_15px_40px_rgba(99,102,241,0.5)] active:translate-y-0 disabled:opacity-60 disabled:cursor-not-allowed group",
                        { "opacity-70 pointer-events-none": isLoading }
                    )}
                >
                    {/* Hover overlay */}
                    <span className="absolute inset-0 bg-gradient-to-br from-secondary to-primary opacity-0 transition-opacity duration-300 group-hover:opacity-100" />

                    <span className="relative z-10">Translate Now</span>
                    <span className="relative z-10 transition-transform duration-300 group-hover:rotate-15 group-hover:scale-110">
                        {isLoading ? <div className="loading loading-spinner text-white w-5 h-5"></div> : <FaMagic />}
                    </span>
                </button>

                {/* Output Area (Conditional) */}
                {(translation || error || isLoading) && (
                    <div className="mt-6 p-6 bg-black/40 border border-glass-border rounded-2xl min-h-[80px] text-base leading-relaxed text-text animate-[fadeIn_0.4s_ease-out]">

                        {isLoading && (
                            <div className="flex items-center justify-center gap-4 py-5 text-secondary">
                                <div className="w-6 h-6 border-4 border-secondary/20 border-t-secondary rounded-full animate-spin"></div>
                                <span>Translating your text...</span>
                            </div>
                        )}

                        {error && (
                            <div className="flex items-center gap-3 p-4 bg-danger/10 border border-danger/30 rounded-xl text-red-400">
                                <FaExclamationTriangle className="text-xl" />
                                <span>{error}</span>
                            </div>
                        )}

                        {translation && !isLoading && (
                            <div className="animate-[fadeIn_0.4s_ease-out]">
                                <div className="flex items-center gap-2.5 mb-4 text-success font-semibold">
                                    <FaCheckCircle className="text-lg" />
                                    <span>Translation Complete</span>
                                </div>
                                <div className="p-4 bg-black/20 rounded-xl mb-4 text-lg leading-7">
                                    {translation}
                                </div>
                                <button
                                    onClick={copyToClipboard}
                                    className={classNames(
                                        "inline-flex items-center gap-2 px-5 py-2.5 rounded-lg text-sm font-medium transition-all duration-300 cursor-pointer border",
                                        isCopied
                                            ? "bg-success/15 border-success/50 text-success"
                                            : "bg-primary/15 border-primary/30 text-primary-light hover:bg-primary/25 hover:border-primary hover:-translate-y-0.5"
                                    )}
                                >
                                    {isCopied ? <FaCheck /> : <FaCopy />}
                                    {isCopied ? 'Copied!' : 'Copy'}
                                </button>
                            </div>
                        )}
                    </div>
                )}
            </div>
        </div>
    );
};

interface LangBadgeProps {
    flagUrl: string;
    label: string;
    type: 'source' | 'target';
}

const LangBadge: React.FC<LangBadgeProps> = ({ flagUrl, label, type }) => {
    return (
        <span className={classNames(
            "flex items-center gap-2.5 px-5 py-2.5 bg-white/5 rounded-xl font-medium text-sm text-text border",
            type === 'source' ? "border-primary/30" : "border-secondary/30"
        )}>
            <img src={flagUrl} alt={label} className="w-6 h-4.5 rounded object-cover" />
            {label}
        </span>
    );
};
