/* eslint-disable no-unused-vars */
import React from 'react';
import { motion } from 'framer-motion';
import { CheckCircle, AlertCircle } from 'lucide-react';

const ResultCard = ({ result }) => {
    if (!result) return null;

    const { filename, prediction, confidence, message } = result;
    const confidencePercent = (confidence * 100).toFixed(1);

    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="w-full max-w-2xl mx-auto mt-8"
        >
            <div className="card overflow-hidden border-t-4 border-t-[var(--color-primary)]">
                <div className="flex flex-col md:flex-row gap-6 items-start md:items-center justify-between mb-6">
                    <div>
                        <h3 className="text-sm font-medium text-[var(--color-text-muted)] uppercase tracking-wider mb-1">
                            Classification Result
                        </h3>
                        <div className="flex items-center gap-2">
                            <span className="text-3xl font-bold text-[var(--color-text-main)]">
                                {prediction}
                            </span>
                            <CheckCircle size={24} className="text-[var(--color-success)]" />
                        </div>
                    </div>
                    
                    <div className="flex items-center gap-3 bg-slate-50 px-4 py-2 rounded-lg border border-slate-100">
                        <div className="text-right">
                            <p className="text-xs text-[var(--color-text-muted)] font-medium">Confidence</p>
                            <p className="text-lg font-bold text-[var(--color-primary)]">{confidencePercent}%</p>
                        </div>
                        <div className="w-12 h-12 relative flex items-center justify-center">
                            <svg className="transform -rotate-90 w-full h-full">
                                <circle
                                    cx="24"
                                    cy="24"
                                    r="20"
                                    stroke="currentColor"
                                    strokeWidth="4"
                                    fill="transparent"
                                    className="text-slate-200"
                                />
                                <circle
                                    cx="24"
                                    cy="24"
                                    r="20"
                                    stroke="currentColor"
                                    strokeWidth="4"
                                    fill="transparent"
                                    strokeDasharray={2 * Math.PI * 20}
                                    strokeDashoffset={2 * Math.PI * 20 * (1 - confidence)}
                                    className="text-[var(--color-primary)]"
                                />
                            </svg>
                        </div>
                    </div>
                </div>

                <div className="bg-slate-50 rounded-lg p-4 text-sm text-[var(--color-text-muted)] border border-slate-100">
                    <p>
                        <span className="font-medium text-[var(--color-text-main)]">File:</span> {filename}
                    </p>
                    <p className="mt-1">
                        <span className="font-medium text-[var(--color-text-main)]">Status:</span> {message}
                    </p>
                </div>
            </div>
        </motion.div>
    );
};

export default ResultCard;
