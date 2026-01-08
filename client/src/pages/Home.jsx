/* eslint-disable no-unused-vars */
import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { classifyDocument } from '../services/api';
import FileUpload from '../components/FileUpload';
import ResultCard from '../components/ResultCard';
import Loader from '../components/Loader';
import { AlertCircle } from 'lucide-react';

const Home = () => {
    const [file, setFile] = useState(null);
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    const handleFileSelect = (selectedFile) => {
        setFile(selectedFile);
        setResult(null);
        setError(null);
    };

    const handleClear = () => {
        setFile(null);
        setResult(null);
        setError(null);
    };

    const handleClassify = async () => {
        if (!file) return;

        setLoading(true);
        setError(null);

        try {
            const data = await classifyDocument(file);
            setResult(data);
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="min-h-[calc(100vh-theme(spacing.32))] bg-gradient-to-b from-[var(--color-background)] to-white">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12 md:py-20">
                <div className="text-center max-w-3xl mx-auto mb-12">
                    <motion.h1
                        initial={{ opacity: 0, y: -20 }}
                        animate={{ opacity: 1, y: 0 }}
                        className="text-4xl md:text-5xl font-bold text-[var(--color-text-main)] mb-6 tracking-tight"
                    >
                        Intelligent Document Classification
                    </motion.h1>
                    <motion.p
                        initial={{ opacity: 0, y: -20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 0.1 }}
                        className="text-lg text-[var(--color-text-muted)] leading-relaxed"
                    >
                        Upload your documents (PDF, DOCX, TXT) and let our AI instantly categorize them with high precision.
                        Streamline your workflow today.
                    </motion.p>
                </div>

                <div className="flex flex-col items-center gap-8">
                    <FileUpload
                        onFileSelect={handleFileSelect}
                        selectedFile={file}
                        onClear={handleClear}
                    />

                    <AnimatePresence>
                        {error && (
                            <motion.div
                                initial={{ opacity: 0, height: 0 }}
                                animate={{ opacity: 1, height: 'auto' }}
                                exit={{ opacity: 0, height: 0 }}
                                className="w-full max-w-2xl bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg flex items-center gap-2"
                            >
                                <AlertCircle size={20} />
                                <p>{error}</p>
                            </motion.div>
                        )}
                    </AnimatePresence>

                    {file && !loading && !result && (
                        <motion.button
                            initial={{ opacity: 0, y: 10 }}
                            animate={{ opacity: 1, y: 0 }}
                            onClick={handleClassify}
                            className="btn-primary text-lg px-8 py-3 shadow-lg shadow-indigo-500/20"
                        >
                            Classify Document
                        </motion.button>
                    )}

                    {loading && <Loader text="Analyzing document content..." />}

                    <ResultCard result={result} />
                </div>
            </div>
        </div>
    );
};

export default Home;
